"""M1b domain-B training: TRM from scratch on RPN stack-machine arithmetic

- from-scratch init (no checkpoint), fixed lr 1e-4 without decay (official yaml hyperparams, avoids the cosine-decay frozen-warmup pitfall)
- batch 128 · wd 0.1 · betas (0.9,0.95) · fused AdamW (same as the M0.1 local patch)
- mlp_t=True, pos_encodings="none" (identical to the eval_ptrm load config, no hidden mismatch)
- eval: K=1 D=16, exact = output slot 0 accuracy (result slot), cell = mean over all 81 slots
- saves a checkpoint + eval every EVAL_EVERY steps, results written to json
--aug-batch (round-3 discriminator): per-sample uniform sampling of equivalence-class forms (ADD-subtree-swap masks,
including mask=0 original form) — dataset/size/pass/steps/compute all matched to the baseline,
the only variable = training distribution (canonical form vs uniform over equivalence classes). Swap function reused from m1_gen_domainB.
Usage: python m1_train_domainB.py --steps 4000 [--lr 1e-4] [--out output-dir] [--aug-batch]"""
import argparse
import json
import os
import time

import numpy as np
import torch
import yaml

from m1_gen_domainB import (PAD, apply_swap_mask, collect_adds, parse_seq,
                            serialize)
from models.losses import ACTLossHead
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

# PTRM_DATA override: round-3 augmented dataset (unchanged by default, zero behavior impact)
DATA = os.environ.get("PTRM_DATA", "../data/domain-b-rpn")
BS = 128
EVAL_EVERY = 500


def load_eval_set(data_dir, n=2000, split="test", seed=0):
    """split=test: first n novel structures; split=train: fixed-seed sample of n train samples
    (train-side ceiling check: train acc vs test acc = generalization gap)"""
    t = os.path.join(data_dir, split)
    inputs = np.load(os.path.join(t, "all__inputs.npy"), mmap_mode="r")
    labels = np.load(os.path.join(t, "all__labels.npy"), mmap_mode="r")
    ids = np.load(os.path.join(t, "all__puzzle_identifiers.npy"), mmap_mode="r")
    if split == "train":
        idx = np.sort(np.random.default_rng(seed).choice(len(inputs), n, replace=False))
        inputs, labels, ids = inputs[idx], labels[idx], ids[idx]
    return (torch.tensor(np.asarray(inputs[:n], dtype=np.int32), device="cuda"),
            torch.tensor(np.asarray(labels[:n], dtype=np.int32), device="cuda"),
            torch.tensor(np.asarray(ids[:n], dtype=np.int32), device="cuda"))


def aug_batch_forms(rows, rng):
    """Round-3 discriminator: uniformly sample one equivalence-class form per row (ADD-subtree-swap mask, incl. mask=0
    original form). Rows = RPN programs left-aligned + PAD fill; output length unchanged (swaps do not change
    the token multiset)."""
    out = np.empty_like(rows)
    for i, row in enumerate(rows):
        seq = row[row != PAD].tolist()
        tree = parse_seq(seq)
        c = 1 << len(collect_adds(tree, []))
        m = int(rng.integers(c))
        if m:
            t = apply_swap_mask(tree, m)
            s = np.asarray(serialize(t), dtype=rows.dtype)
            out[i, :len(s)] = s
            out[i, len(s):] = PAD
        else:
            out[i] = row
    return out


def eval_result(model, inputs, labels, ids):
    """K=1 D=16 standard eval; exact = result-slot (slot 0) accuracy"""
    emod = model  # unwrapped
    emod.config.halt_max_steps = 16
    with torch.inference_mode():
        pred = torch.zeros(inputs.shape[0], 81, dtype=torch.long, device="cuda")
        for start in range(0, inputs.shape[0], 128):
            b = inputs[start:start + 128]
            bs = b.shape[0]
            batch = {"inputs": b, "puzzle_identifiers": ids[start:start + bs]}
            with torch.device("cuda"):
                carry = emod.initial_carry(batch)
            for _ in range(16):
                carry, outputs = emod(carry, batch)
            pred[start:start + bs] = outputs["logits"].argmax(-1)
    exact = (pred[:, 0] == labels[:, 0]).float().mean().item()
    cell = (pred == labels).float().mean().item()
    return exact, cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="outputs/2026-08-10/m1_domainB")
    ap.add_argument("--resume", default=None, help="checkpoint path to resume from")
    ap.add_argument("--start-step", type=int, default=0)
    ap.add_argument("--aug-batch", action="store_true",
                    help="round-3 discriminator: per-sample uniform equivalence-class sampling (matched size/steps/compute)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    meta = json.load(open(f"{DATA}/dataset.json"))
    arch_cfg = yaml.safe_load(open("config/arch/trm.yaml", encoding="utf-8"))
    arch_cfg["puzzle_emb_ndim"] = arch_cfg["hidden_size"]
    arch_cfg.pop("name", None)
    arch_cfg.pop("loss", None)
    arch_cfg.update(dict(batch_size=BS, vocab_size=meta["vocab_size"], seq_len=meta["seq_len"],
                         num_puzzle_identifiers=meta["num_puzzle_identifiers"], causal=False,
                         mlp_t=True, pos_encodings="none"))

    with torch.device("cuda"):
        m = TinyRecursiveReasoningModel_ACTV1(arch_cfg)
    model = ACTLossHead(m, "stablemax_cross_entropy")  # wrapped training
    if args.resume:
        # checkpoint carries the model. prefix; wrapped model expects the same prefix -> load directly
        model.load_state_dict(torch.load(args.resume, map_location="cuda"), assign=True)
        print(f"resumed from {args.resume}", flush=True)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                            betas=(0.9, 0.95), fused=True)

    train_in = np.load(f"{DATA}/train/all__inputs.npy", mmap_mode="r")
    train_lb = np.load(f"{DATA}/train/all__labels.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    eval_in, eval_lb, eval_ids = load_eval_set(DATA)
    tr_in, tr_lb, tr_ids = load_eval_set(DATA, n=2000, split="train")  # train-side ceiling check

    with torch.device("cuda"):
        carry = model.initial_carry({
            "inputs": torch.zeros(BS, 81, dtype=torch.int32, device="cuda"),
            "labels": torch.zeros(BS, 81, dtype=torch.int32, device="cuda"),
            "puzzle_identifiers": torch.zeros(BS, dtype=torch.int32, device="cuda")})

    results = []
    t0 = time.time()
    for step in range(args.start_step, args.steps + 1, EVAL_EVERY):
        if step > args.start_step:
            for _ in range(EVAL_EVERY):
                idxb = rng.choice(len(train_in), BS, replace=False)
                train_x = np.asarray(train_in[idxb], dtype=np.int32)
                if args.aug_batch:
                    train_x = aug_batch_forms(train_x, rng)
                batch = {
                    "inputs": torch.tensor(train_x, device="cuda"),
                    "labels": torch.tensor(np.asarray(train_lb[idxb], dtype=np.int32),
                                           device="cuda"),
                    "puzzle_identifiers": torch.zeros(BS, dtype=torch.int32, device="cuda"),
                }
                opt.zero_grad()
                carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
                (loss / BS).backward()
                opt.step()
            ckpt_path = f"{args.out}/step_{step}"
            torch.save(model.state_dict(), ckpt_path)
            # eval uses the unwrapped model (rebuilt from checkpoint, end-to-end verification)
            m2 = TinyRecursiveReasoningModel_ACTV1(arch_cfg)
            state = {k.removeprefix("model."): v for k, v in torch.load(ckpt_path,
                                                                        map_location="cuda").items()}
            m2.load_state_dict(state, assign=True)
            m2.eval()
            with torch.device("cuda"):
                m2.to("cuda")
            exact, cell = eval_result(m2, eval_in, eval_lb, eval_ids)
            tr_exact, tr_cell = eval_result(m2, tr_in, tr_lb, tr_ids)
            results.append({"step": step, "exact": exact, "cell": cell,
                            "train_exact": tr_exact, "train_cell": tr_cell,
                            "loss": float(loss.item() / BS)})
            print(f"step {step}: exact={exact:.4f} train_exact={tr_exact:.4f} "
                  f"cell={cell:.4f} loss={loss.item()/BS:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            json.dump(results, open(f"{args.out}/result.json", "w"), indent=2)
            del m2
            torch.cuda.empty_cache()

    print("done", flush=True)


if __name__ == "__main__":
    main()
