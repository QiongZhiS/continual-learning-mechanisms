"""PTRM inference: K parallel rollouts + Gaussian noise + q-head selection (M0.1 K-curve)

Usage (from repo root):
    python eval_ptrm.py --ckpt outputs/<run>/step_<N> \
        --data ../data/sudoku-extreme-1k-aug-100 \
        --K 1 10 100 --D 48 --sigma 0.2 --n 200

Baseline: K=1 D=16 sigma=0 (standard eval, matches training config)
PTRM: K=10/100 D=48 sigma=0.2 (upstream PTRM paper PPBench config)"""
import argparse
import json
import os
import time
import yaml

import numpy as np
import torch

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1


def load_model(ckpt_path, arch_yaml, vocab_size, seq_len, n_identifiers):
    arch_cfg = yaml.safe_load(open(arch_yaml, encoding="utf-8"))
    # hydra ${.hidden_size} syntax is not parsed by yaml -> expand manually; name/loss are not Config fields
    if isinstance(arch_cfg.get("puzzle_emb_ndim"), str):
        arch_cfg["puzzle_emb_ndim"] = arch_cfg["hidden_size"]
    arch_cfg.pop("name", None)
    arch_cfg.pop("loss", None)
    arch_cfg.update(dict(
        batch_size=1, vocab_size=vocab_size, seq_len=seq_len,
        num_puzzle_identifiers=n_identifiers, causal=False,
        mlp_t=True, pos_encodings="none",  # M0.1 uses the MLP variant (upstream paper sudoku config)
    ))
    with torch.device("cuda"):
        model = TinyRecursiveReasoningModel_ACTV1(arch_cfg)
    state = torch.load(ckpt_path, map_location="cuda")
    # checkpoint saves the full model with loss head -> keys carry the model. prefix
    state = {k.removeprefix("model."): v for k, v in state.items()}
    model.load_state_dict(state, assign=True)
    model.eval()
    return model


def load_test_data(data_dir, n, seed):
    t = os.path.join(data_dir, "test")
    inputs = np.load(os.path.join(t, "all__inputs.npy"), mmap_mode="r")
    labels = np.load(os.path.join(t, "all__labels.npy"), mmap_mode="r")
    ids = np.load(os.path.join(t, "all__puzzle_identifiers.npy"), mmap_mode="r")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(inputs), n, replace=False))
    return (
        torch.tensor(np.asarray(inputs[idx], dtype=np.int32), device="cuda"),
        torch.tensor(np.asarray(labels[idx], dtype=np.int32), device="cuda"),
        torch.tensor(np.asarray(ids[idx], dtype=np.int32), device="cuda"),
        idx,
    )


def ptrm_infer(model, inputs, ids, K, D, sigma, chunk_seq=512, save_latents=None):
    """K parallel rollouts; inject N(0, sigma^2) noise each deep step; q-head selects the best rollout.

    Returns: best_pred [N, 81] (argmax class, same encoding as labels)
    """
    N = inputs.shape[0]
    puzzles_per_chunk = max(1, chunk_seq // K)
    best_pred = torch.zeros(N, 81, dtype=torch.long, device="cuda")
    latents = {}  # {puzzle_i: {step: z_H_first [512]}}

    with torch.inference_mode():
        for start in range(0, N, puzzles_per_chunk):
            b = inputs[start:start + puzzles_per_chunk]
            bs = b.shape[0]
            batch = {
                "inputs": b.repeat(K, 1),
                "puzzle_identifiers": ids[start:start + bs].repeat(K),
            }
            # empty_carry placeholder z created on the default device -> must call inside a cuda context
            with torch.device("cuda"):
                carry = model.initial_carry(batch)
            for step in range(D):
                if sigma > 0:
                    carry.inner_carry.z_L = carry.inner_carry.z_L + \
                        sigma * torch.randn_like(carry.inner_carry.z_L)
                carry, outputs = model(carry, batch)
                if save_latents is not None and start == 0 and step in save_latents:
                    q = outputs["q_halt_logits"].view(K, bs)
                    pick = q.max(0).indices  # rollout with max q at this step
                    for pi in range(bs):
                        # q-head reads z_H[:, 0] (first puzzle-emb slot) -> M0.2 basins visualize the output latent
                        latents.setdefault(pi, {})[step] = \
                            carry.inner_carry.z_H[pick[pi], 0].float().cpu().numpy()
            q = outputs["q_halt_logits"].view(K, bs)
            preds = outputs["logits"].argmax(-1).view(K, bs, 81)
            qmax, qidx = q.max(0)
            best_pred[start:start + bs] = preds[qidx, torch.arange(bs, device="cuda")]
    return best_pred, latents


def exact_match_acc(pred, labels):
    return (pred == labels).all(dim=1).float().mean().item()


def cell_acc(pred, labels):
    return (pred == labels).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="../data/sudoku-extreme-1k-aug-100")
    ap.add_argument("--arch-yaml", default="config/arch/trm.yaml")
    ap.add_argument("--K", nargs="+", type=int, default=[1, 10, 100])
    ap.add_argument("--D", type=int, default=48)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-latents", default=None, help="steps to save z_H latents, e.g. '0,8,16,24,32,40,47'")
    ap.add_argument("--out", default=None, help="result json path")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.data, "test", "dataset.json")))
    inputs, labels, ids, idx = load_test_data(args.data, args.n, args.seed)
    model = load_model(args.ckpt, args.arch_yaml, meta["vocab_size"], meta["seq_len"],
                       meta["num_puzzle_identifiers"])

    results = {"ckpt": args.ckpt, "n_puzzles": args.n, "seed": args.seed,
               "curves": {}, "deviation_note": "AdamW replaced adam-atan2; data 10x smaller; steps ~15x fewer"}

    save_steps = None
    if args.save_latents:
        save_steps = [int(s) for s in args.save_latents.split(",")]

    for K in args.K:
        # K=1 -> standard eval baseline (D=16, sigma=0, matches training config); K>1 -> PTRM config (--D/--sigma)
        D = 16 if K == 1 else args.D
        sigma = 0.0 if K == 1 else args.sigma
        t0 = time.time()
        model.config.halt_max_steps = D
        pred, latents = ptrm_infer(model, inputs, ids, K, D, sigma,
                                   save_latents=save_steps)
        dt = time.time() - t0
        em = exact_match_acc(pred, labels)
        ca = cell_acc(pred, labels)
        results["curves"][f"K={K}"] = {
            "D": D, "sigma": sigma,
            "exact_match": round(em, 4), "cell_acc": round(ca, 4),
            "seconds": round(dt, 1), "gpu_mem_MB": torch.cuda.max_memory_allocated() // 1048576,
        }
        print(f"K={K} D={D} sigma={sigma}: exact={em:.4f} cell={ca:.4f} ({dt:.0f}s)", flush=True)
        torch.cuda.reset_peak_memory_stats()

    if latents:
        np.savez(os.path.join(os.path.dirname(args.out) if args.out else ".",
                              "latents_zH.npz"),
                 **{f"p{pi}_s{st}": v for pi, steps in latents.items()
                    for st, v in steps.items()})

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
