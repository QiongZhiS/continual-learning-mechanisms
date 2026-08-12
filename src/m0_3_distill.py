"""M0.3c minimal 7M distillation test: does SkillsBench's "self-written skill null gain" reproduce at small scale?

Design (same-sample two-group control):
- sample the first 2000 train puzzles; use the pretrained checkpoint (K=1 D=16) to produce the model's own outputs
- group 1 (self-distill): labels = model's own outputs (self-teaching, no new information)
- group 2 (human): labels = ground truth (corrections of model errors = external information)
- group 3: no fine-tuning (baseline 0.33)
Each group fine-tuned 500 steps (same budget/data/seed) -> compare on 200 test puzzles.

SkillsBench pattern reproduced (E3 motivation holds) = group1 ~= baseline and group2 > group1."""
import json
import os
import time

import numpy as np
import torch

from eval_ptrm import load_test_data, load_model, ptrm_infer
from models.losses import ACTLossHead
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

CKPT = "checkpoints/Sudoku-extreme-1k-aug-100-ACT-torch/TinyRecursiveReasoningModel_ACTV1 precise-muskox/step_23437"
DATA = "../data/sudoku-extreme-1k-aug-100"
OUT = "outputs/2026-08-10/m0_3_distill"
N_SAMPLE = 2000
BS = 128
TOTAL = 500
LR = 1e-4

os.makedirs(OUT, exist_ok=True)
meta = json.load(open(f"{DATA}/test/dataset.json"))
inputs, labels, ids, idx = load_test_data(DATA, 200, 0)

train_in = np.load(f"{DATA}/train/all__inputs.npy", mmap_mode="r")
train_lb = np.load(f"{DATA}/train/all__labels.npy", mmap_mode="r")
X = np.asarray(train_in[:N_SAMPLE], dtype=np.int32)      # [N, 81]
Y_gt = np.asarray(train_lb[:N_SAMPLE], dtype=np.int32)   # [N, 81] human labels

# ---- generate model self-outputs (K=1 D=16 standard inference) ----
gen_model = load_model(CKPT, "config/arch/trm.yaml", meta["vocab_size"], meta["seq_len"],
                       meta["num_puzzle_identifiers"])
gen_model.config.halt_max_steps = 16
t0 = time.time()
Y_self = []
with torch.inference_mode():
    for start in range(0, N_SAMPLE, 128):
        b = torch.tensor(X[start:start + 128], device="cuda")
        batch = {"inputs": b,
                 "puzzle_identifiers": torch.zeros(b.shape[0], dtype=torch.int32, device="cuda")}
        with torch.device("cuda"):
            carry = gen_model.initial_carry(batch)
        for _ in range(16):
            carry, outputs = gen_model(carry, batch)
        Y_self.append(outputs["logits"].argmax(-1).cpu().numpy())
Y_self = np.concatenate(Y_self)
ok_rate = (Y_self == Y_gt).all(-1).mean()
print(f"self-output generated: {N_SAMPLE} samples, exact-ok rate {ok_rate:.3f} "
      f"({time.time()-t0:.0f}s)", flush=True)
del gen_model
torch.cuda.empty_cache()


def make_model():
    arch_cfg = __import__("yaml").safe_load(open("config/arch/trm.yaml", encoding="utf-8"))
    arch_cfg["puzzle_emb_ndim"] = arch_cfg["hidden_size"]
    arch_cfg.pop("name", None)
    arch_cfg.pop("loss", None)
    arch_cfg.update(dict(batch_size=BS, vocab_size=meta["vocab_size"], seq_len=meta["seq_len"],
                         num_puzzle_identifiers=meta["num_puzzle_identifiers"], causal=False,
                         mlp_t=True, pos_encodings="none"))
    with torch.device("cuda"):
        m = TinyRecursiveReasoningModel_ACTV1(arch_cfg)
    m = ACTLossHead(m, "stablemax_cross_entropy")
    m.load_state_dict(torch.load(CKPT, map_location="cuda"), assign=True)  # checkpoint carries the model. prefix
    m.train()
    return m


def train_eval(Y_target, tag):
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1.0,
                            betas=(0.9, 0.95), fused=True)
    rng = np.random.default_rng(7)
    t0 = time.time()
    with torch.device("cuda"):
        carry = model.initial_carry({
            "inputs": torch.zeros(BS, 81, dtype=torch.int32, device="cuda"),
            "labels": torch.zeros(BS, 81, dtype=torch.int32, device="cuda"),
            "puzzle_identifiers": torch.zeros(BS, dtype=torch.int32, device="cuda")})
    for step in range(TOTAL):
        idxb = rng.choice(N_SAMPLE, BS, replace=False)
        batch = {
            "inputs": torch.tensor(X[idxb], device="cuda"),
            "labels": torch.tensor(Y_target[idxb], device="cuda"),
            "puzzle_identifiers": torch.zeros(BS, dtype=torch.int32, device="cuda"),
        }
        opt.zero_grad()
        carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
        (loss / BS).backward()
        opt.step()
    # eval must use the unwrapped model: ptrm_infer unpacks model(carry, batch) and reads outputs["logits"]
    ckpt_path = f"{OUT}/{tag}_{TOTAL}"
    torch.save(model.state_dict(), ckpt_path)
    model.eval()
    del model
    torch.cuda.empty_cache()
    emod = __import__("eval_ptrm").load_model(
        ckpt_path, "config/arch/trm.yaml", meta["vocab_size"], meta["seq_len"],
        meta["num_puzzle_identifiers"])
    emod.config.halt_max_steps = 16
    pred, _ = ptrm_infer(emod, inputs, ids, 1, 16, 0.0)
    em = (pred == labels).all(dim=1).float().mean().item()
    ca = (pred == labels).float().mean().item()
    print(f"[{tag}] {TOTAL} steps: exact={em:.4f} cell={ca:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return {"tag": tag, "exact": em, "cell": ca}


results = [train_eval(Y_self, "self-distill"),
           train_eval(Y_gt, "human-label")]
print("baseline exact=0.3300", flush=True)
json.dump({"self_ok_rate": float(ok_rate), "results": results, "baseline": 0.33,
           "steps": TOTAL, "n_sample": N_SAMPLE},
          open(f"{OUT}/result.json", "w"), indent=2)
print(f"saved -> {OUT}/result.json", flush=True)
