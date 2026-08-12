"""M0.3a domain-A ceiling calibration: continue training from the pretrained checkpoint for 2000 steps, save a checkpoint every 500 steps + K=1 D=16 eval

Produces steps-vs-accuracy saturation curve points (x = global step 23437+500k), joined with existing points (5468->?, 23437->0.33).
Fixed lr=5e-5 without warmup (clean continuation, avoids pretrain.py load_checkpoint restarting warmup from step 0)."""
import json
import os
import time

import numpy as np
import torch
import yaml

from eval_ptrm import load_test_data, ptrm_infer
from models.losses import ACTLossHead
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

CKPT = "checkpoints/Sudoku-extreme-1k-aug-100-ACT-torch/TinyRecursiveReasoningModel_ACTV1 precise-muskox/step_23437"
DATA = "../data/sudoku-extreme-1k-aug-100"
OUT = "outputs/2026-08-10/m0_3_curve"
BASE_STEP = 23437
BS = 128
TOTAL = 2000
EVAL_EVERY = 500
LR = 5e-5

os.makedirs(OUT, exist_ok=True)
meta = json.load(open(f"{DATA}/test/dataset.json"))
arch_cfg = yaml.safe_load(open("config/arch/trm.yaml", encoding="utf-8"))
arch_cfg["puzzle_emb_ndim"] = arch_cfg["hidden_size"]
arch_cfg.pop("name", None)
arch_cfg.pop("loss", None)
arch_cfg.update(dict(batch_size=BS, vocab_size=meta["vocab_size"], seq_len=meta["seq_len"],
                     num_puzzle_identifiers=meta["num_puzzle_identifiers"], causal=False,
                     mlp_t=True, pos_encodings="none"))

with torch.device("cuda"):
    model = TinyRecursiveReasoningModel_ACTV1(arch_cfg)
model = ACTLossHead(model, "stablemax_cross_entropy")
# checkpoint carries the model. prefix; target = ACTLossHead-wrapped model (also expects model. prefix) -> use directly
state = torch.load(CKPT, map_location="cuda")
model.load_state_dict(state, assign=True)
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1.0,
                        betas=(0.9, 0.95), fused=True)

train_in = np.load(f"{DATA}/train/all__inputs.npy", mmap_mode="r")
train_lb = np.load(f"{DATA}/train/all__labels.npy", mmap_mode="r")
rng = np.random.default_rng(0)

# eval data (K=1 D=16, same source as the M0.1 curve, 200 puzzles seed=0)
inputs, labels, ids, idx = load_test_data(DATA, 200, 0)

# placeholder batch must contain labels: ACTLossHead takes loss targets from new_carry.current_data["labels"]
with torch.device("cuda"):
    carry = model.initial_carry({
        "inputs": torch.zeros(BS, 81, dtype=torch.int32, device="cuda"),
        "labels": torch.zeros(BS, 81, dtype=torch.int32, device="cuda"),
        "puzzle_identifiers": torch.zeros(BS, dtype=torch.int32, device="cuda")})

results = []
t0 = time.time()
for step in range(0, TOTAL + 1, EVAL_EVERY):
    if step > 0:
        for _ in range(EVAL_EVERY):
            idxb = rng.choice(len(train_in), BS, replace=False)
            batch = {
                "inputs": torch.tensor(np.asarray(train_in[idxb], dtype=np.int32), device="cuda"),
                "labels": torch.tensor(np.asarray(train_lb[idxb], dtype=np.int32), device="cuda"),
                "puzzle_identifiers": torch.zeros(BS, dtype=torch.int32, device="cuda"),
            }
            opt.zero_grad()
            carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
            (loss / BS).backward()
            opt.step()
        ckpt_path = f"{OUT}/step_{BASE_STEP + step}"
        torch.save(model.state_dict(), ckpt_path)
        # eval: rebuild the unwrapped model (end-to-end verification that the checkpoint loads)
        em, ca = None, None
        try:
            emod = __import__("eval_ptrm").load_model(
                ckpt_path, "config/arch/trm.yaml", meta["vocab_size"], meta["seq_len"],
                meta["num_puzzle_identifiers"])
            emod.config.halt_max_steps = 16
            pred, _ = ptrm_infer(emod, inputs, ids, 1, 16, 0.0)
            em = (pred == labels).all(dim=1).float().mean().item()
            ca = (pred == labels).float().mean().item()
        except Exception as e:
            print(f"eval failed at {BASE_STEP+step}: {e}", flush=True)
        results.append({"step": BASE_STEP + step, "exact": em, "cell": ca})
        print(f"step {BASE_STEP+step}: exact={em} cell={ca} "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)
        json.dump(results, open(f"{OUT}/curve.json", "w"), indent=2)

print("done", flush=True)
