"""M1c E1 main effect: domain-B K-curve (does the exploration mechanism hold in the symbolic domain?)

Config same source as M0.1: test 500 puzzles (first 500 of the domain-B test set),
K=1 D=16 sigma=0 (standard baseline) · K=1 D=48 sigma=0 (noise-free deep expansion) · K=10/100 D=48 sigma=0.2 (PTRM).
exact = output slot 0 (result slot) accuracy (domain-B semantics).
Criterion: K=10/100 significantly better than K=1 -> the exploration mechanism holds across domains (E1 main criterion)."""
import json
import os
import sys
import time

import numpy as np
import torch

from eval_ptrm import load_test_data, load_model, ptrm_infer

CKPT = sys.argv[1] if len(sys.argv) > 1 else "outputs/2026-08-10/m1_domainB/step_4000"
DATA = "../data/domain-b-rpn"
OUT = sys.argv[2] if len(sys.argv) > 2 else "outputs/2026-08-10/m1_domainB/k_curve.json"
N = 500

meta = json.load(open(f"{DATA}/dataset.json"))
inputs, labels, ids, idx = load_test_data(DATA, N, 0)
model = load_model(CKPT, "config/arch/trm.yaml", meta["vocab_size"], meta["seq_len"],
                   meta["num_puzzle_identifiers"])

results = {}
for K, D, sigma in [(1, 16, 0.0), (1, 48, 0.0), (10, 48, 0.2), (100, 48, 0.2)]:
    model.config.halt_max_steps = D
    t0 = time.time()
    pred, _ = ptrm_infer(model, inputs, ids, K, D, sigma)
    exact = (pred[:, 0] == labels[:, 0]).float().mean().item()  # result slot
    cell = (pred == labels).float().mean().item()
    dt = time.time() - t0
    results[f"K={K}_D={D}_s={sigma}"] = {"exact": round(exact, 4), "cell": round(cell, 4),
                                         "seconds": round(dt, 1)}
    print(f"K={K} D={D} sigma={sigma}: exact={exact:.4f} cell={cell:.4f} ({dt:.0f}s)",
          flush=True)
    torch.cuda.reset_peak_memory_stats()

results["ckpt"] = CKPT
results["n"] = N
json.dump(results, open(OUT, "w"), indent=2)
print("saved -> " + OUT, flush=True)
