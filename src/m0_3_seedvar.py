"""M0.3d 补验证 1：种子方差快测——同配方（固定 lr=5e-5 续训 500 步）2 个新种子

决策点 10（种子 5→3）稳健性检验。M0.3a 已有 seed 0 同配方数据点：500 步 → 0.485。
若 500 步 test exact 方差超 ±3pp（200 样本噪声带宽），砍种子方案需回补。
"""
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
OUT = "outputs/2026-08-10/m0_3_seedvar"
BS = 128
STEPS = 500
LR = 5e-5
SEEDS = [1, 2]
BASE = {"step_23437": 0.330, "seed0_+500": 0.485}  # M0.3a 已有数据点

os.makedirs(OUT, exist_ok=True)
meta = json.load(open(f"{DATA}/test/dataset.json"))
inputs, labels, ids, idx = load_test_data(DATA, 200, 0)
train_in = np.load(f"{DATA}/train/all__inputs.npy", mmap_mode="r")
train_lb = np.load(f"{DATA}/train/all__labels.npy", mmap_mode="r")


def make_model():
    arch_cfg = yaml.safe_load(open("config/arch/trm.yaml", encoding="utf-8"))
    arch_cfg["puzzle_emb_ndim"] = arch_cfg["hidden_size"]
    arch_cfg.pop("name", None)
    arch_cfg.pop("loss", None)
    arch_cfg.update(dict(batch_size=BS, vocab_size=meta["vocab_size"], seq_len=meta["seq_len"],
                         num_puzzle_identifiers=meta["num_puzzle_identifiers"], causal=False,
                         mlp_t=True, pos_encodings="none"))
    with torch.device("cuda"):
        m = TinyRecursiveReasoningModel_ACTV1(arch_cfg)
    model = ACTLossHead(m, "stablemax_cross_entropy")
    model.load_state_dict(torch.load(CKPT, map_location="cuda"), assign=True)
    return model


results = {}
for seed in SEEDS:
    model = make_model()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1.0,
                            betas=(0.9, 0.95), fused=True)
    rng = np.random.default_rng(seed)
    with torch.device("cuda"):
        carry = model.initial_carry({
            "inputs": torch.zeros(BS, 81, dtype=torch.int32, device="cuda"),
            "labels": torch.zeros(BS, 81, dtype=torch.int32, device="cuda"),
            "puzzle_identifiers": torch.zeros(BS, dtype=torch.int32, device="cuda")})
    t0 = time.time()
    for step in range(STEPS):
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
    ckpt_path = f"{OUT}/seed_{seed}_{STEPS}"
    torch.save(model.state_dict(), ckpt_path)
    del model
    torch.cuda.empty_cache()
    emod = __import__("eval_ptrm").load_model(
        ckpt_path, "config/arch/trm.yaml", meta["vocab_size"], meta["seq_len"],
        meta["num_puzzle_identifiers"])
    emod.config.halt_max_steps = 16
    pred, _ = ptrm_infer(emod, inputs, ids, 1, 16, 0.0)
    em = (pred == labels).all(dim=1).float().mean().item()
    ca = (pred == labels).float().mean().item()
    results[seed] = {"exact": em, "cell": ca, "loss_end": float(loss.item() / BS)}
    print(f"seed {seed} +500 steps: exact={em:.4f} cell={ca:.4f} "
          f"loss={loss.item()/BS:.4f} ({time.time()-t0:.0f}s)", flush=True)

json.dump({"base": BASE, "results": results, "steps": STEPS, "lr": LR},
          open(f"{OUT}/result.json", "w"), indent=2)
print("saved -> " + f"{OUT}/result.json", flush=True)
