"""M1c q 分离度（E7b 触发前提）：域 B 上 q-head 是否像域 A 一样饱和？

域 A（M0.2）：q AUC=1.0 完美分离正确/错误 rollout → 无冲突事件空间 → E7b 无测量空间。
域 B 检查：若同样饱和（AUC≈1.0 + 双类谜题比例低）→ E7b 前提失败跨域成立，降级；
若分离弱（AUC<1 或双类谜题多）→ 冲突事件空间存在 → E7b 前提成立。
采集：K=50 D=48 σ=0.2，n=100 谜题，最终步 q 值 + 每 rollout 结果位正确性。
"""
import json
import os
import time

import numpy as np
import torch

from eval_ptrm import load_test_data, load_model

CKPT = "outputs/2026-08-10/m1_domainB/step_4000"
DATA = "../data/domain-b-rpn"
OUT = "outputs/2026-08-10/m1_domainB/q_sep.json"
K, D, SIGMA, N = 30, 48, 0.2, 80

meta = json.load(open(f"{DATA}/dataset.json"))
inputs, labels, ids, idx = load_test_data(DATA, N, 0)
model = load_model(CKPT, "config/arch/trm.yaml", meta["vocab_size"], meta["seq_len"],
                   meta["num_puzzle_identifiers"])
model.config.halt_max_steps = D

per_puzzle = []
with torch.inference_mode():
    for pi in range(N):
        b = inputs[pi:pi + 1].repeat(K, 1)
        pid = ids[pi:pi + 1].repeat(K)
        batch = {"inputs": b, "puzzle_identifiers": pid}
        with torch.device("cuda"):
            carry = model.initial_carry(batch)
        for _ in range(D):
            if SIGMA > 0:
                carry.inner_carry.z_L = carry.inner_carry.z_L + \
                    SIGMA * torch.randn_like(carry.inner_carry.z_L)
            carry, outputs = model(carry, batch)
        q = outputs["q_halt_logits"].view(K).float().cpu().numpy()
        pred_ok = (outputs["logits"].argmax(-1)[:, 0] == labels[pi, 0]).cpu().numpy()
        per_puzzle.append({"q": q.tolist(), "ok": pred_ok.tolist()})

# 聚合
n_both = n_only_good = n_only_bad = 0
aucs, margins = [], []
for m in per_puzzle:
    q = np.asarray(m["q"])
    ok = np.asarray(m["ok"], dtype=bool)
    if ok.all():
        n_only_good += 1
    elif (~ok).all():
        n_only_bad += 1
    else:
        n_both += 1
        q_good, q_bad = q[ok], q[~ok]
        aucs.append(float((q_good[:, None] > q_bad[None, :]).mean()))  # 秩比较
        margins.append(float(q_good.mean() - q_bad.mean()))

summary = {
    "n_puzzles": N, "K": K, "D": D, "sigma": SIGMA,
    "n_both": n_both, "n_only_good": n_only_good, "n_only_bad": n_only_bad,
    "both_ratio": round(n_both / N, 3),
    "auc_mean": round(float(np.mean(aucs)), 4) if aucs else None,
    "auc_min": round(float(np.min(aucs)), 4) if aucs else None,
    "margin_mean": round(float(np.mean(margins)), 4) if margins else None,
    "note": "域 A 参照 (M0.2): AUC=1.0, 双类 4/40, margin +5.1 vs -9.2",
}
print(json.dumps(summary, indent=2), flush=True)
json.dump({"summary": summary, "per_puzzle": per_puzzle},
          open(OUT, "w"), indent=2)
print("saved -> " + OUT, flush=True)
