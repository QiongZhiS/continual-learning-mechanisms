"""M1 选择器修复实验：结果一致性投票 vs q 选择（域 B，E1 失败分支行动 A）

瓶颈假设：q-head 跨域不泛化（域 B AUC 0.57）——rollout 里信息足够但 q 选不出。
判别：同一批 rollouts（K=30 D=48 σ=0.2，500 test 谜题，与 K 曲线同源）上，
  q 选择（现状 0.476）vs 纯多数表决 vs 表决+q tiebreak ——
  vote > q 选择 → 假设成立（选择器修复有效，探索机制有路径）；
  vote ≈ q 选择 → 假设证伪（采样信息不足，需语义级探索）。
"""
import json
import os
import time

import numpy as np
import torch

from eval_ptrm import load_test_data, load_model

CKPT = "outputs/2026-08-10/m1_domainB/step_4000"
DATA = "../data/domain-b-rpn"
OUT = "outputs/2026-08-10/m1_domainB/selector_fix.json"
K, D, SIGMA, N = 30, 48, 0.2, 500
CHUNK = 10  # 每批谜题数（CHUNK*K 序列）

meta = json.load(open(f"{DATA}/dataset.json"))
inputs, labels, ids, idx = load_test_data(DATA, N, 0)
model = load_model(CKPT, "config/arch/trm.yaml", meta["vocab_size"], meta["seq_len"],
                   meta["num_puzzle_identifiers"])
model.config.halt_max_steps = D

pred0 = np.zeros((N, K), dtype=np.int64)  # 结果位预测 [谜题, rollout]
qs = np.zeros((N, K), dtype=np.float32)   # 最终步 q 值
t0 = time.time()
with torch.inference_mode():
    for start in range(0, N, CHUNK):
        bs = min(CHUNK, N - start)
        b = inputs[start:start + bs].repeat_interleave(K, 0)  # 谜题内 rollout 分组
        pid = ids[start:start + bs].repeat_interleave(K)
        batch = {"inputs": b, "puzzle_identifiers": pid}
        with torch.device("cuda"):
            carry = model.initial_carry(batch)
        for _ in range(D):
            if SIGMA > 0:
                carry.inner_carry.z_L = carry.inner_carry.z_L + \
                    SIGMA * torch.randn_like(carry.inner_carry.z_L)
            carry, outputs = model(carry, batch)
        p = outputs["logits"].argmax(-1)[:, 0].view(bs, K).cpu().numpy()
        q = outputs["q_halt_logits"].view(bs, K).float().cpu().numpy()
        pred0[start:start + bs] = p
        qs[start:start + bs] = q
print(f"rollouts collected ({time.time()-t0:.0f}s)", flush=True)

labels0 = labels[:, 0].cpu().numpy()

def acc(selected):
    return (selected == labels0).mean()

# 1. q 选择（现状）
q_sel = pred0[np.arange(N), qs.argmax(1)]
# 2. 纯多数表决（tie 随机）
rng = np.random.default_rng(0)
vote = np.array([np.bincount(pred0[i], minlength=13).argmax() for i in range(N)])
# 3. 表决 + q tiebreak：多数集合内取 q 最大
vote_q = np.zeros(N, dtype=np.int64)
for i in range(N):
    vals, counts = np.unique(pred0[i], return_counts=True)
    m = counts.max()
    cands = vals[counts == m]
    if len(cands) == 1:
        vote_q[i] = cands[0]
    else:
        vote_q[i] = cands[qs[i][cands].argmax()]

results = {
    "K": K, "D": D, "sigma": SIGMA, "n": N,
    "q_select_exact": round(acc(q_sel), 4),
    "majority_vote_exact": round(acc(vote), 4),
    "vote_q_tiebreak_exact": round(acc(vote_q), 4),
    "ref_K1_D48_exact": 0.478,  # M1c K 曲线（同 eval 集）
    "ref_q_select_K10_exact": 0.476,
}
print(json.dumps(results, indent=2), flush=True)
json.dump(results, open(OUT, "w"), indent=2)
print("saved -> " + OUT, flush=True)
