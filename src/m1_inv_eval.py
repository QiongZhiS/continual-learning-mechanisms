"""M1 逆向模型推理期整合 + 判别（v0.16 S3 预注册 · 第 3 步）

展开每步: z_L += sigma*randn + alpha*s*d̂(z_L, puzzle)
  d̂ = per-sample 单位化方向；s = 1（norm=unit）或 sigma*sqrt(P*H)（norm=noise，
  使 push per-dim RMS = alpha*sigma，与噪声同尺度）
K=30 D=48 多数表决（输出位 0）· test 500 谜题（seed 0，与 sem_explore 同选集）

判据（预注册）: 存在 (alpha,sigma) ∈ 预注册网格 使 b > a+2pp 且 b > c+2pp → 通过
  a = σ=0.2 α=0 纯噪声多数表决；c = 同 α、σ=0 仅逆向（方向不是答案）
用法:
  python m1_inv_eval.py --alpha 0.5 --sigma 0.2 --norm unit --tag b_unit_0.5_0.2
  python m1_inv_eval.py --grid            # 完整预注册网格（含基线/对照/扩展）
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from eval_ptrm import load_test_data, load_model
from m1_inv_model import DirModel, PuzzleEncoder, puzzle_repr
from models.layers import CastedLinear

DATA = "../data/domain-b-rpn"
CKPT = "outputs/2026-08-10/m1_domainB/step_4000"
INV = "outputs/2026-08-13/inv_model/inv_model.pt"
OUT_DIR = "outputs/2026-08-13/inv_model"
K, D, N = 30, 48, 500
CHUNK = 16
P, H = 97, 512

# 预注册网格（b 条件，集合与原 6 组一致，按"强推在前"排序便于中途观察）
# norm=noise 是 s=σ·√(PH) 归一化的同网格（α 与 σ 同尺度语义）
B_GRID = [(1.0, 0.2), (0.5, 0.2), (0.1, 0.2), (1.0, 0.1), (0.5, 0.1), (0.1, 0.1)]


def load_inv(path, device="cuda"):
    enc = PuzzleEncoder().to(device)
    dm = DirModel().to(device)
    state = torch.load(path, map_location=device)
    enc_state = {k.removeprefix("enc."): v for k, v in state.items() if k.startswith("enc.")}
    dm_state = {k: v for k, v in state.items() if not k.startswith("enc.")}
    enc.load_state_dict(enc_state)
    dm.load_state_dict(dm_state)
    enc.eval()
    dm.eval()
    return enc, dm


def run_config(model, enc, dm, inputs, ids, labels0, prepr, alpha, sigma, norm, tag):
    """单配置 rollout；返回 vote 准确率等统计。prepr [N, 257] 每谜题一次。"""
    N0 = inputs.shape[0]
    preds = torch.zeros(N0, K, dtype=torch.long, device="cuda")
    t0 = time.time()
    with torch.inference_mode():
        for start in range(0, N0, CHUNK):
            bs = min(CHUNK, N0 - start)
            b = inputs[start:start + bs]
            batch = {"inputs": b.repeat(K, 1),
                     "puzzle_identifiers": ids[start:start + bs].repeat(K)}
            prepr_b = prepr[start:start + bs].repeat_interleave(K, 0)  # [K*bs, 257]
            with torch.device("cuda"):
                carry = model.initial_carry(batch)
            for _ in range(D):
                zL = carry.inner_carry.z_L
                if alpha > 0:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        d = dm(zL.float(), prepr_b)                 # [K*bs, P, H]
                    dn = d / (torch.linalg.vector_norm(d.float(), dim=(-2, -1),
                                                       keepdim=True) + 1e-8)
                    if norm == "noise":
                        s = sigma * np.sqrt(P * H)
                    else:
                        s = 1.0
                    push = (alpha * s * dn).to(zL.dtype)
                else:
                    push = None
                if push is not None:
                    zL = zL + push
                if sigma > 0:
                    zL = zL + sigma * torch.randn_like(zL)
                carry.inner_carry.z_L = zL
                carry, outputs = model(carry, batch)
            # repeat(K,1) 行布局 = (p0..p15) 重复 K 次 → view(K, bs) 再转置得到 [i, k]
            preds[start:start + bs] = outputs["logits"].argmax(-1)[:, 0].view(K, bs).t()
    vote = preds.mode(1).values
    vote_exact = float((vote == labels0).float().mean())
    rollout_exact = float((preds == labels0.unsqueeze(1)).float().mean())
    n_unique = np.array([int(preds[i].unique().numel()) for i in range(N0)])
    return {
        "tag": tag, "alpha": alpha, "sigma": sigma, "norm": norm,
        "vote_exact": round(vote_exact, 4),
        "rollout_mean_exact": round(rollout_exact, 4),
        "mean_unique_votes": round(float(n_unique.mean()), 2),
        "pct_single_variant": round(float((n_unique == 1).mean()), 3),
        "seconds": round(time.time() - t0, 1),
        "gpu_mem_MB": torch.cuda.max_memory_allocated() // 1048576,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--norm", default="unit", choices=["unit", "noise"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--n", type=int, default=None, help="谜题数（默认 500；--smoke 用 32）")
    ap.add_argument("--smoke", action="store_true", help="小规模冒烟：n=32 跑 a + b_unit(1.0,0.2)")
    ap.add_argument("--noise-family", action="store_true",
                    help="追加 norm=noise 族 6 组（α 与 σ 同尺度语义；post-grid 自适应）")
    args = ap.parse_args()

    meta = json.load(open(f"{DATA}/dataset.json", encoding="utf-8"))
    model = load_model(CKPT, "config/arch/trm.yaml", meta["vocab_size"],
                       meta["seq_len"], meta["num_puzzle_identifiers"])
    model.config.halt_max_steps = D
    model.eval()
    # 推理期一次性 bf16 权重预转换（消除每步 858 次 weight.to() 开销；数值等价：
    # 权重同为 bf16 后 to(bf16) 为 no-op，matmul 结果与逐次转换一致）
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, CastedLinear):
                m.weight.data = m.weight.data.bfloat16()
    n_use = args.n if args.n else N
    inputs, labels, ids, _ = load_test_data(DATA, n_use, 0)
    labels0 = labels[:, 0]

    enc, dm = load_inv(INV)
    with torch.inference_mode():
        prepr = puzzle_repr(model, inputs, enc)  # [N, 257]

    os.makedirs(OUT_DIR, exist_ok=True)

    def cfg_list():
        # (tag, alpha, sigma, norm)：a 基线 → c 对照（判据前置）→ b 预注册 6 组 → [b_noise 族]
        cfgs = [("a_pure_noise", 0.0, 0.2, "unit")]
        for a in (0.5, 1.0):
            cfgs.append((f"c_unit_{a}_0.0", a, 0.0, "unit"))
        for a, s in B_GRID:
            cfgs.append((f"b_unit_{a}_{s}", a, s, "unit"))
        if args.noise_family:
            for a, s in B_GRID:
                cfgs.append((f"b_noise_{a}_{s}", a, s, "noise"))
        return cfgs

    if args.smoke:
        for tag, a, s, norm in [("a_pure_noise", 0.0, 0.2, "unit"),
                                ("b_unit_1.0_0.2", 1.0, 0.2, "unit")]:
            r = run_config(model, enc, dm, inputs, ids, labels0, prepr, a, s, norm, tag)
            print(f"[smoke] {tag}: vote={r['vote_exact']} ({r['seconds']}s)", flush=True)
        return

    if not args.grid:
        assert args.alpha is not None and args.sigma is not None
        tag = args.tag or f"b_{args.norm}_{args.alpha}_{args.sigma}"
        r = run_config(model, enc, dm, inputs, ids, labels0, prepr,
                       args.alpha, args.sigma, args.norm, tag)
        print(json.dumps(r, indent=2), flush=True)
        json.dump(r, open(f"{OUT_DIR}/cfg_{tag}.json", "w"), indent=2)
        return

    # 网格模式：逐配置执行，已完成则跳过（断点续跑）
    results = {}
    cfgs = cfg_list()
    for tag, a, s, norm in cfgs:
        f = f"{OUT_DIR}/cfg_{tag}.json"
        if os.path.exists(f):
            results[tag] = json.load(open(f))
            print(f"[skip] {tag}", flush=True)
            continue
        torch.cuda.reset_peak_memory_stats()
        r = run_config(model, enc, dm, inputs, ids, labels0, prepr, a, s, norm, tag)
        results[tag] = r
        json.dump(r, open(f, "w"), indent=2)
        print(f"[done] {tag}: vote={r['vote_exact']} rollout={r['rollout_mean_exact']} "
              f"({r['seconds']}s)", flush=True)

    # 判别判定
    a_vote = results.get("a_pure_noise", {}).get("vote_exact", None)
    c_vote = {tag: r["vote_exact"] for tag, r in results.items() if tag.startswith("c_")}
    judged = []
    if a_vote is not None:
        for tag, r in results.items():
            if not tag.startswith("b_"):
                continue
            a_same = tag.split("_")[2]   # b_unit_0.5_0.2 -> alpha
            c_key = f"c_unit_{a_same}_0.0"
            c_val = results.get(c_key, {}).get("vote_exact")
            if c_val is None:
                continue
            beats_a = r["vote_exact"] >= a_vote + 0.02
            beats_c = r["vote_exact"] >= c_val + 0.02
            judged.append({**r, "beats_a_2pp": beats_a, "beats_c_2pp": beats_c,
                           "passes": bool(beats_a and beats_c)})
    passed = any(j["passes"] for j in judged)
    matrix = {
        "n_puzzles": n_use, "K": K, "D": D, "ckpt": CKPT, "inv_model": INV,
        "baseline_a": a_vote,
        "controls_c": c_vote,
        "grid": results,
        "judged_b": judged,
        "criterion": "exists (alpha,sigma) in b-grid: b >= a+0.02 and b >= c(alpha)+0.02",
        "passed": passed,
    }
    json.dump(matrix, open(f"{OUT_DIR}/matrix.json", "w"), indent=2)
    print("=== 判据 ===", flush=True)
    for j in judged:
        print(f"{j['tag']}: vote={j['vote_exact']} beats_a={j['beats_a_2pp']} "
              f"beats_c={j['beats_c_2pp']} passes={j['passes']}", flush=True)
    print(f"PASSED: {passed}", flush=True)
    print(f"saved -> {OUT_DIR}/matrix.json", flush=True)


if __name__ == "__main__":
    main()
