"""M1 逆向模型训练数据采集（v0.16 S3 预注册判别 · 第 1 步）

用 step_4000（K=1 D=48 sigma=0，干净轨迹）在域 B **train** 集采集：
- 每谜题 z_L 轨迹 [D=48, P=97, H=512] fp16（P = 81 token + 16 puzzle-emb 位置）
- 输出位 0 正确性（最终 logits[:, 0] == labels[:, 0]）
- 防 test 泄漏：逆向模型训练只用 train 集轨迹，判别 eval 才用 test 集

输出: outputs/2026-08-13/inv_data/ 下 chunk npz + meta.json + summary.json
用法: python m1_inv_train_data.py [--n 1500] [--seed 42] [--chunk 16]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from eval_ptrm import load_model

DATA = "../data/domain-b-rpn"
CKPT = "outputs/2026-08-10/m1_domainB/step_4000"
OUT = "outputs/2026-08-13/inv_data"
D = 48
PAD = 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(f"{DATA}/dataset.json", encoding="utf-8"))
    model = load_model(CKPT, "config/arch/trm.yaml", meta["vocab_size"],
                       meta["seq_len"], meta["num_puzzle_identifiers"])
    model.config.halt_max_steps = D
    model.eval()

    train_in = np.load(f"{DATA}/train/all__inputs.npy", mmap_mode="r")
    train_lb = np.load(f"{DATA}/train/all__labels.npy", mmap_mode="r")
    train_ids = np.load(f"{DATA}/train/all__puzzle_identifiers.npy", mmap_mode="r")

    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(train_in), args.n, replace=False))
    H = 512
    P = meta["seq_len"] + 16  # puzzle_emb_len=16（trm.yaml 非零）
    N_CHUNK = (args.n + args.chunk - 1) // args.chunk

    # 汇总统计（fp32 下按 per-dim RMS 统计，避免维度量级混淆）
    drift_sq = np.zeros(D)          # per-step 相邻 z_L 差 per-dim 平方和
    zL_sq_final = 0.0               # 末步 z_L per-dim 平方和
    drift_cnt = 0
    max_abs = 0.0
    chunks = []
    n_correct = 0
    t0 = time.time()

    with torch.inference_mode():
        for ci in range(N_CHUNK):
            sl = idx[ci * args.chunk:(ci + 1) * args.chunk]
            bs = len(sl)
            b = torch.tensor(np.asarray(train_in[sl], dtype=np.int32), device="cuda")
            labels0 = torch.tensor(np.asarray(train_lb[sl, 0], dtype=np.int32), device="cuda")
            batch = {"inputs": b, "puzzle_identifiers": torch.zeros(bs, dtype=torch.int32,
                                                                    device="cuda")}
            with torch.device("cuda"):
                carry = model.initial_carry(batch)
            traj = torch.empty(D, bs, P, H, dtype=torch.float16, device="cuda")
            prev = None
            for step in range(D):
                carry, outputs = model(carry, batch)   # sigma=0 干净 rollout
                zL = carry.inner_carry.z_L              # bf16 [bs, P, H]
                traj[step] = zL.half()
                zf = zL.float()
                max_abs = max(max_abs, float(zf.abs().max()))
                if prev is not None:
                    d = (zf - prev.float())
                    drift_sq[step] += float((d * d).sum())
                    drift_cnt += d.numel()
                prev = zf
            zL_sq_final += float((zf * zf).sum())
            pred0 = outputs["logits"].argmax(-1)[:, 0]
            correct = (pred0 == labels0).cpu().numpy()
            n_correct += int(correct.sum())

            f = os.path.join(OUT, f"chunk_{ci:04d}.npz")
            np.savez(f, traj=traj.cpu().numpy(),
                     inputs=np.asarray(train_in[sl], dtype=np.int16),
                     labels0=np.asarray(train_lb[sl, 0], dtype=np.int16),
                     ids=np.asarray(train_ids[sl], dtype=np.int64),
                     correct=correct)
            chunks.append(f)
            torch.cuda.synchronize()
            print(f"chunk {ci}/{N_CHUNK} bs={bs} correct={int(correct.sum())} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    drift_rms = np.sqrt(drift_sq / drift_cnt)
    zL_final_rms = np.sqrt(zL_sq_final / (args.n * P * H))
    summary = {
        "ckpt": CKPT, "data_split": "train", "n": int(args.n), "seed": args.seed,
        "D": D, "P": P, "H": H, "dtype": "float16",
        "n_correct": int(n_correct),
        "correct_rate": round(n_correct / args.n, 4),
        "zL_final_perdim_rms": round(float(zL_final_rms), 4),
        "zL_max_abs": round(float(max_abs), 2),
        "fp16_overflow_risk": max_abs > 1000,
        "per_step_drift_rms": [round(float(x), 4) for x in drift_rms],
        "mean_drift_rms": round(float(drift_rms.mean()), 4),
        "seconds": round(time.time() - t0, 1),
        "chunks": [os.path.basename(c) for c in chunks],
    }
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
    np.save(f"{OUT}/idx.npy", idx)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved -> {OUT}/", flush=True)


if __name__ == "__main__":
    main()
