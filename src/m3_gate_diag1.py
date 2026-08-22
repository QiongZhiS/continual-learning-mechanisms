"""G 门诊断 ①：错误记录质量——高置信错误是否携带规则结构（v0.28 预注册诊断方向 ①）

问题：夜间环节收集的 conf>0.9 错误（域 B train 固定 seed 0 抽 512）是否携带可学习的规则信息，
还是均匀噪声（域 B q 不可分污染的迹象）。

分析：
  1. 复现夜间收集（同一 512 子集 / 更大 N），统计错误率 + 置信度分布
  2. 特征条件错误率：expr_len / n_ops / n_sub / 结果值 —— 错误是否集中于特定特征区间（结构）
  3. K=1 vs K=5 错误稳定性：更多 rollout 计算后仍错的谜题 = 稳健错误（结构性）

用法（repo 目录）：
  python m3_gate_diag1.py --ckpt outputs/2026-08-15/m3_gate/d_s2_step_2000 --n 512 --seed 0
  python m3_gate_diag1.py --ckpt outputs/2026-08-15/m3_gate/d_s2_step_2000 --n 2000 --seed 0 --k 5
"""
import argparse
import json
import os

import numpy as np
import torch

from eval_ptrm import ptrm_infer
from m3_gate import load_arch, B_DATA, CONF_THR
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

OUT = "outputs/2026-08-15/m3_gate_diag1"


def load_model(ckpt):
    meta = json.load(open(f"{B_DATA}/dataset.json"))
    arch_cfg = load_arch(meta)
    m = TinyRecursiveReasoningModel_ACTV1(arch_cfg)
    state = {k.removeprefix("model."): v for k, v in torch.load(ckpt, map_location="cuda").items()}
    m.load_state_dict(state, assign=True)
    m.eval()
    m.to("cuda")
    return m, meta


def rpn_features(row, label):
    toks = row[row != 12]
    n_ops = int(np.isin(toks, [10, 11]).sum())
    return {
        "expr_len": int(len(toks)),
        "n_ops": n_ops,
        "n_sub": int((toks == 11).sum()),
        "n_add": int((toks == 10).sum()),
        "res": int(label[0]),
    }


def infer_conf_pred(m, inputs, ids, batch=128, halt=16):
    """16 步前向 → 输出位 0 的 softmax 置信度与预测（对齐 collect_high_conf_errors）。"""
    inner = m
    N = inputs.shape[0]
    confs = torch.zeros(N, device="cuda")
    preds = torch.zeros(N, dtype=torch.long, device="cuda")
    with torch.inference_mode():
        for start in range(0, N, batch):
            b = inputs[start:start + batch]
            bs = b.shape[0]
            batch_kw = {"inputs": b, "puzzle_identifiers": ids[start:start + bs]}
            with torch.device("cuda"):
                carry = inner.initial_carry(batch_kw)
            for _ in range(halt):
                carry, outputs = inner(carry, batch_kw)
            logits = outputs["logits"][:, 0]
            conf, pred = logits.softmax(-1).max(-1)
            confs[start:start + bs] = conf
            preds[start:start + bs] = pred
    return confs, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=1, help="K=5 时额外测 rollout 稳定性")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    m, meta = load_model(args.ckpt)
    t = os.path.join(B_DATA, "train")
    tr_in = np.load(os.path.join(t, "all__inputs.npy"), mmap_mode="r")
    tr_lb = np.load(os.path.join(t, "all__labels.npy"), mmap_mode="r")
    tr_ids = np.load(os.path.join(t, "all__puzzle_identifiers.npy"), mmap_mode="r")

    idx = np.sort(np.random.default_rng(args.seed).choice(
        len(tr_in), min(args.n, len(tr_in)), replace=False))
    inputs = torch.tensor(np.asarray(tr_in[idx], dtype=np.int32), device="cuda")
    labels = torch.tensor(np.asarray(tr_lb[idx], dtype=np.int32), device="cuda")
    ids = torch.tensor(np.asarray(tr_ids[idx], dtype=np.int32), device="cuda")

    confs, preds = infer_conf_pred(m, inputs, ids)
    correct = (preds == labels[:, 0]).cpu().numpy()
    conf_np = confs.cpu().numpy()
    err = ~correct
    hc_err = err & (conf_np > CONF_THR)

    feats = [rpn_features(np.asarray(tr_in[i]), np.asarray(tr_lb[i])) for i in idx]
    res = {
        "ckpt": args.ckpt, "n": args.n, "seed": args.seed, "k": args.k,
        "n_err": int(err.sum()),
        "n_hc_err": int(hc_err.sum()),
        "err_rate": float(err.mean()),
        "hc_err_rate": float(hc_err.mean()),
        "hc_share_of_errors": float(hc_err.sum() / max(1, err.sum())),
        "err_conf_mean": float(conf_np[err].mean()) if err.any() else None,
        "hc_err_conf_mean": float(conf_np[hc_err].mean()) if hc_err.any() else None,
    }

    # 特征条件错误率
    feat_bins = {
        "expr_len": [3, 5, 7, 9, 100],
        "n_ops": [1, 2, 3, 4, 100],
        "res": [0, 3, 6, 9, 100],
    }
    ftable = {}
    for fname, edges in feat_bins.items():
        vals = np.array([f[fname] for f in feats])
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (vals >= lo) & (vals < hi)
            if mask.sum() == 0:
                continue
            rows.append({
                "bin": f"[{lo},{hi})", "n": int(mask.sum()),
                "err_rate": float(err[mask].mean()),
                "hc_err_rate": float(hc_err[mask].mean()),
            })
        ftable[fname] = rows
    res["feature_conditioned"] = ftable

    # K=1 vs K=5 稳定性（同一批样本）；temp 按位置传（ptrm_infer 签名: (model, inputs, ids, K, D, temp)）
    if args.k > 1:
        pred5, _ = ptrm_infer(m, inputs, ids, args.k, 16, 0.0)
        err5 = (pred5[:, 0] != labels[:, 0]).cpu().numpy()
        res["k5"] = {
            "err_rate": float(err5.mean()),
            "stable_err_rate": float((err & err5).mean()),
            "k1_only_err_rate": float((err & ~err5).mean()),
            "k5_only_err_rate": float((~err & err5).mean()),
            "n_stable_err": int((err & err5).sum()),
        }

    os.makedirs(OUT, exist_ok=True)
    tag = os.path.basename(args.ckpt)
    out_path = f"{OUT}/{tag}_n{args.n}_s{args.seed}_k{args.k}.json"
    json.dump(res, open(out_path, "w"), indent=2)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
