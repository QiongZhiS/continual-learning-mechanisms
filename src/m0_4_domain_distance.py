"""M0.4 域间结构距离实测（v0.7 域序列预注册 · 相似度梯度冻结）

方案 §3.3："域间相似度梯度 = 生成器实测的'域间结构距离'（M0.4 交付时冻结）"。
本脚本对现存全部域（A 数独 / B RPN / C 组合 / D IPD / E1-E3 候选）做**统计结构特征**
实测（pad 无关），输出归一化距离矩阵 + 近→远排序，写入验收记录用 JSON。

特征（每域，从数据集中采样计算，pad_id 无关）：
  1. token_entropy   非 pad 位的词元分布熵（信息量）
  2. pos_entropy     位置占用熵（内容在 81 位上的分布形状）
  3. fill_rate       非 pad 位占比（稀疏度）
  4. seq_var         非 pad 位长度的样本间方差（长度/形状多样性）
  5. label_entropy   label 位 0 的熵（输出多样性）
  6. vocab_used      实际用到的词元数
距离 = 特征向量（z-score 归一化后）欧氏距离。

用法: python m0_4_domain_distance.py [--out outputs/2026-08-12/domain_distance.json]
"""
import argparse
import json
import os

import numpy as np

ROOT = "../data"
DOMAINS = {
    "A-sudoku": "sudoku-extreme-1k-aug-100/train",
    "B-rpn": "domain-b-rpn/train",
    "C-combo": "domain-c-combo/train",
    "D-ipd": "domain-d-ipd/train",
    "E1-prefix": "domain-e-c1-prefix-muldiv/train",
    "E2-signal": "domain-e-c2-signal-infer/train",
    "E3-latin": "domain-e-c3-latin/train",
}
N_SAMPLE = 20000
SEED = 0


def features(path):
    inp = np.load(os.path.join(ROOT, path, "all__inputs.npy"), mmap_mode="r")
    lab = np.load(os.path.join(ROOT, path, "all__labels.npy"), mmap_mode="r")
    # dataset.json 位置两处查：域 A 在 train/ 下，域 B/C/D/E 在数据根目录
    dj = os.path.join(ROOT, path, "dataset.json")
    if not os.path.exists(dj):
        dj = os.path.join(ROOT, path.split("/")[0], "dataset.json")
    meta = json.load(open(dj))
    pad = meta["pad_id"]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(inp), min(N_SAMPLE, len(inp)), replace=False)
    X = np.asarray(inp[idx])
    Y = np.asarray(lab[idx])

    nonpad = X != pad
    # 1. token 分布熵（非 pad 位）
    tokens = X[nonpad]
    hist, _ = np.histogram(tokens, bins=np.arange(0, tokens.max() + 2))
    p = hist / hist.sum()
    token_ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    # 2. 位置占用熵（每列占用率分布）
    occ = nonpad.mean(0)
    p2 = occ / occ.sum()
    pos_ent = float(-(p2[p2 > 0] * np.log2(p2[p2 > 0])).sum())
    # 3. 填充率
    fill_rate = float(occ.mean())
    # 4. 长度方差（非 pad 数）
    lens = nonpad.sum(1)
    seq_var = float(lens.var())
    # 5. label 位 0 熵
    l0 = Y[:, 0]
    hist0, _ = np.histogram(l0, bins=np.arange(-0.5, l0.max() + 1.5))
    p0 = hist0 / hist0.sum()
    label_ent = float(-(p0[p0 > 0] * np.log2(p0[p0 > 0])).sum())
    # 6. vocab 使用数
    vocab_used = int(np.unique(X).size)
    return np.array([token_ent, pos_ent, fill_rate, np.log1p(seq_var),
                     label_ent, vocab_used], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/2026-08-12/domain_distance.json")
    args = ap.parse_args()

    feats = {}
    for name, path in DOMAINS.items():
        feats[name] = features(path)
        print(f"{name}: {feats[name].round(3).tolist()}", flush=True)

    names = list(DOMAINS)
    F = np.stack([feats[n] for n in names])
    mu, sd = F.mean(0), F.std(0) + 1e-9
    Z = (F - mu) / sd
    D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))

    dist = {}
    for i, a in enumerate(names):
        dist[a] = {}
        for j, b in enumerate(names):
            dist[a][b] = round(float(D[i, j]), 3)

    # 近→远排序（以 B 为锚：域序列成员是 A→B→C→D→E，冻结梯度按到 B 的距离）
    anchor = "B-rpn"
    order = sorted(names, key=lambda n: dist[anchor][n])
    print("\ndistance to B (near -> far):")
    for n in order:
        print(f"  {n:12s} {dist[anchor][n]:.3f}")

    out = {"features": {n: feats[n].tolist() for n in names},
           "feature_names": ["token_entropy", "pos_entropy", "fill_rate",
                             "log1p_len_var", "label_entropy", "vocab_used"],
           "z_norm": True, "metric": "euclidean on z-scored features",
           "distance_matrix": dist,
           "gradient_frozen": {"anchor": anchor,
                               "near_to_far": order,
                               "note": "M0.4 delivery freeze (v0.7 domain-sequence "
                                       "preregistration); E6a ordering uses this"}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
