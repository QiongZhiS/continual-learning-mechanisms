"""M0.4a 域 C 生成器：组合域（特征→实体推断 · E4 鸭嘴兽载体 + E6a 序列成员）

设计（与域 A/B 格式同构）：
- 实体 = 颜色×动物 = 36 组合；vocab: 颜色 0-5 + 动物 6-11 + PAD=12（vocab_size=13）
- 输入 81 位：<color> <animal> PAD...；输出 81 位：位 0 = x，位 1 = y，其余 PAD
- 非线性 f（平凡组合防线，v0.6 预注册要求）：x = (c*(a+1)) % 10, y = (a*(c+1)) % 10
  乘法交互（c、a 的乘积项）——线性模型无法外推未见组合（E4 判据需要非平凡任务）
- 训练 24 组合 × N_AUG 重复；test 12 全新组合（鸭嘴兽测试：零次推断）
  （v0.20 扩域：16→36 组合，测试 4→12 样本，E4/E6a 判据噪声 ±25pp→±14pp）
- 验证：① 标签与 f 一致 ② 线性模型外推未见组合失败（平凡防线）③ 格式同构
"""
import json
import os

import numpy as np

OUT = "../data/domain-c-combo"
SEQ_LEN = 81
PAD = 12
N_AUG = 500  # 每组合重复次数（规则学习多示例，对齐域 A/B 增强语义）
COLORS = list(range(6))      # 红橙黄绿蓝紫
ANIMALS = list(range(6, 12)) # 鱼象猫鸟蛇龙


def f_x(c, a):
    return (c * (a + 1)) % 10


def f_y(c, a):
    return (a * (c + 1)) % 10


def encode(c, a):
    inp = np.full(SEQ_LEN, PAD, dtype=np.int64)
    inp[0], inp[1] = c, a
    lab = np.full(SEQ_LEN, PAD, dtype=np.int64)
    lab[0], lab[1] = f_x(c, a), f_y(c, a)
    return inp, lab


def write(split, combos):
    d = os.path.join(OUT, split)
    os.makedirs(d, exist_ok=True)
    inputs, labels = [], []
    for c, a in combos:
        inp, lab = encode(c, a)
        for _ in range(N_AUG if split == "train" else 1):
            inputs.append(inp.copy())
            labels.append(lab.copy())
    n = len(inputs)
    np.save(os.path.join(d, "all__inputs.npy"), np.stack(inputs))
    np.save(os.path.join(d, "all__labels.npy"), np.stack(labels))
    np.save(os.path.join(d, "all__puzzle_identifiers.npy"),
            np.zeros(n, dtype=np.int32))
    print(f"{split}: {n} samples ({len(combos)} combos) -> {d}", flush=True)
    return n


def main():
    rng = np.random.default_rng(42)
    combos = [(c, a) for c in COLORS for a in ANIMALS]  # 36 组合
    rng.shuffle(combos)
    train_combos, test_combos = combos[:24], combos[24:]  # 24 训练 / 12 鸭嘴兽

    # ① 标签正确性（独立验证 f）
    for c, a in combos:
        inp, lab = encode(c, a)
        assert inp[0] == c and inp[1] == a
        assert lab[0] == f_x(c, a) and lab[1] == f_y(c, a)
    print(f"label check: {len(combos)}/{len(combos)} combos OK", flush=True)

    # ② 平凡防线：线性模型（无乘积项）外推未见组合必须失败
    X = np.array([[c, a] for c, a in train_combos], dtype=float)
    yt = np.array([f_x(c, a) for c, a in train_combos], dtype=float)
    A = np.column_stack([np.ones(len(X)), X])  # 1, c, a（无 c*a 乘积项）
    coef, *_ = np.linalg.lstsq(A, yt, rcond=None)
    Xt = np.array([[c, a] for c, a in test_combos], dtype=float)
    pred = np.round(np.column_stack([np.ones(len(Xt)), Xt]) @ coef)
    true = np.array([f_x(c, a) for c, a in test_combos], dtype=float)
    lin_acc = (pred == true).mean()
    print(f"trivial-defense: linear extrapolation on unseen combos: "
          f"acc={lin_acc:.2f} (must be < 1.0; {len(combos)} combos, "
          f"{len(train_combos)} train -> {len(test_combos)} unseen)",
          flush=True)
    assert lin_acc < 1.0, "f is linearly solvable -> E4 平凡组合防线失败，换 f"

    n_train = write("train", train_combos)
    n_test = write("test", test_combos)

    meta = {"pad_id": PAD, "ignore_label_id": PAD, "blank_identifier_id": 0,
            "vocab_size": 13, "seq_len": SEQ_LEN, "num_puzzle_identifiers": 1,
            "total_groups": n_train, "mean_puzzle_examples": N_AUG,
            "total_puzzles": len(train_combos), "sets": ["all"],
            "train_combos": [[c, a] for c, a in train_combos],
            "test_combos": [[c, a] for c, a in test_combos],
            "f": "x=(c*(a+1))%10, y=(a*(c+1))%10 (multiplicative, non-linear)",
            "linear_unseen_acc": float(lin_acc),
            "note": "M0.4a domain C: feature->entity inference; colors 0-5 + "
                    "animals 6-11 + PAD=12; result at output pos 0(x) 1(y); "
                    "duckbill test = 12 unseen combos zero-shot"}
    json.dump(meta, open(os.path.join(OUT, "dataset.json"), "w"), indent=2)
    print("saved -> " + os.path.join(OUT, "dataset.json"), flush=True)


if __name__ == "__main__":
    main()
