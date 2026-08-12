"""M1a 域 B 生成器：RPN 栈机算术（后缀表达式求值）

设计（与域 A 数独格式同构）：
- vocab: 0-9 数字 (10) + ADD=10 + SUB=11 + PAD=12 → vocab_size=13, pad_id=12
- 所有中间结果 ∈ [0,9]（范围采样保证，无死循环）；ops v1 = {ADD, SUB}
- 输入 81 位：RPN 程序左对齐 + PAD 填充
- 输出 81 位：位置 0 = 结果（0-9），其余 PAD（ignore=pad_id → loss 只算结果位）
- train: 1000 结构 × 101 值实例 = 101K（结构重复 = 规则学习多示例，对齐域 A 增强语义）
- test: 2000 全新结构+实例（测规则泛化，非记忆）
- 深度 ∈ {2,3,4}（token 数 7/15/31，均 ≤ 81）

--aug（通道③ · v0.20 E3 先验对照臂 · 独立验证）：
训练侧等价类注入——对每个 (结构,值) 实例枚举**全部** ADD 子树交换等价形式
（2^#ADD 个，上限 --aug-cap，默认 16）加入训练集。交换 = 交换 ADD 节点左右子树，
答案数学上不变（加法交换律；扰动器已在 m1_sem_explore.py 验证等价性）。
- 原样本流与无增广完全一致：增广在 **test 生成之后**才消耗 rng → test 逐字节同基线
- 训练集 = 各实例等价类全集的并集（去重；对称子树去重后可能少于 cap）
- 每个变体用独立栈求值器 stack_eval 断言 == 原答案（等价性防线）
"""
import argparse
import json
import os
import time
from dataclasses import dataclass

import numpy as np

OUT = "../data/domain-b-rpn"
N_STRUCT = 1000
N_AUG = 101
N_TEST = 2000
MAX_VAL = 9
SEQ_LEN = 81
ADD, SUB, PAD = 10, 11, 12
OPS = [ADD, SUB]


@dataclass
class Node:
    op: object = None
    left: object = None
    right: object = None


def sample_structure(rng, depth):
    if depth <= 1:
        return Node()
    return Node(rng.choice(OPS),
                sample_structure(rng, depth - 1),
                sample_structure(rng, depth - 1))


def fill(rng, node, hi):
    """填值并返回 (token 序列, 结果值)。值域 [0, hi] 由范围采样保证。"""
    if node.op is None:
        v = int(rng.integers(0, hi + 1))
        return [v], v
    a_seq, a_v = fill(rng, node.left, hi)
    if node.op == ADD:
        b_hi = hi - a_v
    elif node.op == SUB:
        b_hi = a_v
    else:
        b_hi = hi // max(a_v, 1)
    b_seq, b_v = fill(rng, node.right, b_hi)
    if node.op == ADD:
        v = a_v + b_v
    elif node.op == SUB:
        v = a_v - b_v
    else:
        v = a_v * b_v
    return a_seq + b_seq + [node.op], v


def encode(seq, result):
    inp = np.full(SEQ_LEN, PAD, dtype=np.int64)
    inp[:len(seq)] = seq
    lab = np.full(SEQ_LEN, PAD, dtype=np.int64)
    lab[0] = result
    return inp, lab


def write(split, inputs, labels):
    d = os.path.join(OUT_DIR, split)
    os.makedirs(d, exist_ok=True)
    n = len(inputs)
    np.save(os.path.join(d, "all__inputs.npy"), np.stack(inputs))
    np.save(os.path.join(d, "all__labels.npy"), np.stack(labels))
    np.save(os.path.join(d, "all__puzzle_identifiers.npy"),
            np.zeros(n, dtype=np.int32))
    print(f"{split}: {n} samples -> {d}", flush=True)


def make_instance(rng):
    depth = int(rng.integers(2, 5))
    node = sample_structure(rng, depth)
    seq, v = fill(rng, node, MAX_VAL)
    assert 0 <= v <= MAX_VAL
    return seq, v


# ---- 通道③ 等价类增广（与 m1_sem_explore.py 的扰动器同构） ----

def parse_seq(seq):
    """RPN token → AST（叶子带值）。程序保证单根。"""
    stack = []
    for t in seq:
        if t == PAD:
            break
        if t in (ADD, SUB):
            r, l = stack.pop(), stack.pop()
            stack.append(Node(t, l, r))
        else:
            stack.append(Node(int(t)))
    return stack[0]


def serialize(node):
    """AST → RPN token 列表（与 fill 一致：左 + 右 + op）"""
    if node.op < 10:
        return [node.op]
    return serialize(node.left) + serialize(node.right) + [node.op]


def stack_eval(seq):
    """独立栈求值器（等价性断言用；与 fill 的递归实现不同源）"""
    st = []
    for t in seq:
        if t in (ADD, SUB):
            b, a = st.pop(), st.pop()
            st.append(a + b if t == ADD else a - b)
        else:
            st.append(int(t))
    return st[0]


def copy_tree(node):
    if node.op < 10:
        return Node(node.op)
    return Node(node.op, copy_tree(node.left), copy_tree(node.right))


def collect_adds(node, acc):
    if node.op in (ADD, SUB):
        collect_adds(node.left, acc)
        collect_adds(node.right, acc)
        if node.op == ADD:
            acc.append(node)
    return acc


def apply_swap_mask(tree, mask):
    """mask 第 i 位 = 1 → 交换第 i 个 ADD 节点的左右子树（左先序编号）。
    答案不变（加法交换律）。"""
    out = copy_tree(tree)
    adds = collect_adds(out, [])
    for i, n in enumerate(adds):
        if (mask >> i) & 1:
            n.left, n.right = n.right, n.left
    return out


def swap_variants(seq, cap):
    """一个 (结构,值) 实例的全部 ADD 交换等价形式（含原程序；上限 cap 个）。
    返回 (token 列表列表, 去重后形式数)。等价性由调用方用 stack_eval 断言。"""
    tree = parse_seq(seq)
    k = len(collect_adds(tree, []))
    forms, seen = [], set()
    for m in range(min(1 << k, cap)):
        s = tuple(serialize(apply_swap_mask(tree, m)))
        if s not in seen:
            seen.add(s)
            forms.append(list(s))
    return forms, len(seen)


def augment(inputs, labels, cap):
    """为每个训练样本追加其全部 ADD 交换等价形式（含原程序去重后跳过）。
    确定性（掩码枚举，不消耗 rng）。返回 (增广后 inputs, labels, 增广统计)。
    等价性逐样本断言。"""
    out_in, out_lb = list(inputs), list(labels)
    n_added = 0
    n_orig = len(inputs)
    bad = 0
    t0 = time.time()
    for i in range(n_orig):
        seq = inputs[i][inputs[i] != PAD].tolist()
        v = int(labels[i][0])
        forms, nf = swap_variants(seq, cap)
        for s in forms:
            if stack_eval(s) != v:
                bad += 1
                continue  # 不等价的形式丢弃（理论上不发生，防线兜底）
            inp, lab = encode(s, v)
            out_in.append(inp)
            out_lb.append(lab)
        n_added += nf - 1  # 减 1 = 原程序已在基线集中
    print(f"aug: +{n_added} variants (orig {n_orig} -> total {len(out_in)}), "
          f"equiv_bad={bad}, {time.time()-t0:.0f}s", flush=True)
    assert bad == 0, "等价性断言失败：ADD 交换改变了答案"
    return out_in, out_lb, {"n_added": n_added, "n_total": len(out_in),
                            "n_orig": n_orig, "equiv_bad": bad,
                            "cap": cap}


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", action="store_true",
                    help="通道③：训练侧 ADD 等价类注入（全类枚举）")
    ap.add_argument("--aug-cap", type=int, default=16, help="每实例等价形式上限")
    ap.add_argument("--out", default=OUT, help="数据输出目录")
    args = ap.parse_args()
    OUT_DIR = args.out
    os.makedirs(OUT_DIR, exist_ok=True)

    rng = np.random.default_rng(42)

    # train: 结构 × 值实例（原样本流，与基线逐字节一致）
    structs = [sample_structure(rng, int(rng.integers(2, 5))) for _ in range(N_STRUCT)]
    inputs, labels = [], []
    for node in structs:
        for _ in range(N_AUG):
            seq, v = fill(rng, node, MAX_VAL)
            assert 0 <= v <= MAX_VAL
            inp, lab = encode(seq, v)
            inputs.append(inp)
            labels.append(lab)

    # test: 全新结构（在增广之前生成 → rng 状态与基线一致 → 逐字节相同）
    test_inputs, test_labels = [], []
    for _ in range(N_TEST):
        seq, v = make_instance(rng)
        inp, lab = encode(seq, v)
        test_inputs.append(inp)
        test_labels.append(lab)
    write("test", test_inputs, test_labels)

    aug_info = None
    if args.aug:
        # 增广在 test 之后：消耗 rng 只影响训练集扩展，不动 test
        inputs, labels, aug_info = augment(inputs, labels, args.aug_cap)
    write("train", inputs, labels)

    meta = {"pad_id": PAD, "ignore_label_id": PAD, "blank_identifier_id": 0,
            "vocab_size": 13, "seq_len": SEQ_LEN, "num_puzzle_identifiers": 1,
            "total_groups": N_STRUCT * N_AUG, "mean_puzzle_examples": 1.0,
            "total_puzzles": N_STRUCT, "sets": ["all"],
            "note": "M1a domain B: RPN stack-machine arithmetic; "
                    "ADD=10 SUB=11 PAD=12; result at output pos 0"}
    if aug_info:
        meta["aug"] = {"mode": "train_side_add_swap_full_class",
                       **aug_info}
        meta["note"] += "; channel-3 equivariant-class injection"
    json.dump(meta, open(os.path.join(OUT_DIR, "dataset.json"), "w"), indent=2)
    print(f"meta -> {os.path.join(OUT_DIR, 'dataset.json')}", flush=True)


if __name__ == "__main__":
    main()
