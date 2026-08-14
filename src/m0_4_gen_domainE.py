"""M0.4 域 E 候选三实例化（E6a 终极判据测试物 · v0.6.1 预注册三候选）

候选（选择判据 = A-D 训练后零样本迁移预测量，E6a 时三选一）：
  --candidate 1  新伪语言（近域 · 域 B 变体）：
                 前缀记法栈机算术，op ∈ {MUL, DIV}（域 B = 后缀 {ADD, SUB}）。
                 语法反转（前缀 vs 后缀）+ 语义结构不同（乘除 vs 加减，含非交换）。
                 vocab 13：0-9 + MUL=10 + DIV=11 + PAD=12；exact = 输出位 0
  --candidate 2  信号推断（远域 · 域 D 逆结构）：
                 信号博弈信息流反向——从发送者 10 轮信号反推隐藏状态 s ∈ [0,9]，
                 类型 t ∈ {诚实(0.5): 80% m=s + 20% 噪声, 欺骗(0.5): 80% m=(s+3)%10 + 20% 噪声}，
                 类型线索在位 10（10=诚实 11=欺骗）。任务 = 噪声鲁棒众数 + 条件平移。
                 天花板 1.0（10 轮聚合）。exact = 输出位 0
  --candidate 3  拉丁方补全（中域 · 数独对偶）：
                 9×9 拉丁方（行/列 1-9 各一次；无宫约束 = 数独约束结构颠倒）。
                 结构 = 挖空掩码（唯一解 carve 自模板解）；实例 = 同掩码 × 值符号置换
                 σ(模板解)（σ ∈ S9；唯一性在置换下保真：|解集| 不变 → 零逐实例验证）。
                 输入挖空位 = PAD=12；vocab 13；exact = 全网格（cell 基准，判据 E6a 定）。
                 掩码生成 multiprocessing 并行（carve 3.8s/掩码，3000 掩码单进程 3h+）
格式同构域 B/C：vocab 13 · seq_len 81 · identifier 全 0 · pad_id 12。
三候选冻结于本脚本 + dataset.json（E6a 选择前不修改）。
"""
import argparse
import json
import os
import time

import numpy as np

SEQ_LEN = 81
PAD = 12
MUL, DIV = 10, 11
MAX_VAL = 9
N_STRUCT = 1000
N_AUG = 101
N_TEST = 2000
KEEP_HINTS = 30          # 拉丁方保留提示数
E2_NOISE = 0.2           # 信号噪声率
E2_OFFSET = 3            # 欺骗型偏移


def encode(seq, lab_pos0):
    """通用 81 位编码：input = seq 左对齐 + PAD；label = 位 0 = lab_pos0"""
    inp = np.full(SEQ_LEN, PAD, dtype=np.int64)
    inp[:len(seq)] = seq
    lab = np.full(SEQ_LEN, PAD, dtype=np.int64)
    lab[0] = lab_pos0
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


# ================= 候选 1：伪语言（前缀 MUL/DIV 栈机算术） =================

def sample_struct1(rng, depth):
    """前缀 AST：depth<=1 → 叶子；否则 op 在前，左右子树 depth-1"""
    if depth <= 1:
        return ("leaf",)
    return (rng.choice([MUL, DIV]), sample_struct1(rng, depth - 1),
            sample_struct1(rng, depth - 1))


def fill1(rng, node, hi, lo=0):
    """前缀填充。值域 [lo, hi]。返回 (token 序列, 结果)。
    MUL：右子树 hi = hi // max(左值,1)（保 a*b ≤ 9）
    DIV：右子树 [1, 左值]（保整除商 ≥ 1 且非零）"""
    if node[0] == "leaf":
        v = int(rng.integers(lo, hi + 1))
        return [v], v
    if node[0] == MUL:
        a_seq, a_v = fill1(rng, node[1], hi, lo)
        b_seq, b_v = fill1(rng, node[2], hi // max(a_v, 1), lo)
        return [MUL] + a_seq + b_seq, a_v * b_v
    a_seq, a_v = fill1(rng, node[1], hi, 1)    # 左值 ≥ 1（除数非零）
    b_seq, b_v = fill1(rng, node[2], a_v, 1)   # b ∈ [1, a]
    return [DIV] + a_seq + b_seq, a_v // b_v


def eval_prefix(seq):
    """独立前缀求值器（等价性断言；与 fill1 递归实现不同源）。反转 → 后缀求值"""
    st = []
    for t in reversed(seq):
        if t in (MUL, DIV):
            a, b = st.pop(), st.pop()
            st.append(a * b if t == MUL else a // b)
        else:
            st.append(int(t))
    return st[0]


def gen_e1(rng, n_struct, n_aug, n_test):
    """候选 1 数据。返回 (train_inputs, train_labels, test_inputs, test_labels)"""
    structs = [sample_struct1(rng, int(rng.integers(2, 5))) for _ in range(n_struct)]
    tr_in, tr_lb = [], []
    for node in structs:
        for _ in range(n_aug):
            seq, v = fill1(rng, node, MAX_VAL)
            assert 0 <= v <= MAX_VAL
            inp, lab = encode(seq, v)
            tr_in.append(inp)
            tr_lb.append(lab)
    te_in, te_lb = [], []
    for _ in range(n_test):
        seq, v = fill1(rng, sample_struct1(rng, int(rng.integers(2, 5))), MAX_VAL)
        assert eval_prefix(seq) == v          # 等价性防线（独立求值器）
        inp, lab = encode(seq, v)
        te_in.append(inp)
        te_lb.append(lab)
    return tr_in, tr_lb, te_in, te_lb


# ================= 候选 2：信号推断（域 D 逆结构） =================

def gen_e2(rng, n_train, n_test):
    """候选 2：输入位 0-9 = 信号序列，位 10 = 类型（10/11），输出位 0 = 状态。
    可学性：类型已知时 10 轮众数 → 诚实直读 / 欺骗 (众数-3)%10，天花板 ~1.0"""
    def inst():
        s = int(rng.integers(0, 10))
        t = int(rng.random() < 0.5)
        seq = []
        for _ in range(10):
            if rng.random() < E2_NOISE:
                m = int(rng.integers(0, 10))
            else:
                m = (s + E2_OFFSET * t) % 10
            seq.append(m)
        seq.append(10 + t)                    # 类型线索位
        return seq, s
    tr_in, tr_lb = [], []
    for _ in range(n_train):
        seq, s = inst()
        inp, lab = encode(seq, s)
        tr_in.append(inp)
        tr_lb.append(lab)
    te_in, te_lb = [], []
    for _ in range(n_test):
        seq, s = inst()
        inp, lab = encode(seq, s)
        te_in.append(inp)
        te_lb.append(lab)
    return tr_in, tr_lb, te_in, te_lb


# ================= 候选 3：拉丁方补全（数独对偶） =================

def gen_latin(rng):
    """随机拉丁方：首行打乱 + 行轮换 + 列轮换（群构造，覆盖主要同构类）"""
    base = np.arange(1, 10)
    rng.shuffle(base)
    rows = np.array([(base + i - 1) % 9 + 1 for i in range(9)])
    return rows[:, rng.permutation(9)]


def latin_ok(g, r, c, v):
    return v not in g[r] and v not in g[:, c]


def count_solutions(grid, limit=2, budget=2_000_000):
    """回溯计数（MRV 启发），达 limit 即停。超预算抛 RuntimeError。"""
    g = grid.copy()
    nodes = 0
    def rec():
        nonlocal nodes
        nodes += 1
        if nodes > budget:
            raise RuntimeError("budget")
        best, br, bc, bcands = 10, -1, -1, []
        for r in range(9):
            for c in range(9):
                if g[r, c] == 0:
                    cands = [v for v in range(1, 10) if latin_ok(g, r, c, v)]
                    if len(cands) < best:
                        best, br, bc, bcands = len(cands), r, c, cands
                        if best == 1:
                            break
            if best == 1:
                break
        if br < 0:
            return 1
        if not bcands:
            return 0
        n = 0
        for v in bcands:
            g[br, bc] = v
            n += rec()
            g[br, bc] = 0
            if n >= limit:
                break
        return n
    return rec()


def carve(rng, full):
    """从完整解挖空到 KEEP_HINTS 提示，逐步保持唯一解。返回掩码(True=保留)。"""
    mask = np.ones((9, 9), bool)
    n_keep = 81
    cells = list(range(81))
    rng.shuffle(cells)
    for cell in cells:
        if n_keep <= KEEP_HINTS:
            break
        r, c = divmod(cell, 9)
        mask[r, c] = False
        g = full.copy()
        g[~mask] = 0
        try:
            if count_solutions(g, 2) != 1:
                mask[r, c] = True
            else:
                n_keep -= 1
        except RuntimeError:                  # 预算超 → 恢复该格
            mask[r, c] = True
    return mask, n_keep


def masked_input(full, mask):
    inp = np.full(SEQ_LEN, PAD, dtype=np.int64)
    inp[:81] = np.where(mask.ravel(), full.ravel(), PAD)
    return inp


def carve_job(seed):
    """并行 worker：seed → (模板解, 掩码, 提示数)。rng 全程本地，可复现。"""
    rng = np.random.default_rng(seed)
    full = gen_latin(rng)
    mask, nk = carve(rng, full)
    return full, mask, nk


def gen_e3(n_struct, n_aug, n_test, n_workers=8):
    """候选 3：结构 = 挖空掩码（并行 carve）；实例 = 同掩码 × σ(模板解)，
    σ = S9 随机置换（唯一性保真，零验证）。test = 新掩码 × 恒等置换。
    label = 完整解 81 位（1-9）。"""
    from multiprocessing import Pool
    seeds = [100_000 + i * 7919 for i in range(n_struct + n_test)]
    with Pool(n_workers) as p:
        results = list(p.imap(carve_job, seeds, chunksize=8))
    structs = results[:n_struct]
    tests = results[n_struct:]

    sig_rng = np.random.default_rng(0)          # σ 生成（与 carve rng 分离）
    tr_in, tr_lb = [], []
    for full, mask, _ in structs:
        sigmas = [np.arange(9) + 1]             # 恒等置换（原模板实例）
        for _ in range(n_aug - 1):
            sigmas.append(sig_rng.permutation(9) + 1)
        for sigma in sigmas:
            inst = sigma[full - 1]              # σ(模板解)：唯一性保真
            tr_in.append(masked_input(inst, mask))
            lab = np.full(SEQ_LEN, PAD, dtype=np.int64)
            lab[:81] = inst.ravel()
            tr_lb.append(lab)
    te_in, te_lb = [], []
    for full, mask, nk in tests:
        g = full.copy()
        g[~mask] = 0
        assert count_solutions(g, 2) == 1       # sanity（carve 已保证）
        te_in.append(masked_input(full, mask))
        lab = np.full(SEQ_LEN, PAD, dtype=np.int64)
        lab[:81] = full.ravel()
        te_lb.append(lab)
    return tr_in, tr_lb, te_in, te_lb


# ================= 主流程 =================

def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", type=int, choices=[1, 2, 3], required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    candidates = {1: "domain-e-c1-prefix-muldiv",
                  2: "domain-e-c2-signal-infer",
                  3: "domain-e-c3-latin"}
    OUT_DIR = args.out or f"../data/{candidates[args.candidate]}"
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(42)
    t0 = time.time()

    if args.candidate == 1:
        tr_in, tr_lb, te_in, te_lb = gen_e1(rng, N_STRUCT, N_AUG, N_TEST)
        note = ("M0.4 domain E candidate 1: prefix-notation stack arithmetic, "
                "MUL=10 DIV=11 PAD=12; result at output pos 0; "
                "near-domain variant of B (syntax reversed + op semantics changed)")
        exact_sem = "pos 0"
    elif args.candidate == 2:
        tr_in, tr_lb, te_in, te_lb = gen_e2(rng, 10000, 2000)
        note = ("M0.4 domain E candidate 2: signal inference (reversed info flow); "
                "10 noisy signals at pos 0-9, type cue at pos 10 (10=honest 11=deceiver), "
                "state at output pos 0; ceiling ~1.0 via majority+shift")
        exact_sem = "pos 0"
    else:
        tr_in, tr_lb, te_in, te_lb = gen_e3(N_STRUCT, N_AUG, N_TEST)
        note = ("M0.4 domain E candidate 3: 9x9 Latin square completion (sudoku dual, "
                "no box constraint); mask = carved structure (unique-solution), "
                "instances = symbol permutations sigma(template) (uniqueness preserved); "
                "blanks = PAD; full grid label; cell-based exact semantics TBD at E6a")
        exact_sem = "full grid (cell basis)"

    write("train", tr_in, tr_lb)
    write("test", te_in, te_lb)

    meta = {"pad_id": PAD, "ignore_label_id": PAD, "blank_identifier_id": 0,
            "vocab_size": 13, "seq_len": SEQ_LEN, "num_puzzle_identifiers": 1,
            "total_groups": len(tr_in), "mean_puzzle_examples": 1.0,
            "total_puzzles": len(tr_in), "sets": ["all"],
            "candidate": args.candidate,
            "exact_semantics": exact_sem,
            "note": note}
    json.dump(meta, open(os.path.join(OUT_DIR, "dataset.json"), "w"), indent=2)
    print(f"meta -> {os.path.join(OUT_DIR, 'dataset.json')} · "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
