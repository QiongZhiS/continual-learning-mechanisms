"""M1a domain-B generator: RPN stack-machine arithmetic (postfix expression evaluation)

Design (isomorphic to the domain-A sudoku format):
- vocab: digits 0-9 (10) + ADD=10 + SUB=11 + PAD=12 -> vocab_size=13, pad_id=12
- all intermediate results in [0,9] (range-sampled, no infinite loops); ops v1 = {ADD, SUB}
- input 81 slots: RPN program left-aligned + PAD fill
- output 81 slots: slot 0 = result (0-9), rest PAD (ignore=pad_id -> loss only on the result slot)
- train: 1000 structures x 101 value instances = 101K (structure repetition = multiple examples of rule learning, aligned with the domain-A augmentation semantics)
- test: 2000 novel structures+instances (tests rule generalization, not memorization)
- depth in {2,3,4} (token counts 7/15/31, all <= 81)

--aug (train-side equivalence-class augmentation, control arm of the semantic-exploration discrimination):
inject train-side equivalence classes — for each (structure, value) instance enumerate ALL ADD-subtree-swap equivalent forms
(2^#ADD, capped at --aug-cap, default 16) into the training set. Swap = swap the ADD node's left/right subtrees,
answer mathematically unchanged (additive commutativity; the perturber was validated equivalent in m1_sem_explore.py).
- original sample stream identical to no-augmentation: augmentation consumes rng only AFTER test generation -> test byte-identical to baseline
- training set = union of full equivalence classes per instance (dedup; symmetric-subtree dedup may reduce below cap)
- every variant asserted == original answer by the standalone stack evaluator stack_eval (equivalence guard)"""
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
    """Fill values and return (token sequence, result value). Value range [0, hi] guaranteed by range sampling."""
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


# ---- round-3 equivalence-class augmentation (isomorphic to the m1_sem_explore.py perturber) ----

def parse_seq(seq):
    """RPN tokens -> AST (leaves carry values). Programs are guaranteed single-root."""
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
    """AST -> RPN token list (consistent with fill: left + right + op)"""
    if node.op < 10:
        return [node.op]
    return serialize(node.left) + serialize(node.right) + [node.op]


def stack_eval(seq):
    """Standalone stack evaluator (for equivalence assertions; different implementation from fill's recursion)"""
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
    """mask bit i = 1 -> swap the left/right subtrees of the i-th ADD node (left-preorder numbering).
    Answer unchanged (additive commutativity)."""
    out = copy_tree(tree)
    adds = collect_adds(out, [])
    for i, n in enumerate(adds):
        if (mask >> i) & 1:
            n.left, n.right = n.right, n.left
    return out


def swap_variants(seq, cap):
    """All ADD-swap equivalent forms of one (structure, value) instance (incl. original program; capped at cap forms).
    Returns (list of token lists, deduped form count). Equivalence asserted by the caller with stack_eval."""
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
    """Append all ADD-swap equivalent forms of each training sample (deduped; skip the original).
    Deterministic (mask enumeration, consumes no rng). Returns (augmented inputs, labels, augmentation stats).
    Equivalence asserted per sample."""
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
                continue  # drop non-equivalent forms (should not happen; defensive guard)
            inp, lab = encode(s, v)
            out_in.append(inp)
            out_lb.append(lab)
        n_added += nf - 1  # minus 1 = original program already in the baseline set
    print(f"aug: +{n_added} variants (orig {n_orig} -> total {len(out_in)}), "
          f"equiv_bad={bad}, {time.time()-t0:.0f}s", flush=True)
    assert bad == 0, "equivalence assertion failed: ADD swap changed the answer"
    return out_in, out_lb, {"n_added": n_added, "n_total": len(out_in),
                            "n_orig": n_orig, "equiv_bad": bad,
                            "cap": cap}


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", action="store_true",
                    help="round-3: train-side ADD equivalence-class injection (full-class enumeration)")
    ap.add_argument("--aug-cap", type=int, default=16, help="max equivalent forms per instance")
    ap.add_argument("--out", default=OUT, help="data output directory")
    args = ap.parse_args()
    OUT_DIR = args.out
    os.makedirs(OUT_DIR, exist_ok=True)

    rng = np.random.default_rng(42)

    # train: structures x value instances (original sample stream, byte-identical to baseline)
    structs = [sample_structure(rng, int(rng.integers(2, 5))) for _ in range(N_STRUCT)]
    inputs, labels = [], []
    for node in structs:
        for _ in range(N_AUG):
            seq, v = fill(rng, node, MAX_VAL)
            assert 0 <= v <= MAX_VAL
            inp, lab = encode(seq, v)
            inputs.append(inp)
            labels.append(lab)

    # test: novel structures (generated before augmentation -> rng state matches baseline -> byte-identical)
    test_inputs, test_labels = [], []
    for _ in range(N_TEST):
        seq, v = make_instance(rng)
        inp, lab = encode(seq, v)
        test_inputs.append(inp)
        test_labels.append(lab)
    write("test", test_inputs, test_labels)

    aug_info = None
    if args.aug:
        # augmentation runs after test: rng consumption only affects train-set expansion, not test
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
        meta["note"] += "; round-3 discriminator equivariant-class injection"
    json.dump(meta, open(os.path.join(OUT_DIR, "dataset.json"), "w"), indent=2)
    print(f"meta -> {os.path.join(OUT_DIR, 'dataset.json')}", flush=True)


if __name__ == "__main__":
    main()
