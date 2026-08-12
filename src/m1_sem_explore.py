"""M1 semantic-level exploration (E1 failure-branch action B): equivalence-semantic perturbation (ESE) majority voting

Failure mechanism (falsified antecedent): domain-B z_L noise (latent noise injected at layer L) = homogeneous jitter
(no incremental information across rollouts, voting 0.478 = deep-expansion baseline). Unresolved question: is the model's error
"input-positional" (sensitive to token positions / operand order) or "rule-based" (learned the wrong rule)?

ESE discrimination design: for program P, generate K **semantically equivalent perturbations** (swap ADD subtrees via additive commutativity,
answer mathematically unchanged), rollout each variant D=48, majority vote.
- ESE vote > K=1 D=48 baseline (0.478) -> error is input-positional, semantic perturbation breaks
  systematicity -> semantic-level exploration effective (input-side perturbation is a domain-agnostic exploration carrier)
- ≈ baseline -> error is rule-based (the model fails equivalently on equivalent programs), input-side perturbation cannot help
  -> semantic-level exploration fails, the failure is fundamental (needs a train-side fix)

Controls: sigma=0 pure semantic / sigma=0.2 semantic + latent mixture (does z_L noise add gain on variants?).
Variant equivalence verified by the standalone stack evaluator at the smoke stage (variant answer must == original answer)."""
import json
import os
import sys
import time

import numpy as np
import torch

from eval_ptrm import load_test_data, load_model

CKPT = "outputs/2026-08-10/m1_domainB/step_4000"
DATA = "../data/domain-b-rpn"
OUT = "outputs/2026-08-10/m1_domainB/sem_explore.json"
K, D, N = 30, 48, 500
CHUNK = 10
ADD, SUB, PAD = 10, 11, 12


class Node:
    """op < 10 = leaf value; op in {ADD, SUB} = internal node"""
    __slots__ = ("op", "left", "right")

    def __init__(self, op, left=None, right=None):
        self.op, self.left, self.right = op, left, right


def parse(seq):
    """RPN tokens -> AST (stack-machine reversal; programs guaranteed single-root)"""
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
    """AST -> RPN token list (consistent with m1_gen_domainB.fill: left + right + op)"""
    if node.op < 10:
        return [node.op]
    return serialize(node.left) + serialize(node.right) + [node.op]


def stack_eval(seq):
    """Standalone stack evaluator (smoke-verifies variant equivalence; different implementation from fill's recursion)"""
    st = []
    for t in seq:
        if t in (ADD, SUB):
            b, a = st.pop(), st.pop()
            st.append(a + b if t == ADD else a - b)
        else:
            st.append(int(t))
    return st[0]


def collect_add(node, acc):
    if node.op == ADD:
        collect_add(node.left, acc)
        collect_add(node.right, acc)
        acc.append(node)
    elif node.op == SUB:
        collect_add(node.left, acc)
        collect_add(node.right, acc)
    return acc


def copy_tree(node):
    if node.op < 10:
        return Node(node.op)
    return Node(node.op, copy_tree(node.left), copy_tree(node.right))


def random_walk(node, add_nodes, steps, rng):
    """Random walk of steps moves; each move swaps the subtrees of a randomly chosen ADD node"""
    out = copy_tree(node)
    for _ in range(steps):
        target = add_nodes[rng.integers(len(add_nodes))]
        target.left, target.right = target.right, target.left
    return out


def variants(node, K, rng):
    """K variants; **variant 0 = original program** (= K=1 D=48 paired baseline), rest = distinct
    random-walk variants (1-3 moves; perturbation space = ADD-node combinations). No ADD nodes (e.g.
    `a b -`) -> variant pool is just the original, filled honestly (semantic exploration structurally
    unavailable for that puzzle; caller reports stratified by pool_size). Returns (variants, pool_size)."""
    adds = collect_add(node, [])
    out, seen = [node], {tuple(serialize(node))}
    max_steps = min(3, len(adds))
    guard = 0
    while len(out) < K and guard < 20 * K:
        guard += 1
        if max_steps == 0:
            break  # no ADD: use only the original
        t = random_walk(node, adds, rng.integers(1, max_steps + 1), rng)
        s = tuple(serialize(t))
        if s not in seen:
            seen.add(s)
            out.append(t)
    while len(out) < K:
        out.append(node)  # perturbation space exhausted, fill with the original
    return out, len(seen)


def encode_variants(vs):
    arr = np.full((len(vs), 81), PAD, dtype=np.int64)
    for i, v in enumerate(vs):
        s = serialize(v)
        arr[i, :len(s)] = s
    return arr


def run_sigma(model, inputs, ids, labels0, sigma, rng):
    """Rollout K semantic variants + majority vote.
    Returns (acc_vote, pred0, pool_sizes); pred0[:,0] = original-program rollout (paired baseline)"""
    N0 = inputs.shape[0]
    pred0 = np.zeros((N0, K), dtype=np.int64)
    pool = np.zeros(N0, dtype=np.int64)
    t0 = time.time()
    with torch.inference_mode():
        for start in range(0, N0, CHUNK):
            bs = min(CHUNK, N0 - start)
            vlist = [variants(parse(inputs[i].cpu().numpy()), K, rng)
                     for i in range(start, start + bs)]
            vs = [v for v, _ in vlist]
            pool[start:start + bs] = [p for _, p in vlist]
            b = np.concatenate([encode_variants(v) for v in vs], axis=0)
            batch = {
                "inputs": torch.tensor(b, dtype=torch.int64, device="cuda"),
                "puzzle_identifiers": ids[start:start + bs].repeat_interleave(K),
            }
            with torch.device("cuda"):
                carry = model.initial_carry(batch)
            for _ in range(D):
                if sigma > 0:
                    carry.inner_carry.z_L = carry.inner_carry.z_L + \
                        sigma * torch.randn_like(carry.inner_carry.z_L)
                carry, outputs = model(carry, batch)
            pred0[start:start + bs] = \
                outputs["logits"].argmax(-1)[:, 0].view(bs, K).cpu().numpy()
    print(f"  sigma={sigma} rollouts done ({time.time()-t0:.0f}s)", flush=True)

    vote = np.array([np.bincount(pred0[i], minlength=13).argmax()
                     for i in range(N0)])
    n_unique = np.array([len(np.unique(pred0[i])) for i in range(N0)])
    return (vote == labels0).mean(), pred0, pool, n_unique


def main():
    N_use = int(sys.argv[1]) if len(sys.argv) > 1 else N
    meta = json.load(open(f"{DATA}/dataset.json"))
    inputs, labels, ids, idx = load_test_data(DATA, N_use, 0)
    labels0 = labels[:, 0].cpu().numpy()
    model = load_model(CKPT, "config/arch/trm.yaml", meta["vocab_size"],
                       meta["seq_len"], meta["num_puzzle_identifiers"])
    model.config.halt_max_steps = D

    rng = np.random.default_rng(0)

    # smoke self-check (N=20): parse/serialize round-trip + variant equivalence (standalone evaluator)
    bad_roundtrip = bad_equiv = 0
    for i in range(20):
        toks = inputs[i].cpu().numpy()
        toks = toks[toks != PAD]
        if serialize(parse(toks)) != list(toks):
            bad_roundtrip += 1
        ans = stack_eval(toks)
        if ans != labels0[i]:
            raise AssertionError(f"puzzle {i}: data mismatch {ans} vs {labels0[i]}")
        vs, _ = variants(parse(toks), 5, rng)
        for v in vs:
            if stack_eval(serialize(v)) != ans:
                bad_equiv += 1
    print(f"smoke: roundtrip_bad={bad_roundtrip} equiv_bad={bad_equiv} "
          f"(n=20, variants must match the original answer)", flush=True)
    assert bad_roundtrip == 0 and bad_equiv == 0, "ESE precondition failed"

    results = {"K": K, "D": D, "n": N_use,
               "ref_K1_D48_exact": 0.478}  # M1c K-curve (same eval set)
    for sigma in (0.0, 0.2):
        acc, pred0, pool, n_unique = run_sigma(model, inputs, ids, labels0,
                                               sigma, rng)
        cons = n_unique == 1
        r = {
            "ese_vote_exact": round(float(acc), 4),
            "paired_base_exact": round(float((pred0[:, 0] == labels0).mean()), 4),
            "mean_unique_variants": round(float(n_unique.mean()), 2),
            "pct_single_variant_puzzles": round(float(cons.mean()), 3),
            "consistent_subset_acc": round(float((pred0[cons, 0] == labels0[cons]).mean()), 4),
            "inconsistent_subset_acc": round(float((pred0[~cons, 0] == labels0[~cons]).mean()), 4),
            "pools": {},
        }
        # stratified by variant-pool size (paired: original-program rollout vs vote on the same puzzle)
        for lo, hi in [(1, 1), (2, 2), (3, 99)]:
            m = (pool >= lo) & (pool <= hi)
            if m.sum() == 0:
                continue
            v = np.array([np.bincount(pred0[i], minlength=13).argmax()
                          for i in np.where(m)[0]])
            r["pools"][f"pool{lo}-{hi}"] = {
                "n": int(m.sum()),
                "base_acc": round(float((pred0[m, 0] == labels0[m]).mean()), 4),
                "vote_acc": round(float((v == labels0[m]).mean()), 4),
            }
        results[f"sigma={sigma}"] = r
        print(json.dumps({f"sigma={sigma}": r}, indent=2), flush=True)
        if sigma == 0.0:  # dump details for later analysis
            np.savez("outputs/2026-08-10/m1_domainB/sem_explore_detail.npz",
                     pred0=pred0, pool=pool, labels0=labels0)
    json.dump(results, open(OUT, "w"), indent=2)
    print("saved -> " + OUT, flush=True)


if __name__ == "__main__":
    main()
