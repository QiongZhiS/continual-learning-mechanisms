"""M0.4a domain-C generator: composition domain (feature->entity inference, carrier for the E4 platypus test + E6a sequence member)

Design (isomorphic to the domains A/B format):
- entity = color x animal = 36 combinations; vocab: colors 0-5 + animals 6-11 + PAD=12 (vocab_size=13)
- input 81 slots: <color> <animal> PAD...; output 81 slots: slot 0 = x, slot 1 = y, rest PAD
- nonlinear f (trivial-composition defense, per pre-registration): x = (c*(a+1)) % 10, y = (a*(c+1)) % 10
  multiplicative interaction (product term of c, a) — linear models cannot extrapolate unseen combinations (the E4 criterion needs a non-trivial task)
- train 24 combinations x N_AUG repeats; test 12 novel combinations (platypus test: zero-shot inference)
  (design revision: 16->36 combinations, test 4->12 samples, E4/E6a criterion noise ±25pp->±14pp)
- validation: (1) labels consistent with f (2) linear model fails to extrapolate unseen combinations (trivial defense) (3) format isomorphism"""
import json
import os

import numpy as np

OUT = "../data/domain-c-combo"
SEQ_LEN = 81
PAD = 12
N_AUG = 500  # repeats per combination (multiple examples for rule learning, aligned with domain A/B augmentation semantics)
COLORS = list(range(6))      # red orange yellow green blue purple
ANIMALS = list(range(6, 12)) # fish elephant cat bird snake dragon


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
    combos = [(c, a) for c in COLORS for a in ANIMALS]  # 36 combinations
    rng.shuffle(combos)
    train_combos, test_combos = combos[:24], combos[24:]  # 24 train / 12 platypus

    # (1) label correctness (independent check of f)
    for c, a in combos:
        inp, lab = encode(c, a)
        assert inp[0] == c and inp[1] == a
        assert lab[0] == f_x(c, a) and lab[1] == f_y(c, a)
    print(f"label check: {len(combos)}/{len(combos)} combos OK", flush=True)

    # (2) trivial defense: a linear model (no product term) must fail to extrapolate unseen combinations
    X = np.array([[c, a] for c, a in train_combos], dtype=float)
    yt = np.array([f_x(c, a) for c, a in train_combos], dtype=float)
    A = np.column_stack([np.ones(len(X)), X])  # 1, c, a (no c*a product term)
    coef, *_ = np.linalg.lstsq(A, yt, rcond=None)
    Xt = np.array([[c, a] for c, a in test_combos], dtype=float)
    pred = np.round(np.column_stack([np.ones(len(Xt)), Xt]) @ coef)
    true = np.array([f_x(c, a) for c, a in test_combos], dtype=float)
    lin_acc = (pred == true).mean()
    print(f"trivial-defense: linear extrapolation on unseen combos: "
          f"acc={lin_acc:.2f} (must be < 1.0; {len(combos)} combos, "
          f"{len(train_combos)} train -> {len(test_combos)} unseen)",
          flush=True)
    assert lin_acc < 1.0, "f is linearly solvable -> E4 trivial-composition defense failed; replace f"

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
