"""M0.4 domain-D instantiation: D1 iterated prisoner's dilemma IPD (4th domain in the E6a sequence, gap 4: social intelligence)

Spec (domain-D instance design frozen; this script fixes the instance parameters):
- 2 players x 100 rounds; payoff matrix R=3 / S=0 / T=5 / P=1 (payoff[my, opp])
- opponent pool of 8: {always-cooperate, always-defect, tit-for-tat,
  generous tit-for-tat (tolerance 0.1), random, punisher, masquerader-latent, masquerader-probabilistic}
  - punisher (frozen definition): cooperate first; if opponent played D last round -> single-round D punishment this round; opponent C -> resume C
  - latent: C for the first 30 rounds, then all D
  - probabilistic: 0.8 C / 0.2 D per round, no retaliation mechanism
- observation L1 (frozen layout, vocab: 0=C 1=D 2-9=numeric 12=PAD):
  slot 0 = round bucket round//10 (0-9)
  slots 1-10 = my last 10 actions (slot 1 = most recent; empty history = PAD)
  slots 11-20 = opponent's last 10 actions
  slot 21 = my cumulative score bucket min(score//50, 9)
  slot 22 = opponent cumulative score bucket
  slot 23+ = PAD
- label: slot 0 = this model's action (0/1), rest PAD (reference dataset; training protocol to be pre-registered before E6a)
- criterion anchor: adaptation premium = policy expected payoff > TFT baseline x (1+eps), eps=0.1 (sensitivity 0.05/0.2 reported)
- ceiling = analytical best-response rollout (averaged over multiple random-opponent games), no mechanism-free baseline calibration needed
- reference dataset: 50 games x 100 rounds per opponent (best-response rollout) = 40K (observation, action) pairs"""
import json
import os

import numpy as np

OUT = "../data/domain-d-ipd"
N_ROUNDS = 100
N_GAMES = 50          # games per opponent in the reference dataset
CEIL_GAMES = 200      # ceiling rollout games (denoising over random opponents)
C, D = 0, 1
PAD = 12
SEQ_LEN = 81
R, S, T, P = 3, 0, 5, 1
PAYOFF = np.array([[R, S], [T, P]])          # payoff[my_action, opp_action]
SLEEPER_CUT = 30     # sleeper cooperation rounds
GTFT_EPS = 0.1       # generous-TFT tolerance
PD_PROB_C = 0.8      # probabilistic-type cooperation probability
SCORE_BUCKET = 50    # score-bucket granularity (0-500 -> 0-9)

OPPONENTS = ["always-cooperate", "always-defect", "tit-for-tat",
             "generous-tft", "random", "punisher", "sleeper", "prob-deceiver"]


# ---- opponent strategies: fn(round_idx, my_hist, opp_hist, rng) -> action ----

def strat_always_cooperate(r, mh, oh, rng):
    return C


def strat_always_defect(r, mh, oh, rng):
    return D


def strat_tft(r, mh, oh, rng):
    return C if r == 0 else oh[-1]


def strat_generous_tft(r, mh, oh, rng):
    if r == 0 or oh[-1] == C:
        return C
    return C if rng.random() < GTFT_EPS else D  # epsilon-probability forgiveness after betrayal


def strat_random(r, mh, oh, rng):
    return int(rng.random() < 0.5)


def strat_punisher(r, mh, oh, rng):
    if r == 0:
        return C
    return D if oh[-1] == D else C  # recover after the single-round punishment


def strat_sleeper(r, mh, oh, rng):
    return C if r < SLEEPER_CUT else D


def strat_prob_deceiver(r, mh, oh, rng):
    return C if rng.random() < PD_PROB_C else D


STRATS = {"always-cooperate": strat_always_cooperate,
          "always-defect": strat_always_defect,
          "tit-for-tat": strat_tft,
          "generous-tft": strat_generous_tft,
          "random": strat_random,
          "punisher": strat_punisher,
          "sleeper": strat_sleeper,
          "prob-deceiver": strat_prob_deceiver}


# ---- best-response strategies (ceiling; derived analytically, verified by rollout) ----

def optimal_vs(opp):
    """Opponent class -> best-response strategy (analytical). Returns fn(round_idx, my_hist, opp_hist, rng)."""
    if opp == "always-cooperate":
        return lambda r, mh, oh, rng: D                      # all-D: 500
    if opp == "always-defect":
        return lambda r, mh, oh, rng: D                      # all-D: 100 (C scores 0)
    if opp in ("tit-for-tat", "punisher"):
        return lambda r, mh, oh, rng: D if r == N_ROUNDS - 1 else C  # 99C+final-D: 302
    if opp == "generous-tft":
        return lambda r, mh, oh, rng: D if r == N_ROUNDS - 1 else C  # all-C+final-D: ~302
    if opp == "random":
        return lambda r, mh, oh, rng: D                      # all-D: expected 300
    if opp == "sleeper":
        # all-D is the accepted best response (acceptance record M0.4-2026-08-12):
        # 30 x T=5 + 70 x P=1 = 220. (30C+70D would score only 30x3 + 70x1 = 160 --
        # sleeper defects late, so early cooperation never collects T; the
        # identification value of detecting the sleeper = 220 - 159 TFT baseline.)
        return lambda r, mh, oh, rng: D  # all-D: 30*5 + 70*1 = 220
    if opp == "prob-deceiver":
        return lambda r, mh, oh, rng: D                      # all-D: expected 420
    raise ValueError(opp)


# ---- environment and encoding ----

def play(policy, opp, rng, record=False):
    """One IPD game. policy/opp = fn(r, my_hist, opp_hist, rng).
    record=True -> returns (observation sequence, action sequence, payoff)."""
    my_hist, opp_hist = [], []
    my_score = opp_score = 0
    if record:
        obs_list, act_list = [], []
    for r in range(N_ROUNDS):
        a = policy(r, my_hist, opp_hist, rng)
        b = STRATS[opp](r, opp_hist, my_hist, rng)
        my_score += int(PAYOFF[a, b])
        opp_score += int(PAYOFF[b, a])
        if record:
            obs_list.append(encode_obs(r, my_hist, opp_hist, my_score, opp_score))
            act_list.append(a)
        my_hist.append(a)
        opp_hist.append(b)
    if record:
        return obs_list, act_list, my_score
    return my_score


def encode_obs(round_idx, my_hist, opp_hist, my_score, opp_score):
    inp = np.full(SEQ_LEN, PAD, dtype=np.int64)
    inp[0] = min(round_idx // 10, 9)
    for i, a in enumerate(reversed(my_hist[-10:])):
        inp[1 + i] = a
    for i, a in enumerate(reversed(opp_hist[-10:])):
        inp[11 + i] = a
    inp[21] = min(my_score // SCORE_BUCKET, 9)
    inp[22] = min(opp_score // SCORE_BUCKET, 9)
    return inp


def encode_label(action):
    lab = np.full(SEQ_LEN, PAD, dtype=np.int64)
    lab[0] = action
    return lab


# ---- outputs ----

def write_dataset():
    d = os.path.join(OUT, "train")
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(0)
    inputs, labels = [], []
    for opp in OPPONENTS:
        pol = optimal_vs(opp)
        for _ in range(N_GAMES):
            obs, acts, score = play(pol, opp, rng, record=True)
            inputs.extend(obs)
            labels.extend(encode_label(a) for a in acts)
    n = len(inputs)
    np.save(os.path.join(d, "all__inputs.npy"), np.stack(inputs))
    np.save(os.path.join(d, "all__labels.npy"), np.stack(labels))
    np.save(os.path.join(d, "all__puzzle_identifiers.npy"), np.zeros(n, dtype=np.int32))
    print(f"train: {n} samples ({N_GAMES} games x {len(OPPONENTS)} opponents x {N_ROUNDS} rounds) -> {d}",
          flush=True)


def ceiling():
    """Per opponent: best-response expected payoff (CEIL_GAMES average) + TFT baseline expected payoff."""
    rng = np.random.default_rng(0)
    rng_tft = np.random.default_rng(1)
    ceils, tfts = {}, {}
    for opp in OPPONENTS:
        scores = [play(optimal_vs(opp), opp, rng) for _ in range(CEIL_GAMES)]
        t_scores = [play(strat_tft, opp, rng_tft) for _ in range(CEIL_GAMES)]
        ceils[opp] = round(float(np.mean(scores)), 2)
        tfts[opp] = round(float(np.mean(t_scores)), 2)
    return ceils, tfts


def main():
    os.makedirs(OUT, exist_ok=True)
    write_dataset()

    ceils, tfts = ceiling()
    tft_pool_mean = round(float(np.mean(list(tfts.values()))), 2)
    print("ceiling (best-response expected payoff) vs TFT baseline (pool-mean baseline = "
          f"{tft_pool_mean}）:", flush=True)
    for opp in OPPONENTS:
        print(f"  {opp:18s} ceil={ceils[opp]:7.2f}  tft={tfts[opp]:7.2f}", flush=True)

    # analytical assertions (ceiling sanity: matches hand computation)
    assert abs(ceils["always-cooperate"] - 500) < 1e-6, ceils["always-cooperate"]
    assert abs(ceils["always-defect"] - 100) < 1e-6
    assert abs(ceils["sleeper"] - 220) < 1e-6  # all-D: 30*5 + 70*1
    # tolerance 3.0 = ~2x the 200-game Monte-Carlo mean std (~1.1-1.4); 1.0 was flaky
    assert abs(ceils["prob-deceiver"] - 420) < 3.0  # random opponent, denoised over 200 games
    assert abs(ceils["random"] - 300) < 3.0
    assert abs(ceils["tit-for-tat"] - 302) < 1e-6
    assert abs(ceils["punisher"] - 302) < 1e-6

    meta = {"pad_id": PAD, "ignore_label_id": PAD, "blank_identifier_id": 0,
            "vocab_size": 13, "seq_len": SEQ_LEN, "num_puzzle_identifiers": 1,
            "total_groups": len(OPPONENTS) * N_GAMES, "mean_puzzle_examples": 1.0,
            "total_puzzles": len(OPPONENTS) * N_GAMES, "sets": ["all"],
            "game": {"type": "IPD", "n_rounds": N_ROUNDS,
                     "payoff": {"R": R, "S": S, "T": T, "P": P}},
            "opponents": OPPONENTS,
            "opponent_params": {"sleeper_cut": SLEEPER_CUT, "gtft_eps": GTFT_EPS,
                                "pd_prob_c": PD_PROB_C,
                                "punisher": "single-round-punish"},
            "obs_layout": "pos0=round//10 pos1-10=my_last10 pos11-20=opp_last10 "
                          "pos21=my_score//50 pos22=opp_score//50, C=0 D=1 PAD=12",
            "criterion": {"name": "adaptation_premium",
                          "eps": 0.1, "eps_sensitivity": [0.05, 0.2],
                          "base": "tit-for-tat pool mean"},
            "ceiling_optimal_response": ceils,
            "tft_baseline_per_opponent": tfts,
            "tft_pool_mean": tft_pool_mean,
            "note": "M0.4 domain D1: IPD; reference dataset = optimal-response "
                    "trajectories (training scheme TBD at E6a)"}
    json.dump(meta, open(os.path.join(OUT, "dataset.json"), "w"), indent=2)
    print("meta -> " + os.path.join(OUT, "dataset.json"), flush=True)


if __name__ == "__main__":
    main()
