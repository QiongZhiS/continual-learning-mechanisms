"""M0.4 域 D 实例化：D1 囚徒困境 IPD（E6a 序列第 4 域 · 空白4 社会智能）

规格（域D-实例化设计.md v0.1，已拍板；本脚本冻结实例参数）：
- 2 人 × 100 轮；收益矩阵 R=3 / S=0 / T=5 / P=1（payoff[my, opp]）
- 对手池 8 个：{always-cooperate, always-defect, tit-for-tat,
  慷慨 tit-for-tat(容忍 0.1), random, 惩罚者, 伪装者-潜伏型, 伪装者-概率型}
  - 惩罚者（冻结定义）：先合作；对手上轮 D → 本轮单轮惩罚 D，对手 C → 恢复 C
  - 潜伏型：前 30 轮 C，之后全 D
  - 概率型：每轮 0.8 C / 0.2 D，无报复机制
- 观测 L1（冻结布局，vocab: 0=C 1=D 2-9=数值 12=PAD）：
  位 0 = 轮数分桶 round//10（0-9）
  位 1-10 = 我方最近 10 轮动作（位 1 = 最近一轮；空历史 = PAD）
  位 11-20 = 对手最近 10 轮动作
  位 21 = 我方累计收益分桶 min(score//50, 9)
  位 22 = 对手累计收益分桶
  位 23+ = PAD
- label：位 0 = 本模型动作（0/1），其余 PAD（参考数据集；训练方案 E6a 前另定）
- 判据锚：适配性溢价 = 策略期望收益 > TFT 基线 × (1+ε)，ε=0.1（灵敏度 0.05/0.2 报告）
- 天花板 = 解析最优响应 rollout（随机对手多局平均），不依赖无机制基线标定
- 参考数据集：每对手 50 局 × 100 轮（最优响应策略 rollout）= 40K (观测, 动作) 对
"""
import json
import os

import numpy as np

OUT = "../data/domain-d-ipd"
N_ROUNDS = 100
N_GAMES = 50          # 参考数据集每对手局数
CEIL_GAMES = 200      # 天花板 rollout 局数（随机对手降噪）
C, D = 0, 1
PAD = 12
SEQ_LEN = 81
R, S, T, P = 3, 0, 5, 1
PAYOFF = np.array([[R, S], [T, P]])          # payoff[my_action, opp_action]
SLEEPER_CUT = 30     # 潜伏型合作轮数
GTFT_EPS = 0.1       # 慷慨 TFT 容忍度
PD_PROB_C = 0.8      # 概率型合作概率
SCORE_BUCKET = 50    # 收益分桶粒度（0-500 → 0-9）

OPPONENTS = ["always-cooperate", "always-defect", "tit-for-tat",
             "generous-tft", "random", "punisher", "sleeper", "prob-deceiver"]


# ---- 对手策略：fn(round_idx, my_hist, opp_hist, rng) -> action ----

def strat_always_cooperate(r, mh, oh, rng):
    return C


def strat_always_defect(r, mh, oh, rng):
    return D


def strat_tft(r, mh, oh, rng):
    return C if r == 0 else oh[-1]


def strat_generous_tft(r, mh, oh, rng):
    if r == 0 or oh[-1] == C:
        return C
    return C if rng.random() < GTFT_EPS else D  # 被背叛后 ε 概率宽恕


def strat_random(r, mh, oh, rng):
    return int(rng.random() < 0.5)


def strat_punisher(r, mh, oh, rng):
    if r == 0:
        return C
    return D if oh[-1] == D else C  # 单轮惩罚后恢复


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


# ---- 最优响应策略（天花板，解析推导后 rollout 验证） ----

def optimal_vs(opp):
    """对手类 → 最优响应策略（解析）。返回 fn(round_idx, my_hist, opp_hist, rng)。"""
    if opp == "always-cooperate":
        return lambda r, mh, oh, rng: D                      # 全 D：500
    if opp == "always-defect":
        return lambda r, mh, oh, rng: D                      # 全 D：100（C 得 0）
    if opp in ("tit-for-tat", "punisher"):
        return lambda r, mh, oh, rng: D if r == N_ROUNDS - 1 else C  # 99C+终局D：302
    if opp == "generous-tft":
        return lambda r, mh, oh, rng: D if r == N_ROUNDS - 1 else C  # 全C+终局D：~302
    if opp == "random":
        return lambda r, mh, oh, rng: D                      # 全 D：期望 300
    if opp == "sleeper":
        return lambda r, mh, oh, rng: D if r >= SLEEPER_CUT else C  # 30C+70D：440
    if opp == "prob-deceiver":
        return lambda r, mh, oh, rng: D                      # 全 D：期望 420
    raise ValueError(opp)


# ---- 环境与编码 ----

def play(policy, opp, rng, record=False):
    """一局 IPD。policy/opp = fn(r, my_hist, opp_hist, rng)。
    record=True → 返回 (观测序列, 动作序列, 收益)。"""
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


# ---- 产出 ----

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
    print(f"train: {n} samples ({N_GAMES} 局 × {len(OPPONENTS)} 对手 × {N_ROUNDS} 轮) -> {d}",
          flush=True)


def ceiling():
    """每对手：最优响应期望收益（CEIL_GAMES 局平均）+ TFT 基线期望收益。"""
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
    print("天花板（最优响应期望收益）vs TFT 基线（池均值基线 = "
          f"{tft_pool_mean}）:", flush=True)
    for opp in OPPONENTS:
        print(f"  {opp:18s} ceil={ceils[opp]:7.2f}  tft={tfts[opp]:7.2f}", flush=True)

    # 解析断言（天花板 sanity：与手算一致）
    assert abs(ceils["always-cooperate"] - 500) < 1e-6, ceils["always-cooperate"]
    assert abs(ceils["always-defect"] - 100) < 1e-6
    assert abs(ceils["sleeper"] - 440) < 1e-6
    assert abs(ceils["prob-deceiver"] - 420) < 1.0  # 随机对手，200 局降噪
    assert abs(ceils["random"] - 300) < 1.0
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
