"""domain_distance.py — 域间结构距离计算器（M0.4 后档 · 决策 1 落地）

⚠️ 定位：**辅助报告**用。相似度梯度的冻结物 = 行为距离
（零样本迁移率矩阵，见 m0_4_zero_shot_matrix.py），结构距离不参与冻结。

语义：
- family 不同 -> 距离 1.0（跨族 = 最大距离，如 puzzle vs game）
- family 相同 -> 加权特征距离：
    syntax 不同       +0.40
    ops 交集比例      +0.25 * (1 - jaccard)
    depth 范围差异    +0.20 * 归一化差
    value_range 差    +0.15 * 归一化差
  （权重是启发式，仅辅助解释用）

输出：4x4 矩阵（A-D；E/F/G 待填）
"""
from __future__ import annotations

import itertools

from domain_params import DOMAIN_PARAMS

WEIGHTS = {
    "syntax": 0.40,
    "ops": 0.25,
    "depth": 0.20,
    "value_range": 0.15,
}


def _jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0  # 都无 ops -> 该项无差异
    return len(sa & sb) / len(sa | sb)


def _range_norm_diff(a, b):
    """值域归一化差异：相对跨度差 / 2（∈[0,1]），None 视为不参与。"""
    if a is None or b is None:
        return 0.0
    lo_a, hi_a = a
    lo_b, hi_b = b
    span_a = max(hi_a - lo_a, 1)
    span_b = max(hi_b - lo_b, 1)
    return abs(span_a - span_b) / max(span_a, span_b)


def _depth_norm_diff(a, b):
    if a is None or b is None:
        return 0.0
    lo_a, hi_a = a
    lo_b, hi_b = b
    return abs(hi_a - lo_a - (hi_b - lo_b)) / max(hi_a - lo_a, hi_b - lo_b, 1)


def structural_distance(p1: dict, p2: dict) -> float:
    if p1["family"] != p2["family"]:
        return 1.0
    d = 0.0
    if p1["syntax"] != p2["syntax"]:
        d += WEIGHTS["syntax"]
    d += WEIGHTS["ops"] * (1.0 - _jaccard(p1["ops"], p2["ops"]))
    d += WEIGHTS["depth"] * _depth_norm_diff(p1["depth"], p2["depth"])
    d += WEIGHTS["value_range"] * _range_norm_diff(
        p1["value_range"], p2["value_range"])
    return round(min(d, 1.0), 3)


def distance_matrix(domains: list[str]) -> dict[str, dict[str, float]]:
    m = {d: {} for d in domains}
    for d1, d2 in itertools.combinations(domains, 2):
        p1, p2 = DOMAIN_PARAMS[d1], DOMAIN_PARAMS[d2]
        if p1 is None or p2 is None:
            continue
        d = structural_distance(p1, p2)
        m[d1][d2] = d
        m[d2][d1] = d
    return m


if __name__ == "__main__":
    domains = [d for d in ["A", "B", "C", "D"] if DOMAIN_PARAMS[d]]
    mat = distance_matrix(domains)
    print("structural distance matrix (A-D, 辅助报告用，非冻结物):")
    print("    " + "  ".join(f"{d:>5}" for d in domains))
    for d1 in domains:
        row = " ".join(
            f"{mat[d1].get(d2, 0.0):5.2f}" if d1 != d2 else "  0.00"
            for d2 in domains)
        print(f"{d1}: {row}")
