"""domain_params.py — 全部域的机器可读结构化参数（M0.4 后档 · 决策 1 落地）

用途（辅助报告 + 渐变域插值接口）：
1. domain_distance.py 从本文件读参数，算"结构距离矩阵"（辅助报告用）
2. 渐变域插值对（域 B ↔ 前缀变体）需要同家族参数路径
3. E/F/G 占位：域 E 选定后 + F/G 延伸设计后填入

冻结规则（预注册）：
- 相似度梯度的**冻结物** = 行为距离（零样本迁移率矩阵，m0_4_zero_shot_matrix.py）
- 结构距离只作辅助报告，不参与冻结
- A-D 参数 = 已交付生成器的实际值（域 B: m1_gen_domainB.py；域 C: m0_4_gen_domainC.py；
  域 A 数独 = M0.1 复现配置；域 D = 域D-实例化设计.md v0.1）
"""
from __future__ import annotations

# family 取值：
#   puzzle   — 谜题域（格/约束填充）
#   seq-map  — 序列映射域（程序/序列 → 值）
#   game     — 博弈域（状态 → 动作）
DOMAIN_PARAMS = {
    "A": {
        "family": "puzzle",
        "syntax": "grid-sudoku",   # 9x9 格，行/列/宫约束
        "ops": [],                 # 无显式操作符
        "depth": None,             # 非树结构
        "value_range": [1, 9],     # 数字 1-9
        "seq_len": 81,             # 9x9 展平
        "output_style": "fill-grid",
        "note": "M0.1 复现 PTRM 数独基线",
    },
    "B": {
        "family": "seq-map",
        "syntax": "postfix",       # RPN 后缀表达式
        "ops": [10, 11],           # ADD=10, SUB=11 (vocab id)
        "depth": [2, 4],           # 树深度范围
        "value_range": [0, 9],     # 中间结果/答案范围
        "seq_len": 81,
        "output_style": "result",  # 位 0 = 结果
        "note": "M1a RPN 栈机算术 (m1_gen_domainB.py)",
    },
    "C": {
        "family": "seq-map",
        "syntax": "fixed-pos",     # 固定位置特征 (c,a) -> (x,y)
        "ops": ["mulmod"],         # f: x=(c*(a+1))%10, y=(a*(c+1))%10
        "depth": None,
        "value_range": [0, 9],
        "seq_len": 81,
        "output_style": "2-pos",   # 位 0,1 = x,y
        "note": "M0.4a 组合域鸭嘴兽载体 (m0_4_gen_domainC.py)",
    },
    "D": {
        "family": "game",
        "syntax": "iterated-game", # 离散状态博弈
        "ops": ["IPD", "PG", "SG"],# 囚徒困境/公共品/信号博弈 (D1/D2/D3)
        "depth": None,
        "value_range": None,       # 收益值域，非 token 域
        "seq_len": None,           # 观测窗口 10 轮
        "output_style": "action",  # 动作选择
        "note": "域D-实例化设计.md v0.1（D1 最小实例 = E6a 第 4 域）",
    },
    # 占位：域 E 三候选选定后填入；F/G 从 E 族延伸后填入
    "E": None,
    "F": None,
    "G": None,
}

# 辅助：人类可读的域间近邻预期（用于 sanity check 报告）
# 不做冻结物，仅文档性质
KNOWN_ORDER = ["A", "B", "C", "D"]  # v0.7 预注册的近→远排列（行为距离验证用）


def get_params(domain: str) -> dict | None:
    return DOMAIN_PARAMS.get(domain)


if __name__ == "__main__":
    for k, v in DOMAIN_PARAMS.items():
        if v is None:
            print(f"{k}: pending")
        else:
            print(f"{k}: family={v['family']} syntax={v['syntax']}")
