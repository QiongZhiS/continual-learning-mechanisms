"""domain_params.py — machine-readable structured parameters for all domains (M0.4 auxiliary)

Purpose (auxiliary reports + gradient-domain interpolation interface):
1. domain_distance.py reads parameters from this file to compute the structural distance matrix (auxiliary reporting)
2. gradient-domain interpolation pairs (domain B <-> prefix variants) need same-family parameter paths
3. E/F/G placeholders: to be filled after domain E is selected, then F/G extended from E's family

Freeze rules (pre-registration):
- the frozen similarity gradient = behavioral distance (zero-shot transfer matrix)
- structural distance is auxiliary reporting only, not part of the freeze
- A-D parameters = actual values of delivered generators (domain B: m1_gen_domainB.py; domain C: m0_4_gen_domainC.py;
  domain A sudoku = M0.1 reproduction config; domain D = domain-D instance spec)"""
from __future__ import annotations

# family values:
#   puzzle   — puzzle domain (grid/constraint fill)
#   seq-map  — sequence-mapping domain (program/sequence -> value)
#   game     — game domain (state -> action)
DOMAIN_PARAMS = {
    "A": {
        "family": "puzzle",
        "syntax": "grid-sudoku",   # 9x9 grid, row/column/box constraints
        "ops": [],                 # no explicit operators
        "depth": None,             # not a tree structure
        "value_range": [1, 9],     # digits 1-9
        "seq_len": 81,             # 9x9 flattened
        "output_style": "fill-grid",
        "note": "M0.1 reproduction of the PTRM sudoku baseline",
    },
    "B": {
        "family": "seq-map",
        "syntax": "postfix",       # RPN postfix expression
        "ops": [10, 11],           # ADD=10, SUB=11 (vocab id)
        "depth": [2, 4],           # tree-depth range
        "value_range": [0, 9],     # intermediate-result/answer range
        "seq_len": 81,
        "output_style": "result",  # slot 0 = result
        "note": "M1a RPN stack-machine arithmetic (m1_gen_domainB.py)",
    },
    "C": {
        "family": "seq-map",
        "syntax": "fixed-pos",     # fixed-position features (c,a) -> (x,y)
        "ops": ["mulmod"],         # f: x=(c*(a+1))%10, y=(a*(c+1))%10
        "depth": None,
        "value_range": [0, 9],
        "seq_len": 81,
        "output_style": "2-pos",   # slots 0,1 = x,y
        "note": "M0.4a composition-domain platypus carrier (m0_4_gen_domainC.py)",
    },
    "D": {
        "family": "game",
        "syntax": "iterated-game", # discrete-state game
        "ops": ["IPD", "PG", "SG"],  # prisoner's dilemma / public goods / signaling games (D1/D2/D3)
        "depth": None,
        "value_range": None,       # payoff range, not a token domain
        "seq_len": None,           # 10-round observation window
        "output_style": "action",  # action selection
        "note": "domain-D instance spec (D1 minimal instance = E6a 4th domain)",
    },
    # placeholders: fill after domain E is chosen; F/G extend from E's family
    "E": None,
    "F": None,
    "G": None,
}

# helper: human-readable expected inter-domain neighborhoods (for sanity-check reports)
# not frozen; documentation only
KNOWN_ORDER = ["A", "B", "C", "D"]  # pre-registered near->far ordering (for behavioral-distance verification)


def get_params(domain: str) -> dict | None:
    return DOMAIN_PARAMS.get(domain)


if __name__ == "__main__":
    for k, v in DOMAIN_PARAMS.items():
        if v is None:
            print(f"{k}: pending")
        else:
            print(f"{k}: family={v['family']} syntax={v['syntax']}")
