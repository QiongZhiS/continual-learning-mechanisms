"""G 门诊断 ③（v0.31 五向 · 回放采样模式）：全量回放（uniform）变体

背景（归因边界，G门-判读记录.md §五-5 / §六-3）：
  G 门 d 臂按 v0.23 S4 预注册用 **priority 采样**执行回放，而 priority 已被 E2 证伪
  （惊喜写 0.050 < 全量回放 0.160）。因此 NO-GO 无法区分"蒸馏不加速"与
  "沿用已证伪采样"——本诊断把回放采样改为 **uniform（全量回放）** 重测 T。

实现：monkey-patch MemoryField.sample → 强制 mode="uniform"（夜间与白天回放均生效），
其余配置与 G 门 d 臂完全一致（同底座、同域 B 训练协议、同夜间 LoRA 配置）。

用法（repo 目录，建议后台 + 日志重定向）：
  $env:PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128'
  python m3_gate_diag3_fullreplay.py --seed 0
"""
import argparse

import m2_memory

_orig_sample = m2_memory.MemoryField.sample


def _uniform_sample(self, n, mode="priority"):
    return _orig_sample(self, n, mode="uniform")


m2_memory.MemoryField.sample = _uniform_sample

import m3_gate  # noqa: E402  (patch 先于 m3_gate 内部 import m2_e2/m2_memory)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=m3_gate.STEPS)
    args = ap.parse_args()
    m3_gate.OUT = "outputs/2026-08-15/m3_gate_diag3_fullreplay"
    print(f"[diag3] full-replay (uniform) d-arm seed={args.seed} steps={args.steps} -> {m3_gate.OUT}", flush=True)
    m3_gate.run_arm("d", args.seed, args.steps)


if __name__ == "__main__":
    main()
