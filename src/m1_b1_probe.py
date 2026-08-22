"""B1 探针：错误信号可学习性验证（M3 go/no-go 第二道门 · 预注册 2026-08-18 · 阶段 B 执行）

判据来源：实验记录/B1探针-错误可学习性-设计预注册.md（预注册冻结，本脚本不改任何判据数值）
复用：m3_gate.py（G 门管线：夜间 LoRA / 记忆场回放 / eval / 断点续跑）+ m3_gate_diag1.py（错误质量度量）
      m2_e2.build_memory_field（域 A 记忆场，全量写，E2 结论：全量 > 惊喜；v0.30 全量回放口径一致）

臂（预注册冻结 5 臂 × 3 种子，载体 = 域 B，与 G 门直接可比；因子：信号源 × 学习目标）：
  a0 对照：标准微调（= G 门 ctl，无夜间）
  a1 现状：高置信错误（conf>0.9 且错）+ 错误重放 LoRA（= G 门方案 D 复现基线）
  a2 低置信：低置信错误（conf<0.7 且错）+ 错误重放 LoRA（信号源变化）
  a3 校准：高置信错误 + 置信度校准 BCE + 重放（学习目标变化）
     操作化 = 夜间损失加 λ·BCE(p_max, correct)——惩罚"确定地错"，直接针对诊断①过自信化
  a4 正则：高置信错误 + 重放 + LoRA 权重正则（配置变化）
     操作化 = 夜间损失加 λ·(||A||²+||B||²)——防过自信的权重正则

度量：域 B 达阈值（0.56 = 候选天花板 0.707×0.8）步数 T + 高置信错误率/总错误率（夜间前后 + 终态 n=2000）
判据（预注册）：
  可学习 = 任一臂 ① 高置信错误率 < A0 且 ② 总错误率不升 且 ③ T@0.56 中位数比值 ≤ 0.7 → M3 go
  不可学习 = 全部臂 ① 或 ② 失败 → M3 不按蒸馏轴投入，负结论归档；先验轴 c2/c3/c4 单独评估
  中间态 = 恰 1 臂部分改善（① 过 ③ 不过）→ 以该臂配置做 1 次扩展（不超预算）

用法（repo 目录）:
  python m1_b1_probe.py --arm a0 --seed 0                  # 单臂单种子（自动断点续跑）
  python m1_b1_probe.py --grid                             # 全网格 5 臂 × 3 种子 + judge.json
  python m1_b1_probe.py --judge-only                       # 仅从已有 summary 判读
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from m3_gate import (load_arch, load_a_ckpt_into, inject_lora, merge_and_restore,
                     eval_domain, A_CKPT, B_DATA, BS, LR, CONF_THR, LORA_LR,
                     REPLAY_R, ERR_N, RANK, ALPHA)
from models.losses import ACTLossHead
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from models.sparse_embedding import CastedSparseEmbedding


# --- sparse embedding 动态 batch 补丁（m2_e2 同款：训练分支取前 N 行，支持非整除分块）---
def _dyn_batch_forward(self, inputs):
    if not self.training:
        return self.weights[inputs].to(self.cast_to)
    bs = inputs.shape[0]
    with torch.no_grad():
        self.local_weights[:bs].copy_(self.weights[inputs])
        self.local_ids[:bs].copy_(inputs)
    return self.local_weights[:bs].to(self.cast_to)


CastedSparseEmbedding.forward = _dyn_batch_forward

STEPS = 10000
EVAL_EVERY = 500
NIGHT_EVERY = 2000
NIGHT_K = 200
LOW_CONF_THR = 0.70        # A2 低置信错误阈值（操作化：conf < 0.70 且预测错）
CAL_LAMBDA = 1.0           # A3 置信度 BCE 权重
CAL_SUBSET = 32            # A3 校准项子批上限（8GB 显存适配：32 行前向 ≈1.7GB，可复用缓存池；CE 主损失仍全错误批）
WD_LAMBDA = 0.05           # A4 LoRA 权重正则权重
TARGETS = [0.50, 0.56]     # 主判据阈值 0.56 = 候选天花板 0.707 × 0.8（预注册）；0.50 为对照记录
OUT = "outputs/2026-08-20/m1_b1_probe"

ARMS = {
    "a0": dict(signal=None, objective="none", desc="对照：标准微调（= G 门 ctl，无夜间）"),
    "a1": dict(signal="high", objective="replay", desc="现状：高置信错误 + 错误重放 LoRA（= G 门方案 D 复现基线）"),
    "a2": dict(signal="low", objective="replay", desc="信号源变化：低置信错误 + 错误重放 LoRA"),
    "a3": dict(signal="high", objective="calib", desc="学习目标变化：高置信错误 + 置信度校准 BCE + 重放"),
    "a4": dict(signal="high", objective="reg", desc="配置变化：高置信错误 + 重放 + LoRA 权重正则"),
}
SEEDS = (0, 1, 2)


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return "unknown"


def inner_of(model):
    return model.model if hasattr(model, "model") else model


def forward_conf_pred(inner, inputs, ids, halt=16, batch=128):
    """16 步前向 → 输出位 0 的 softmax 置信度与预测（对齐 collect_high_conf_errors / diag1）。"""
    N = inputs.shape[0]
    confs = torch.zeros(N, device="cuda")
    preds = torch.zeros(N, dtype=torch.long, device="cuda")
    with torch.inference_mode():
        for start in range(0, N, batch):
            b = inputs[start:start + batch]
            bs = b.shape[0]
            bk = {"inputs": b, "puzzle_identifiers": ids[start:start + bs]}
            with torch.device("cuda"):
                carry = inner.initial_carry(bk)
            for _ in range(halt):
                carry, outputs = inner(carry, bk)
            logits = outputs["logits"][:, 0]
            conf, pred = logits.softmax(-1).max(-1)
            confs[start:start + bs] = conf
            preds[start:start + bs] = pred
    return confs, preds


def collect_errors(model, signal, n=ERR_N, seed=0):
    """收集错误训练对。signal: high（conf>0.9 且错）/ low（conf<0.7 且错）。
    返回 (inputs [m,81], labels [m,81]) cuda；m=0 → None（夜间跳过）。"""
    t = os.path.join(B_DATA, "train")
    tr_in = np.load(os.path.join(t, "all__inputs.npy"), mmap_mode="r")
    tr_lb = np.load(os.path.join(t, "all__labels.npy"), mmap_mode="r")
    tr_ids = np.load(os.path.join(t, "all__puzzle_identifiers.npy"), mmap_mode="r")
    idx = np.sort(np.random.default_rng(seed).choice(len(tr_in), min(n, len(tr_in)), replace=False))
    inputs = torch.tensor(np.asarray(tr_in[idx], dtype=np.int32), device="cuda")
    labels = torch.tensor(np.asarray(tr_lb[idx], dtype=np.int32), device="cuda")
    ids = torch.tensor(np.asarray(tr_ids[idx], dtype=np.int32), device="cuda")
    confs, preds = forward_conf_pred(inner_of(model), inputs, ids)
    wrong = (preds != labels[:, 0])
    if signal == "high":
        sel = wrong & (confs > CONF_THR)
    elif signal == "low":
        sel = wrong & (confs < LOW_CONF_THR)
    else:
        raise ValueError(signal)
    wi = torch.nonzero(sel).squeeze(1)
    if wi.numel() == 0:
        return None
    return inputs[wi].contiguous(), labels[wi].contiguous()


def measure_error_quality(model, n=512, seed=0):
    """错误质量度量（diag1 核心统计）：err_rate / hc_err_rate / hc_share / conf 分布。"""
    t = os.path.join(B_DATA, "train")
    tr_in = np.load(os.path.join(t, "all__inputs.npy"), mmap_mode="r")
    tr_lb = np.load(os.path.join(t, "all__labels.npy"), mmap_mode="r")
    tr_ids = np.load(os.path.join(t, "all__puzzle_identifiers.npy"), mmap_mode="r")
    idx = np.sort(np.random.default_rng(seed).choice(len(tr_in), min(n, len(tr_in)), replace=False))
    inputs = torch.tensor(np.asarray(tr_in[idx], dtype=np.int32), device="cuda")
    labels = torch.tensor(np.asarray(tr_lb[idx], dtype=np.int32), device="cuda")
    ids = torch.tensor(np.asarray(tr_ids[idx], dtype=np.int32), device="cuda")
    confs, preds = forward_conf_pred(inner_of(model), inputs, ids)
    err = (preds != labels[:, 0]).cpu().numpy()
    conf_np = confs.cpu().numpy()
    hc = err & (conf_np > CONF_THR)
    return {
        "n": int(n), "n_err": int(err.sum()), "n_hc_err": int(hc.sum()),
        "err_rate": float(err.mean()), "hc_err_rate": float(hc.mean()),
        "hc_share_of_errors": float(hc.sum() / max(1, err.sum())),
        "err_conf_mean": float(conf_np[err].mean()) if err.any() else None,
        "hc_err_conf_mean": float(conf_np[hc].mean()) if hc.any() else None,
    }


def calibration_loss(inner, b_in, b_lb, halt=16, max_n=CAL_SUBSET):
    """A3：置信度校准 BCE——当前预测的 max 类概率 vs 是否正确（可导；惩罚高置信错 = 反过自信）。
    子批 ≤ CAL_SUBSET 行（显存适配；CE 主损失不受影响）。"""
    n = min(max_n, b_in.shape[0])
    b_in, b_lb = b_in[:n], b_lb[:n]
    bs = b_in.shape[0]
    bk = {"inputs": b_in, "puzzle_identifiers": torch.zeros(bs, dtype=torch.int32, device="cuda")}
    with torch.device("cuda"):
        carry = inner.initial_carry(bk)
    for _ in range(halt):
        carry, outputs = inner(carry, bk)
    logits0 = outputs["logits"][:, 0]
    p_max, pred = logits0.softmax(-1).max(-1)
    correct = (pred == b_lb[:, 0]).float()
    return F.binary_cross_entropy(p_max.float(), correct, reduction="sum")


def night_step(model, mf, seed, signal, objective, night_k=NIGHT_K):
    """夜间环节：错误对 ∪ 记忆场回放 r=0.25 的 LoRA 微调 → merge（G 门同款；objective 变体）。
    返回错误样本数 m（无样本 → None）。"""
    pair = collect_errors(model, signal)
    if pair is None:
        print(f"[night] {signal} 无错误样本，跳过", flush=True)
        return None
    torch.cuda.empty_cache()   # 释放收集前向的缓存块，防夜间训练峰值 OOM（8GB 显存边界）
    err_in, err_lb = pair
    m = err_in.shape[0]
    lora_params = inject_lora(model)
    lopt = torch.optim.AdamW(lora_params, lr=LORA_LR, weight_decay=0.0, fused=True)
    model.train()
    n_replay = int(BS * REPLAY_R) if mf is not None else 0
    inner = inner_of(model)
    with torch.device("cuda"):
        carry_e = model.initial_carry({
            "inputs": torch.zeros(min(m, BS), 81, dtype=torch.int32, device="cuda"),
            "labels": torch.zeros(min(m, BS), 81, dtype=torch.int32, device="cuda"),
            "puzzle_identifiers": torch.zeros(min(m, BS), dtype=torch.int32, device="cuda")})
        carry_r = model.initial_carry({
            "inputs": torch.zeros(n_replay, 81, dtype=torch.int32, device="cuda"),
            "labels": torch.zeros(n_replay, 81, dtype=torch.int32, device="cuda"),
            "puzzle_identifiers": torch.zeros(n_replay, dtype=torch.int32, device="cuda")}) if mf is not None else None
    rng = np.random.default_rng(seed)
    for k in range(night_k):
        sel = rng.choice(m, min(m, BS), replace=False)
        b_in, b_lb = err_in[sel], err_lb[sel]
        bs = b_in.shape[0]
        denom = bs + n_replay
        lopt.zero_grad()
        carry_e, loss_e, _, _, _ = model(carry=carry_e, batch={
            "inputs": b_in, "labels": b_lb,
            "puzzle_identifiers": torch.zeros(bs, dtype=torch.int32, device="cuda")},
            return_keys=[])
        total = loss_e / denom
        total.backward()
        # 分次 backward（m3_gate 同款）：每个损失算完立即释放计算图，防 8GB 显存峰值 OOM
        if objective == "calib":
            (CAL_LAMBDA * calibration_loss(inner, b_in, b_lb) / denom).backward()
        if objective == "reg":
            wd = sum((p ** 2).sum() for p in lora_params)
            (WD_LAMBDA * wd / denom).backward()
        if mf is not None:
            rb = mf.sample(n_replay)
            if rb is not None:
                ri, rl, rid = rb
                carry_r, loss_r, _, _, _ = model(carry=carry_r, batch={
                    "inputs": torch.tensor(ri, dtype=torch.int32, device="cuda"),
                    "labels": torch.tensor(rl, dtype=torch.int32, device="cuda"),
                    "puzzle_identifiers": torch.tensor(rid, dtype=torch.int32, device="cuda")},
                    return_keys=[])
                (loss_r / denom).backward()
        lopt.step()
    merge_and_restore(model)
    del lora_params
    torch.cuda.empty_cache()
    print(f"[night] s{seed} {signal} LoRA {night_k} 步, 错误样本 {m}, obj={objective}, 已 merge", flush=True)
    return m


def write_run_meta(out, arm, seed, cfg, steps, eval_every, night_every, night_k):
    meta = {
        "experiment": "B1 探针：错误信号可学习性验证（M3 go/no-go 第二道门）",
        "script": "m1_b1_probe.py",
        "git_commit": git_head(),
        "config": {
            "arm": arm, "desc": cfg["desc"], "signal": cfg["signal"],
            "objective": cfg["objective"], "seed": seed, "steps": steps,
            "eval_every": eval_every, "night_every": night_every, "night_k": night_k,
            "batch_size": BS, "lr": LR, "weight_decay": 0.1, "betas": [0.9, 0.95],
            "optimizer": "AdamW fused", "lora": {"rank": RANK, "alpha": ALPHA, "lr": LORA_LR},
            "replay": f"E2 全量写记忆场 r={REPLAY_R}（v0.30 全量回放口径一致）",
            "low_conf_thr": LOW_CONF_THR, "cal_lambda": CAL_LAMBDA, "wd_lambda": WD_LAMBDA,
            "signal_op": "high: conf>0.9 且错；low: conf<0.7 且错",
            "objective_op": {
                "replay": "CE(错误对) 直接监督",
                "calib": f"CE + λ·BCE(conf,correct) 防过自信（操作化；校准项子批 ≤{CAL_SUBSET} 行，8GB 显存适配）",
                "reg": "CE + λ·(||A||²+||B||²) 权重正则（操作化）",
                "none": "无夜间",
            },
            "targets": TARGETS,
            "data": "../data/domain-b-rpn（2026-08-12 冻结）",
            "env": "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128; DISABLE_COMPILE=1",
        },
        "judge": "① 高置信错误率 < A0 且 ② 总错误率不升 且 ③ T@0.56 中位数比值 ≤ 0.7 → M3 go（预注册：B1探针-错误可学习性-设计预注册.md）",
        "run_meta_written_by": "m1_b1_probe.py（§3.5 自动落盘）",
    }
    json.dump(meta, open(os.path.join(out, "run_meta.json"), "w"), indent=2, ensure_ascii=False)


def run_arm(arm, seed, steps=STEPS, eval_every=EVAL_EVERY, night_every=NIGHT_EVERY, night_k=NIGHT_K):
    cfg = ARMS[arm]
    out = os.path.join(OUT, arm, f"s{seed}")
    os.makedirs(out, exist_ok=True)
    meta = json.load(open(f"{B_DATA}/dataset.json"))
    arch_cfg = load_arch(meta)

    # ---- 断点续跑：已有 result.json → 从最后 step 的 checkpoint 续训（G 门同款） ----
    # 崩溃恢复健壮性：若最后档位 ckpt 缺失（保存中断）→ 回退到有 ckpt 的档位并截断记录
    resume_from = 0
    results = []
    res_path = os.path.join(out, "result.json")
    if os.path.exists(res_path):
        results = json.load(open(res_path))
        while results and not os.path.exists(os.path.join(out, f"step_{results[-1]['step']}")):
            print(f"[{arm}] s{seed} 档位 step {results[-1]['step']} 无 ckpt，回退", flush=True)
            results.pop()
        if results:
            resume_from = results[-1]["step"]
            json.dump(results, open(res_path, "w"), indent=2)   # 同步截断
            print(f"[{arm}] s{seed} 续跑: 已有 {len(results)} 点, 从 step {resume_from} 继续", flush=True)

    with torch.device("cuda"):
        m = TinyRecursiveReasoningModel_ACTV1(arch_cfg)
        if resume_from > 0:
            ckpt = os.path.join(out, f"step_{resume_from}")
            state = {k.removeprefix("model."): v for k, v in torch.load(
                ckpt, map_location="cuda").items()}
            m.load_state_dict(state, assign=True)
            del state
            torch.cuda.empty_cache()
            print(f"[{arm}] s{seed} 已加载 ckpt step {resume_from}", flush=True)
        else:
            load_a_ckpt_into(m, A_CKPT)
        m.eval()
    model = ACTLossHead(m, "stablemax_cross_entropy")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1,
                            betas=(0.9, 0.95), fused=True)

    tr_in = np.load(f"{B_DATA}/train/all__inputs.npy", mmap_mode="r")
    tr_lb = np.load(f"{B_DATA}/train/all__labels.npy", mmap_mode="r")
    tr_ids = np.load(f"{B_DATA}/train/all__puzzle_identifiers.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)

    mf = None
    if arm != "a0":
        from m2_e2 import build_memory_field
        mf = build_memory_field(m, 1, 2000, seed=seed)   # 全量写（E2 结论：全量 > 惊喜）
        print(f"[{arm}] 记忆场: {mf.stats()}", flush=True)

    n_replay = int(BS * REPLAY_R) if mf is not None else 0
    batch_n = BS + n_replay
    with torch.device("cuda"):
        carry_b = model.initial_carry({
            "inputs": torch.zeros(BS, 81, dtype=torch.int32, device="cuda"),
            "labels": torch.zeros(BS, 81, dtype=torch.int32, device="cuda"),
            "puzzle_identifiers": torch.zeros(BS, dtype=torch.int32, device="cuda")})
        carry_r = model.initial_carry({
            "inputs": torch.zeros(n_replay, 81, dtype=torch.int32, device="cuda"),
            "labels": torch.zeros(n_replay, 81, dtype=torch.int32, device="cuda"),
            "puzzle_identifiers": torch.zeros(n_replay, dtype=torch.int32, device="cuda")}) if mf is not None else None

    t0 = time.time()
    for step in range(0, steps + 1, eval_every):
        if step <= resume_from:
            continue
        if step > 0:
            for it in range(eval_every):
                if it % 100 == 0:
                    print(f"[{arm}] s{seed} step {step} iter {it} t={time.time()-t0:.0f}s", flush=True)
                    torch.cuda.empty_cache()
                idxb = rng.choice(len(tr_in), BS, replace=False)
                batch_b = {
                    "inputs": torch.tensor(np.asarray(tr_in[idxb], dtype=np.int32), device="cuda"),
                    "labels": torch.tensor(np.asarray(tr_lb[idxb], dtype=np.int32), device="cuda"),
                    "puzzle_identifiers": torch.tensor(np.asarray(tr_ids[idxb], dtype=np.int32), device="cuda"),
                }
                opt.zero_grad()
                carry_b, loss_b, _, _, _ = model(carry=carry_b, batch=batch_b, return_keys=[])
                (loss_b / batch_n).backward()
                loss = loss_b
                if mf is not None:
                    rb = mf.sample(n_replay)
                    if rb is not None:
                        ri, rl, rid = rb
                        carry_r, loss_r, _, _, _ = model(carry=carry_r, batch={
                            "inputs": torch.tensor(ri, dtype=torch.int32, device="cuda"),
                            "labels": torch.tensor(rl, dtype=torch.int32, device="cuda"),
                            "puzzle_identifiers": torch.tensor(rid, dtype=torch.int32, device="cuda")},
                            return_keys=[])
                        (loss_r / batch_n).backward()
                        loss = loss_b + loss_r
                opt.step()
        ckpt_path = os.path.join(out, f"step_{step}")
        torch.save(model.state_dict(), ckpt_path)
        torch.cuda.empty_cache()
        m2 = TinyRecursiveReasoningModel_ACTV1(arch_cfg)
        state2 = {k.removeprefix("model."): v for k, v in torch.load(
            ckpt_path, map_location="cuda").items()}
        m2.load_state_dict(state2, assign=True)
        m2.eval()
        with torch.device("cuda"):
            m2.to("cuda")
        m.train()
        acc = eval_domain(m2, B_DATA, 500, seed=0)
        rec = {"step": step, "acc": acc, "loss": float(loss.item() / batch_n),
               "t_sec": round(time.time() - t0)}
        # 夜间边界（a1-a4）：夜间前后错误质量度量
        if mf is not None and step % night_every == 0 and step > 0:
            rec["night_before"] = measure_error_quality(model, 512, seed=0)
            torch.cuda.empty_cache()   # 释放度量前向缓存块，防夜间峰值 OOM
            nm = night_step(model, mf, seed, cfg["signal"], cfg["objective"], night_k)
            rec["night_err_n"] = nm
            rec["night_after"] = measure_error_quality(model, 512, seed=0) if nm is not None else None
            torch.cuda.empty_cache()
            # 崩溃恢复一致性：夜间 merge 后重存 ckpt，resume 从此档位加载含夜间状态
            torch.save(model.state_dict(), ckpt_path)
        results.append(rec)
        print(f"[{arm}] s{seed} step {step}: acc={acc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(results, open(res_path, "w"), indent=2)
        del m2, state2
        torch.cuda.empty_cache()

    final_q = measure_error_quality(model, 2000, seed=0)
    summary = {
        "arm": arm, "seed": seed, "steps": steps, "config": cfg,
        "curve": results, "error_quality_final": final_q, "git_commit": git_head(),
    }
    for tg in TARGETS:
        t_steps = None
        for r_ in results:
            if r_["acc"] >= tg:
                t_steps = r_["step"]
                break
        summary[f"T@{tg}"] = t_steps
        print(f"[{arm}] s{seed} T@{tg} = {t_steps}", flush=True)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2, ensure_ascii=False)
    write_run_meta(out, arm, seed, cfg, steps, eval_every, night_every, night_k)
    print(f"[{arm}] s{seed} done", flush=True)


def judge():
    """三因子归因表 + T 比值分布 → go/no-go 裁决（预注册判据）。"""
    def load(arm, seed):
        p = os.path.join(OUT, arm, f"s{seed}", "summary.json")
        return json.load(open(p)) if os.path.exists(p) else None

    def med(vals):
        vv = [v for v in vals if v is not None]
        return float(np.median(vv)) if vv else None

    a0 = [s for s in (load("a0", sd) for sd in SEEDS) if s]
    ref = {
        "hc_err_rate": med([s["error_quality_final"]["hc_err_rate"] for s in a0]),
        "err_rate": med([s["error_quality_final"]["err_rate"] for s in a0]),
        "T@0.56": med([s["T@0.56"] for s in a0]),
        "T@0.5": med([s["T@0.5"] for s in a0]),
        "n_seeds": len(a0),
    }

    arms_out = {}
    improved, full, partial = [], [], []
    for arm in ("a1", "a2", "a3", "a4"):
        sums = [s for s in (load(arm, sd) for sd in SEEDS) if s]
        hc = med([s["error_quality_final"]["hc_err_rate"] for s in sums])
        err = med([s["error_quality_final"]["err_rate"] for s in sums])
        t56 = med([s["T@0.56"] for s in sums])
        t50 = med([s["T@0.5"] for s in sums])
        c1 = hc is not None and ref["hc_err_rate"] is not None and hc < ref["hc_err_rate"]
        c2 = err is not None and ref["err_rate"] is not None and err <= ref["err_rate"]
        c3 = (t56 is not None and ref["T@0.56"] is not None
              and t56 / ref["T@0.56"] <= 0.7)
        arms_out[arm] = {
            "hc_err_rate": hc, "err_rate": err, "T@0.56": t56, "T@0.5": t50,
            "T_ratio_vs_a0": (round(t56 / ref["T@0.56"], 3)
                              if t56 is not None and ref["T@0.56"] else None),
            "c1_hc_down": c1, "c2_err_not_up": c2, "c3_T_ratio_le_07": c3,
        }
        if c1 and c2:
            improved.append(arm)
            if c3:
                full.append(arm)
            else:
                partial.append(arm)

    if full:
        verdict = "M3_GO"
        expl = (f"可学习：{full} 满足 ①②③ → M3 go，用胜出臂配置恢复 E3 全因子"
                f"（D 位更新为探针胜出配置，预注册修订）")
    elif not improved:
        verdict = "M3_NO_GO"
        expl = ("不可学习：全部臂 ① 或 ② 失败（高置信错误率不降/总错误率升）→ "
                "M3 不按蒸馏轴投入，负结论归档（v0.23 措辞：只对实测方案成立）；"
                "先验轴 c2/c3/c4 单独评估（与蒸馏正交）")
    elif len(partial) == 1:
        verdict = "MIDDLE_1_ARM"
        expl = f"中间态：{partial[0]} 部分改善（①过 ③不过）→ 以该臂配置做 1 次扩展（不超预算）"
    else:
        verdict = "MIDDLE_MULTI"
        expl = f"多臂部分改善（{improved}）——预注册未定义情形，报告 + 选改善最大臂扩展"

    j = {"a0_reference": ref, "arms": arms_out, "improved": improved,
         "full": full, "partial": partial, "verdict": verdict,
         "explanation": expl, "git_commit": git_head()}
    json.dump(j, open(os.path.join(OUT, "judge.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps(j, indent=2, ensure_ascii=False))
    print("saved ->", os.path.join(OUT, "judge.json"))


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    ap.add_argument("--night-every", type=int, default=NIGHT_EVERY)
    ap.add_argument("--night-k", type=int, default=NIGHT_K)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--grid", action="store_true", help="全网格 5 臂 × 3 种子 + judge.json")
    ap.add_argument("--judge-only", action="store_true", help="仅从已有 summary 判读")
    args = ap.parse_args()
    OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    if args.judge_only:
        judge()
        return
    if args.grid:
        for arm in ARMS:
            for seed in SEEDS:
                run_arm(arm, seed, args.steps, args.eval_every, args.night_every, args.night_k)
        judge()
        return
    if args.arm is None:
        ap.error("--arm 或 --grid 必选")
    run_arm(args.arm, args.seed, args.steps, args.eval_every, args.night_every, args.night_k)


if __name__ == "__main__":
    main()
