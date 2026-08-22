"""M3 G 门探针（v0.28 预注册 · E3 go/no-go · 方案 D vs 标准微调）

两臂（同底座 = 域 A ckpt precise-muskox，同域 B 训练协议）：
  ① 对照：标准微调（= E6a ② 同构：m1_train_domainB 协议，lr 1e-4 · batch 128 · fused AdamW）
  ② 方案 D：标准微调 + 周期夜间环节——白天高置信错误记录（softmax 置信度>0.9 且结果位错）
     → 夜间 LoRA 微调（训练对 = 错误记录 ∪ 记忆场回放 r=0.25）→ merge 回主模型（= 次日加载）

测量：域 B 达目标线 exact 所需白天训练步数 T（目标线参数化：0.40/0.45/0.50）
判据（预注册）：T 中位数比值（T_D / T_对照）≤ 0.7 → go；> 0.7 → no-go 信号
种子：3（报告完整分布 + 中位数比值，探索性）

夜间环节（探针版，对齐 v0.28 原文）：
  - 每 NIGHT_EVERY 白天步后触发一次（step % NIGHT_EVERY == 0，在 eval 之后）
  - collect：域 B train 子集推理 → softmax 置信度 >0.9 且预测错 → (inputs, labels) 训练对
  - LoRA（rank 16，lr 3e-4，M0.3b 配置）训练 NIGHT_K 步：错误对 ∪ 记忆场回放（r=0.25）
  - merge：W += (alpha/r)·A·B，还原原 CastedLinear，恢复 requires_grad
  - 夜间不改变 T 计数（T = 白天微调步数）；夜间训练在 eval 后执行，不污染已评估曲线

显存：错误 batch（≤BS）与回放（32）拆分 forward 梯度累积（峰值 = max batch，m2_e2 同款）

用法（repo 目录）:
  python m3_gate.py --arm ctl --seed 0          # 对照臂单种子
  python m3_gate.py --arm d --seed 0            # 方案 D 臂单种子
  python m3_gate.py --grid                      # 全网格：两臂 × 3 种子（含判据判定）
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import yaml

from eval_ptrm import load_test_data, ptrm_infer
from models.layers import CastedLinear
from models.losses import ACTLossHead
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

A_DATA = "../data/sudoku-extreme-1k-aug-100"
B_DATA = "../data/domain-b-rpn"
A_CKPT = "checkpoints/Sudoku-extreme-1k-aug-100-ACT-torch/TinyRecursiveReasoningModel_ACTV1 precise-muskox/step_23437"
BS = 128
LR = 1e-4
STEPS = 10000          # 域 B 训练预算上限（对齐探针原文 4000-10000 步）
EVAL_EVERY = 500
NIGHT_EVERY = 2000     # 每 2000 白天步插入一次夜间环节
NIGHT_K = 200          # 夜间 LoRA 微调步数
CONF_THR = 0.90        # 高置信错误阈值（v0.28 原文：置信度 >90% 但结果错误）
ERR_N = 512            # 每次夜间从 train 前 ERR_N 样本中收集错误
RANK = 16
ALPHA = 16
LORA_LR = 3e-4
REPLAY_R = 0.25
TARGETS = [0.40, 0.45, 0.50]
OUT = "outputs/2026-08-15/m3_gate"


class LoraCasted(torch.nn.Module):
    """零初始化 B → 初始行为 = 原 CastedLinear（m0_3_lora 同款）"""
    def __init__(self, linear, r, alpha):
        super().__init__()
        self.linear = linear
        self.scale = alpha / r
        dev = linear.weight.device
        self.lora_A = torch.nn.Parameter(torch.randn(linear.weight.shape[1], r, device=dev) * 0.01)
        self.lora_B = torch.nn.Parameter(torch.zeros(r, linear.weight.shape[0], device=dev))
        for p in linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        dtype = x.dtype
        return self.linear(x) + (x @ self.lora_A.to(dtype) @ self.lora_B.to(dtype)) * self.scale


def load_arch(meta, batch_size=BS):
    arch_cfg = yaml.safe_load(open("config/arch/trm.yaml", encoding="utf-8"))
    arch_cfg["puzzle_emb_ndim"] = arch_cfg["hidden_size"]
    arch_cfg.pop("name", None)
    arch_cfg.pop("loss", None)
    arch_cfg.update(dict(batch_size=batch_size, vocab_size=meta["vocab_size"],
                         seq_len=meta["seq_len"],
                         num_puzzle_identifiers=meta["num_puzzle_identifiers"], causal=False,
                         mlp_t=True, pos_encodings="none"))
    return arch_cfg


def load_a_ckpt_into(m, ckpt):
    """域 A ckpt（vocab=11）载入 vocab=13 模型：embedding/lm_head 行移位迁移（m2_e2 同款）"""
    st = torch.load(ckpt, map_location="cuda")
    st = {k.removeprefix("model."): v for k, v in st.items()}
    for key in ("inner.embed_tokens.embedding_weight", "inner.lm_head.weight"):
        old = st[key]
        new = m.state_dict()[key].clone()
        new[:10] = old[1:11]
        st[key] = new
    m.load_state_dict(st, assign=True)


def inject_lora(model):
    """替换全部 CastedLinear → LoraCasted；冻结底座，只留 LoRA 参数可训"""
    lora_params = []
    for name, mod in list(model.named_modules()):
        if isinstance(mod, CastedLinear):
            parent = model
            for p in name.split(".")[:-1]:
                parent = getattr(parent, p)
            lc = LoraCasted(mod, RANK, ALPHA)
            setattr(parent, name.split(".")[-1], lc)
            lora_params += [lc.lora_A, lc.lora_B]
    for p in model.parameters():
        p.requires_grad = False
    for p in lora_params:
        p.requires_grad = True
    return lora_params


def merge_and_restore(model):
    """LoRA merge 回主权重并还原原 CastedLinear（含 requires_grad 恢复）
    CastedLinear.weight 布局 = (out, in)；LoraCasted 前向 = x @ A @ B (A:(in,r) B:(r,out))
    → 增量 = (A@B).t() 形状 (out, in)"""
    with torch.no_grad():
        for name, mod in list(model.named_modules()):
            if isinstance(mod, LoraCasted):
                delta = (mod.lora_A.float() @ mod.lora_B.float()).t()
                mod.linear.weight.add_(delta, alpha=mod.scale)
                parent = model
                for p in name.split(".")[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, name.split(".")[-1], mod.linear)
    for p in model.parameters():
        p.requires_grad = True


def eval_domain(model, data_dir, n, seed=0, K=1, D=16):
    """标准 eval：exact = 输出位 0（域 B 判据）"""
    inputs, labels, ids, _ = load_test_data(data_dir, n, seed)
    pred, _ = ptrm_infer(model, inputs, ids, K, D, 0.0)
    return float((pred[:, 0] == labels[:, 0]).float().mean().item())


def collect_high_conf_errors(model, n=ERR_N):
    """白天推理：域 B train 前 n 样本 → softmax 置信度 >0.9 且结果位错 → 错误训练对。
    返回 (inputs [m,81], labels [m,81]) cuda；m=0 时返回 None。"""
    t = os.path.join(B_DATA, "train")
    tr_in = np.load(os.path.join(t, "all__inputs.npy"), mmap_mode="r")
    tr_lb = np.load(os.path.join(t, "all__labels.npy"), mmap_mode="r")
    tr_ids = np.load(os.path.join(t, "all__puzzle_identifiers.npy"), mmap_mode="r")
    idx = np.sort(np.random.default_rng(0).choice(len(tr_in), min(n, len(tr_in)), replace=False))
    inputs = torch.tensor(np.asarray(tr_in[idx], dtype=np.int32), device="cuda")
    labels = torch.tensor(np.asarray(tr_lb[idx], dtype=np.int32), device="cuda")
    ids = torch.tensor(np.asarray(tr_ids[idx], dtype=np.int32), device="cuda")
    # 用底层 TRM 直接前向（wrapper 的 return_keys 过滤会丢弃 logits；halt 固定 16 步）
    inner = model.model if hasattr(model, "model") else model
    inner.config.halt_max_steps = 16
    inner.eval()
    with torch.inference_mode():
        N = inputs.shape[0]
        confs = torch.zeros(N, device="cuda")
        preds = torch.zeros(N, dtype=torch.long, device="cuda")
        for start in range(0, N, 128):
            b = inputs[start:start + 128]
            bs = b.shape[0]
            batch = {"inputs": b, "puzzle_identifiers": ids[start:start + bs]}
            with torch.device("cuda"):
                carry = inner.initial_carry(batch)
            for _ in range(16):
                carry, outputs = inner(carry, batch)
            logits = outputs["logits"][:, 0]           # [bs, vocab]
            conf, pred = logits.softmax(-1).max(-1)
            confs[start:start + bs] = conf
            preds[start:start + bs] = pred
    wrong = (preds != labels[:, 0]) & (confs > CONF_THR)
    wi = torch.nonzero(wrong).squeeze(1)
    inner.train()   # 恢复训练态（与 wrapper 同步：ACTLossHead.train 递归设置子模块）
    if wi.numel() == 0:
        return None
    return inputs[wi].contiguous(), labels[wi].contiguous()


def night_step(model, mf, seed):
    """方案 D 夜间环节：高置信错误 → LoRA 微调（错误 ∪ 记忆场回放 r=0.25）→ merge。
    拆分 forward 梯度累积（错误 batch 与回放 32 分开），峰值显存 = max batch。"""
    pair = collect_high_conf_errors(model)
    if pair is None:
        print("[night] 无高置信错误，跳过", flush=True)
        return
    err_in, err_lb = pair
    m = err_in.shape[0]
    lora_params = inject_lora(model)
    lopt = torch.optim.AdamW(lora_params, lr=LORA_LR, weight_decay=0.0, fused=True)
    model.train()
    n_replay = int(BS * REPLAY_R) if mf is not None else 0
    with torch.device("cuda"):
        carry_e = model.initial_carry({
            "inputs": torch.zeros(min(m, BS), 81, dtype=torch.int32, device="cuda"),
            "labels": torch.zeros(min(m, BS), 81, dtype=torch.int32, device="cuda"),
            "puzzle_identifiers": torch.zeros(min(m, BS), dtype=torch.int32, device="cuda")})
        carry_r = model.initial_carry({
            "inputs": torch.zeros(n_replay, 81, dtype=torch.int32, device="cuda"),
            "labels": torch.zeros(n_replay, 81, dtype=torch.int32, device="cuda"),
            "puzzle_identifiers": torch.zeros(n_replay, dtype=torch.int32, device="cuda")})
    rng = np.random.default_rng(seed)
    for k in range(NIGHT_K):
        sel = rng.choice(m, min(m, BS), replace=False)
        b_in = err_in[sel]
        b_lb = err_lb[sel]
        bs = b_in.shape[0]
        lopt.zero_grad()
        carry_e, loss_e, _, _, _ = model(carry=carry_e, batch={
            "inputs": b_in, "labels": b_lb,
            "puzzle_identifiers": torch.zeros(bs, dtype=torch.int32, device="cuda")},
            return_keys=[])
        (loss_e / (bs + n_replay)).backward()
        if mf is not None:
            rb = mf.sample(n_replay)
            if rb is not None:
                ri, rl, rid = rb
                carry_r, loss_r, _, _, _ = model(carry=carry_r, batch={
                    "inputs": torch.tensor(ri, dtype=torch.int32, device="cuda"),
                    "labels": torch.tensor(rl, dtype=torch.int32, device="cuda"),
                    "puzzle_identifiers": torch.tensor(rid, dtype=torch.int32, device="cuda")},
                    return_keys=[])
                (loss_r / (bs + n_replay)).backward()
        lopt.step()
    merge_and_restore(model)
    del lora_params
    torch.cuda.empty_cache()
    print(f"[night] s{seed} LoRA {NIGHT_K} 步, 错误样本 {m}, 已 merge", flush=True)


def run_arm(arm, seed, steps=STEPS):
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(f"{B_DATA}/dataset.json"))
    arch_cfg = load_arch(meta)

    # ---- 断点续跑：已有 result.json → 从最后 step 的 checkpoint 续训 ----
    resume_from = 0
    results = []
    res_path = f"{OUT}/{arm}_s{seed}_result.json"
    if os.path.exists(res_path):
        results = json.load(open(res_path))
        if results:
            resume_from = results[-1]["step"]
            print(f"[{arm}] s{seed} 续跑: 已有 {len(results)} 点, 从 step {resume_from} 继续", flush=True)

    with torch.device("cuda"):
        m = TinyRecursiveReasoningModel_ACTV1(arch_cfg)
        if resume_from > 0:
            ckpt = f"{OUT}/{arm}_s{seed}_step_{resume_from}"
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
    if arm == "d":
        from m2_e2 import build_memory_field
        mf = build_memory_field(m, 1, 2000, seed=seed)   # 全量写（E2 结论：全量 > 惊喜）
        print(f"[{arm}] 记忆场: {mf.stats()}", flush=True)

    n_replay = int(BS * REPLAY_R) if mf is not None else 0
    batch_n = BS + n_replay
    # 拆分 forward：域B carry(BS) + 回放 carry(n_replay)（m2_e2 同款显存适配）
    with torch.device("cuda"):
        carry_b = model.initial_carry({
            "inputs": torch.zeros(BS, 81, dtype=torch.int32, device="cuda"),
            "labels": torch.zeros(BS, 81, dtype=torch.int32, device="cuda"),
            "puzzle_identifiers": torch.zeros(BS, dtype=torch.int32, device="cuda")})
        carry_r = model.initial_carry({
            "inputs": torch.zeros(n_replay, 81, dtype=torch.int32, device="cuda"),
            "labels": torch.zeros(n_replay, 81, dtype=torch.int32, device="cuda"),
            "puzzle_identifiers": torch.zeros(n_replay, dtype=torch.int32, device="cuda")})

    t0 = time.time()
    for step in range(0, steps + 1, EVAL_EVERY):
        if step <= resume_from:
            continue
        if step > 0:
            for it in range(EVAL_EVERY):
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
            ckpt_path = f"{OUT}/{arm}_s{seed}_step_{step}"
            torch.save(model.state_dict(), ckpt_path)
            torch.cuda.empty_cache()   # eval 前回收缓存池，降低 m2 重建峰值
            m2 = TinyRecursiveReasoningModel_ACTV1(arch_cfg)
            state2 = {k.removeprefix("model."): v for k, v in torch.load(
                ckpt_path, map_location="cuda").items()}
            m2.load_state_dict(state2, assign=True)
            m2.eval()
            with torch.device("cuda"):
                m2.to("cuda")
            m.train()
            acc = eval_domain(m2, B_DATA, 500, seed=0)
            results.append({"step": step, "acc": acc, "loss": float(loss.item() / batch_n),
                            "t_sec": round(time.time() - t0)})
            print(f"[{arm}] s{seed} step {step}: acc={acc:.4f} ({time.time()-t0:.0f}s)", flush=True)
            json.dump(results, open(f"{OUT}/{arm}_s{seed}_result.json", "w"), indent=2)
            del m2, state2
            torch.cuda.empty_cache()
            # 夜间环节（方案 D）：每 NIGHT_EVERY 白天步后（eval 之后，不污染曲线）
            if mf is not None and step % NIGHT_EVERY == 0:
                night_step(model, mf, seed)

    summary = {"arm": arm, "seed": seed, "steps": steps, "curve": results}
    for tg in TARGETS:
        t_steps = None
        for r_ in results:
            if r_["acc"] >= tg:
                t_steps = r_["step"]
                break
        summary[f"T@{tg}"] = t_steps
        print(f"[{arm}] s{seed} T@{tg} = {t_steps}", flush=True)
    json.dump(summary, open(f"{OUT}/{arm}_s{seed}_summary.json", "w"), indent=2)
    print(f"[{arm}] s{seed} done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["ctl", "d"], default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", action="store_true", help="全网格：两臂 × 3 种子")
    ap.add_argument("--steps", type=int, default=STEPS)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.grid:
        for arm in ("ctl", "d"):
            for seed in (0, 1, 2):
                run_arm(arm, seed, args.steps)
        judge = {}
        for tg in TARGETS:
            t_ctl = [json.load(open(f"{OUT}/ctl_s{s}_summary.json"))[f"T@{tg}"] for s in range(3)]
            t_d = [json.load(open(f"{OUT}/d_s{s}_summary.json"))[f"T@{tg}"] for s in range(3)]
            t_ctl_v = [x for x in t_ctl if x is not None]
            t_d_v = [x for x in t_d if x is not None]
            ratio = None
            if t_ctl_v and t_d_v:
                ratio = float(np.median(t_d_v) / np.median(t_ctl_v))
            judge[f"T@{tg}"] = {"T_ctl": t_ctl, "T_d": t_d, "median_ratio": ratio,
                                "go": ratio is not None and ratio <= 0.7}
            print(f"T@{tg}: T_ctl={t_ctl} T_d={t_d} ratio={ratio} go={judge[f'T@{tg}']['go']}", flush=True)
        json.dump(judge, open(f"{OUT}/judge.json", "w"), indent=2)
        print("saved -> outputs/2026-08-15/m3_gate/judge.json", flush=True)
        return
    run_arm(args.arm, args.seed, args.steps)


if __name__ == "__main__":
    main()
