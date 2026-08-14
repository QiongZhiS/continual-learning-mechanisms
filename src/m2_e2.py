"""E2 记忆写入确认实验：惊喜写入 vs 全量写入的容量效率 + 记忆价值（域 A→B 跨任务保持）

协议（AGI-实验方案.md §4 E2 + v0.23 S4）：
- 三臂：臂 0 无记忆场（纯域 B 微调 = 灾难性遗忘对照）｜臂 1 全量写+回放｜臂 2 惊喜写+回放
- 起点：域 A ckpt（precise-muskox/step_23437，M0.3 天花板标定基线，exact 0.330）
- 域 A 记忆条目：域 A ckpt 在域 A train 上前向，逐样本 CE loss = 惊喜分数 → 两臂各建记忆场
  （容量 C 相同；开臂只写高 surprise 子集，关臂全量 FIFO——内容不同容量相同）
- 域 B 微调：m1_train_domainB 协议（lr 1e-4 · batch 128 · fused AdamW · fixed lr）
  训练对 = 域 B batch ∪ 记忆场回放（r=0.25，S4 接口，priority 采样）
- 测量：域 A test 保持率（200 谜题 K=1 D=16，与 M0.3/E2b 同配置）+ 域 B test exact
- 判据：臂2 域 A 保持 ≥ 臂1（容量效率）且 > 臂0（记忆价值）；域 B 进度不显著退化
- E2b：--eval-only 对已存 ckpt 重测域 A 保持（t=0 = 训练结束值；1/3/7/30 天后重跑得衰减点）

用法（repo 目录）:
    python m2_e2.py --steps 4000 --cap 2000                    # 三臂主实验
    python m2_e2.py --eval-only <ckpt_dir> --arm <n>           # E2b 间隔重测
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import yaml

from eval_ptrm import load_test_data, load_model, ptrm_infer
from m2_memory import MemoryField
from models.losses import ACTLossHead, stablemax_cross_entropy
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

A_DATA = "../data/sudoku-extreme-1k-aug-100"
B_DATA = "../data/domain-b-rpn"
A_CKPT = "checkpoints/Sudoku-extreme-1k-aug-100-ACT-torch/TinyRecursiveReasoningModel_ACTV1 precise-muskox/step_23437"
BS = 128
REPLAY_R = 0.25          # S4 预注册：回放比例 r=0.25
EVAL_EVERY = 500
A_EVAL_N = 200           # 域 A 保持率测量规模（与 M0.3/E2b 同配置）


def load_arch(meta):
    arch_cfg = yaml.safe_load(open("config/arch/trm.yaml", encoding="utf-8"))
    arch_cfg["puzzle_emb_ndim"] = arch_cfg["hidden_size"]
    arch_cfg.pop("name", None)
    arch_cfg.pop("loss", None)
    arch_cfg.update(dict(batch_size=BS, vocab_size=meta["vocab_size"], seq_len=meta["seq_len"],
                         num_puzzle_identifiers=meta["num_puzzle_identifiers"], causal=False,
                         mlp_t=True, pos_encodings="none"))
    return arch_cfg


def load_split(data_dir, n, split, seed=0):
    t = os.path.join(data_dir, split)
    inputs = np.load(os.path.join(t, "all__inputs.npy"), mmap_mode="r")
    labels = np.load(os.path.join(t, "all__labels.npy"), mmap_mode="r")
    ids = np.load(os.path.join(t, "all__puzzle_identifiers.npy"), mmap_mode="r")
    if split == "train":
        idx = np.sort(np.random.default_rng(seed).choice(len(inputs), n, replace=False))
        inputs, labels, ids = inputs[idx], labels[idx], ids[idx]
    return (torch.tensor(np.asarray(inputs[:n], dtype=np.int32), device="cuda"),
            torch.tensor(np.asarray(labels[:n], dtype=np.int32), device="cuda"),
            torch.tensor(np.asarray(ids[:n], dtype=np.int32), device="cuda"))


def remap_A(t):
    """域 A token t → t-1：域 A 编码 = 数字+1（数字 0-9 ↔ token 1-10，label 2-10=数字 1-9）。
    重映射后与域 B 数字区（token 0-9 = 数字 0-9）完全对齐；域 A 数据无 token 0（pad 未用），无负值。"""
    return t - 1


def load_a_ckpt_into(m, ckpt):
    """域 A ckpt（vocab=11）载入 vocab=13 模型：embedding/lm_head 行移位迁移数字语义。
    旧行 i（i=1..10）= 数字 i-1 的表示 → 新行 i-1；新行 10-12（ADD/SUB/PAD）保持随机初始化。"""
    st = torch.load(ckpt, map_location="cuda")
    st = {k.removeprefix("model."): v for k, v in st.items()}
    for key in ("inner.embed_tokens.embedding_weight", "inner.lm_head.weight"):
        old = st[key]                       # (11, 512)
        new = m.state_dict()[key].clone()   # (13, 512) 随机初始化
        new[:10] = old[1:11]                # 数字 0-9 表示迁移
        st[key] = new
    m.load_state_dict(st, assign=True)


def per_sample_loss(model, inputs, labels, ids, D=16):
    """eval 模式前向 → 逐样本 CE loss（惊喜分数）。与 eval 同 halt 步数。"""
    model.config.halt_max_steps = D
    out = []
    with torch.inference_mode():
        for start in range(0, inputs.shape[0], 128):
            b = inputs[start:start + 128]
            bs = b.shape[0]
            batch = {"inputs": b, "puzzle_identifiers": ids[start:start + bs]}
            with torch.device("cuda"):
                carry = model.initial_carry(batch)
            for _ in range(D):
                carry, outputs = model(carry, batch)
            lbl = labels[start:start + bs]
            ploss = stablemax_cross_entropy(outputs["logits"], lbl).sum(-1)  # [bs]
            out.append(ploss.cpu().numpy())
    return np.concatenate(out)


def eval_domain(model, data_dir, n, seed=0, K=1, D=16):
    """标准 eval（K=1 D=16 sigma=0，与 eval_ptrm / E2b 基线同配置）。
    域 A (sudoku): 数据 remap(-1) 后 exact = 81 位全对（E2b 基线 0.330 同判据，双射可比）
    域 B (RPN):    exact = 输出位 0（m1_train_domainB 判据，baseline 0.478）
    """
    inputs, labels, ids, _ = load_test_data(data_dir, n, seed)
    if "sudoku" in data_dir:
        inputs, labels = remap_A(inputs), remap_A(labels)
    pred, _ = ptrm_infer(model, inputs, ids, K, D, 0.0)
    if "sudoku" in data_dir:
        return float((pred == labels).all(dim=1).float().mean().item())
    return float((pred[:, 0] == labels[:, 0]).float().mean().item())


def build_memory_field(model, arm, cap, n_train=2000, seed=0):
    """域 A train 上前向得逐样本 surprise → 按臂策略建记忆场（容量 cap）。
    数据 remap(-1)（域 A 数字语义对齐域 B vocab 13 模型）"""
    inputs, labels, ids = load_split(A_DATA, n_train, "train", seed)
    inputs, labels = remap_A(inputs), remap_A(labels)
    surprise = per_sample_loss(model, inputs, labels, ids)
    mf = MemoryField(capacity=cap, strategy=("surprise" if arm == 2 else "full"), seed=seed)
    mf.update(inputs.cpu().numpy(), labels.cpu().numpy(), ids.cpu().numpy(), surprise, "A")
    return mf


def run_arm(arm, steps, cap, out_dir, seed=0):
    """arm: 0=无记忆场 1=全量写 2=惊喜写。起点 = 域 A ckpt，域 B 微调 steps 步。"""
    os.makedirs(out_dir, exist_ok=True)
    meta_b = json.load(open(f"{B_DATA}/dataset.json"))
    arch_cfg = load_arch(meta_b)

    with torch.device("cuda"):
        m = TinyRecursiveReasoningModel_ACTV1(arch_cfg)
        load_a_ckpt_into(m, A_CKPT)   # vocab 11→13 行移位迁移
        m.eval()
    model = ACTLossHead(m, "stablemax_cross_entropy")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1,
                            betas=(0.9, 0.95), fused=True)

    # 域 B 数据（训练分布：m1_train_domainB 同源协议）
    tr_in = np.load(f"{B_DATA}/train/all__inputs.npy", mmap_mode="r")
    tr_lb = np.load(f"{B_DATA}/train/all__labels.npy", mmap_mode="r")
    tr_ids = np.load(f"{B_DATA}/train/all__puzzle_identifiers.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)

    # 域 A 记忆场（臂 1/2）
    mf = None
    if arm > 0:
        mf = build_memory_field(m, arm, cap, seed=seed)
        print(f"[arm{arm}] 记忆场构建完成: {mf.stats()}", flush=True)

    # 臂 0 无记忆场 → 无回放，batch = BS（carry 维度必须匹配真实 batch）
    n_replay = int(BS * REPLAY_R) if mf is not None else 0
    batch_n = BS + n_replay
    with torch.device("cuda"):
        carry = model.initial_carry({
            "inputs": torch.zeros(batch_n, 81, dtype=torch.int32, device="cuda"),
            "labels": torch.zeros(batch_n, 81, dtype=torch.int32, device="cuda"),
            "puzzle_identifiers": torch.zeros(batch_n, dtype=torch.int32, device="cuda")})

    results = []
    t0 = time.time()
    for step in range(0, steps + 1, EVAL_EVERY):
        if step > 0:
            for it in range(EVAL_EVERY):
                if it % 100 == 0:
                    print(f"[arm{arm}] step {step} iter {it} t={time.time()-t0:.0f}s", flush=True)
                idxb = rng.choice(len(tr_in), BS, replace=False)
                batch = {
                    "inputs": torch.tensor(np.asarray(tr_in[idxb], dtype=np.int32),
                                           device="cuda"),
                    "labels": torch.tensor(np.asarray(tr_lb[idxb], dtype=np.int32),
                                           device="cuda"),
                    "puzzle_identifiers": torch.tensor(np.asarray(tr_ids[idxb],
                                                                  dtype=np.int32),
                                                       device="cuda"),
                }
                if mf is not None:
                    rb = mf.sample(n_replay)
                    if rb is not None:
                        ri, rl, rid = rb
                        batch["inputs"] = torch.cat([
                            batch["inputs"],
                            torch.tensor(ri, dtype=torch.int32, device="cuda")])
                        batch["labels"] = torch.cat([
                            batch["labels"],
                            torch.tensor(rl, dtype=torch.int32, device="cuda")])
                        batch["puzzle_identifiers"] = torch.cat([
                            batch["puzzle_identifiers"],
                            torch.tensor(rid, dtype=torch.int32, device="cuda")])
                opt.zero_grad()
                carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
                (loss / batch_n).backward()
                opt.step()
            ckpt_path = f"{out_dir}/arm{arm}_step_{step}"
            torch.save(model.state_dict(), ckpt_path)
            m2 = TinyRecursiveReasoningModel_ACTV1(arch_cfg)
            state2 = {k.removeprefix("model."): v for k, v in torch.load(
                ckpt_path, map_location="cuda").items()}
            m2.load_state_dict(state2, assign=True)
            m2.eval()
            with torch.device("cuda"):
                m2.to("cuda")
            m.train()   # 训练态恢复
            keep_a = eval_domain(m2, A_DATA, A_EVAL_N, seed=0)
            prog_b = eval_domain(m2, B_DATA, 500, seed=0)
            results.append({"step": step, "keep_A": keep_a, "prog_B": prog_b,
                            "loss": float(loss.item() / batch_n),
                            "mem": mf.stats() if mf else None,
                            "t_sec": round(time.time() - t0)})
            print(f"[arm{arm}] step {step}: keep_A={keep_a:.4f} prog_B={prog_b:.4f} "
                  f"loss={loss.item()/batch_n:.4f} ({time.time()-t0:.0f}s)", flush=True)
            json.dump(results, open(f"{out_dir}/result.json", "w"), indent=2)
            del m2
            torch.cuda.empty_cache()
    print(f"[arm{arm}] done", flush=True)


def eval_only(ckpt_dir, arm, out_json):
    """E2b 间隔重测：对已存 ckpt 测域 A 保持率（t=0 值 = 主实验 result.json 末行 keep_A）"""
    meta_b = json.load(open(f"{B_DATA}/dataset.json"))
    arch_cfg = load_arch(meta_b)
    m = TinyRecursiveReasoningModel_ACTV1(arch_cfg)
    state = {k.removeprefix("model."): v for k, v in torch.load(
        ckpt_dir, map_location="cuda").items()}
    m.load_state_dict(state, assign=True)
    m.eval()
    with torch.device("cuda"):
        m.to("cuda")
    keep_a = eval_domain(m, A_DATA, A_EVAL_N, seed=0)
    prog_b = eval_domain(m, B_DATA, 500, seed=0)
    rec = {"arm": arm, "ckpt": ckpt_dir, "keep_A": keep_a, "prog_B": prog_b}
    json.dump(rec, open(out_json, "w"), indent=2)
    print(f"[arm{arm}] eval-only: keep_A={keep_a:.4f} prog_B={prog_b:.4f} -> {out_json}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--cap", type=int, default=2000, help="记忆场容量 C")
    ap.add_argument("--arms", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="outputs/2026-08-13/m2_e2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", default=None, help="E2b 重测：ckpt 目录")
    ap.add_argument("--arm", type=int, default=None)
    ap.add_argument("--eval-out", default=None)
    args = ap.parse_args()

    if args.eval_only:
        eval_only(args.eval_only, args.arm, args.eval_out or
                  f"{args.eval_only}/eval_retest.json")
        return
    for arm in args.arms:
        run_arm(arm, args.steps, args.cap, os.path.join(args.out, f"arm{arm}"), args.seed)


if __name__ == "__main__":
    main()
