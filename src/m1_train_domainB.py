"""M1b 域 B smoke 训练：TRM 从零训 RPN 栈机算术

- 从零初始化（无 checkpoint），固定 lr 1e-4 无衰减（yaml 官方超参，避开 cosine 衰减冻结坑）
- batch 128 · wd 0.1 · betas (0.9,0.95) · fused AdamW（M0.1 patch 同款）
- mlp_t=True, pos_encodings="none"（与 eval_ptrm 加载配置完全一致，无隐藏不一致）
- eval：K=1 D=16，exact = 输出位 0 正确率（结果位），cell = 全 81 位平均
- 每 EVAL_EVERY 步存 checkpoint + eval，结果写 json
--aug-batch（通道③ 判别器）：逐样本均匀采样等价类形式（ADD 子树交换掩码，
含 mask=0 原形式）——数据集/尺寸/pass/步数/计算全部与基线匹配，
唯一变量 = 训练分布（规范形 vs 等价类均匀）。交换函数复用 m1_gen_domainB。
用法: python m1_train_domainB.py --steps 3000 [--lr 1e-4] [--out 输出目录] [--aug-batch]
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import yaml

from m1_gen_domainB import (PAD, apply_swap_mask, collect_adds, parse_seq,
                            serialize)
from models.losses import ACTLossHead
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

# PTRM_DATA 覆盖：通道③ 增广数据集（默认不变，零行为影响）
DATA = os.environ.get("PTRM_DATA", "../data/domain-b-rpn")
BS = 128
EVAL_EVERY = 500


def load_eval_set(data_dir, n=2000, split="test", seed=0):
    """split=test: 前 n 个全新结构；split=train: 固定 seed 采样 n 个训练样本
    （训练侧上限判别：train acc vs test acc = 泛化缺口）"""
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


def aug_batch_forms(rows, rng):
    """通道③ 判别器：对每行均匀采样其等价类（ADD 子树交换掩码，含 mask=0
    原形式）。行 = RPN 程序左对齐 + PAD 填充；输出长度不变（交换不改变
    token 多重集）。"""
    out = np.empty_like(rows)
    for i, row in enumerate(rows):
        seq = row[row != PAD].tolist()
        tree = parse_seq(seq)
        c = 1 << len(collect_adds(tree, []))
        m = int(rng.integers(c))
        if m:
            t = apply_swap_mask(tree, m)
            s = np.asarray(serialize(t), dtype=rows.dtype)
            out[i, :len(s)] = s
            out[i, len(s):] = PAD
        else:
            out[i] = row
    return out


def eval_result(model, inputs, labels, ids):
    """K=1 D=16 标准 eval；exact = 结果位（位 0）正确率"""
    emod = model  # unwrapped
    emod.config.halt_max_steps = 16
    with torch.inference_mode():
        pred = torch.zeros(inputs.shape[0], 81, dtype=torch.long, device="cuda")
        for start in range(0, inputs.shape[0], 128):
            b = inputs[start:start + 128]
            bs = b.shape[0]
            batch = {"inputs": b, "puzzle_identifiers": ids[start:start + bs]}
            with torch.device("cuda"):
                carry = emod.initial_carry(batch)
            for _ in range(16):
                carry, outputs = emod(carry, batch)
            pred[start:start + bs] = outputs["logits"].argmax(-1)
    exact = (pred[:, 0] == labels[:, 0]).float().mean().item()
    cell = (pred == labels).float().mean().item()
    return exact, cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="outputs/2026-08-10/m1_domainB")
    ap.add_argument("--resume", default=None, help="续训 checkpoint 路径")
    ap.add_argument("--start-step", type=int, default=0)
    ap.add_argument("--aug-batch", action="store_true",
                    help="通道③ 判别器：逐样本等价类均匀采样（匹配尺寸/步数/计算）")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    meta = json.load(open(f"{DATA}/dataset.json"))
    arch_cfg = yaml.safe_load(open("config/arch/trm.yaml", encoding="utf-8"))
    arch_cfg["puzzle_emb_ndim"] = arch_cfg["hidden_size"]
    arch_cfg.pop("name", None)
    arch_cfg.pop("loss", None)
    arch_cfg.update(dict(batch_size=BS, vocab_size=meta["vocab_size"], seq_len=meta["seq_len"],
                         num_puzzle_identifiers=meta["num_puzzle_identifiers"], causal=False,
                         mlp_t=True, pos_encodings="none"))

    with torch.device("cuda"):
        m = TinyRecursiveReasoningModel_ACTV1(arch_cfg)
    model = ACTLossHead(m, "stablemax_cross_entropy")  # wrapped 训练
    if args.resume:
        # checkpoint 自带 model. 前缀，wrapped 模型期望同前缀 → 直接加载
        model.load_state_dict(torch.load(args.resume, map_location="cuda"), assign=True)
        print(f"resumed from {args.resume}", flush=True)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                            betas=(0.9, 0.95), fused=True)

    train_in = np.load(f"{DATA}/train/all__inputs.npy", mmap_mode="r")
    train_lb = np.load(f"{DATA}/train/all__labels.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    eval_in, eval_lb, eval_ids = load_eval_set(DATA)
    tr_in, tr_lb, tr_ids = load_eval_set(DATA, n=2000, split="train")  # 训练侧上限判别

    with torch.device("cuda"):
        carry = model.initial_carry({
            "inputs": torch.zeros(BS, 81, dtype=torch.int32, device="cuda"),
            "labels": torch.zeros(BS, 81, dtype=torch.int32, device="cuda"),
            "puzzle_identifiers": torch.zeros(BS, dtype=torch.int32, device="cuda")})

    results = []
    t0 = time.time()
    for step in range(args.start_step, args.steps + 1, EVAL_EVERY):
        if step > args.start_step:
            for _ in range(EVAL_EVERY):
                idxb = rng.choice(len(train_in), BS, replace=False)
                train_x = np.asarray(train_in[idxb], dtype=np.int32)
                if args.aug_batch:
                    train_x = aug_batch_forms(train_x, rng)
                batch = {
                    "inputs": torch.tensor(train_x, device="cuda"),
                    "labels": torch.tensor(np.asarray(train_lb[idxb], dtype=np.int32),
                                           device="cuda"),
                    "puzzle_identifiers": torch.zeros(BS, dtype=torch.int32, device="cuda"),
                }
                opt.zero_grad()
                carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
                (loss / BS).backward()
                opt.step()
            ckpt_path = f"{args.out}/step_{step}"
            torch.save(model.state_dict(), ckpt_path)
            # eval 用 unwrapped 模型（从 checkpoint 重建，端到端验证）
            m2 = TinyRecursiveReasoningModel_ACTV1(arch_cfg)
            state = {k.removeprefix("model."): v for k, v in torch.load(ckpt_path,
                                                                        map_location="cuda").items()}
            m2.load_state_dict(state, assign=True)
            m2.eval()
            with torch.device("cuda"):
                m2.to("cuda")
            exact, cell = eval_result(m2, eval_in, eval_lb, eval_ids)
            tr_exact, tr_cell = eval_result(m2, tr_in, tr_lb, tr_ids)
            results.append({"step": step, "exact": exact, "cell": cell,
                            "train_exact": tr_exact, "train_cell": tr_cell,
                            "loss": float(loss.item() / BS)})
            print(f"step {step}: exact={exact:.4f} train_exact={tr_exact:.4f} "
                  f"cell={cell:.4f} loss={loss.item()/BS:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            json.dump(results, open(f"{args.out}/result.json", "w"), indent=2)
            del m2
            torch.cuda.empty_cache()

    print("done", flush=True)


if __name__ == "__main__":
    main()
