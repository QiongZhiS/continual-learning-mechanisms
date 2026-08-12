"""M0.4 域 C 可学习性 smoke：TRM 从零训组合域（E4 前置 · M0.4 后档）

- 任务：x=(c*(a+1))%10, y=(a*(c+1))%10（乘法交互非线性 f）
- 数据：train 24 组合 × 500 = 12K；test 12 全新组合（鸭嘴兽：零次推断，1 样本/组合）
- 协议：与 m1_train_domainB.py 完全一致（batch 128 · lr 1e-4 · fused AdamW ·
  mlp_t=True pos_encodings="none" · ACTLossHead wrapped 训练）
- eval：exact = 输出位 0+1 都正确（x,y 两坐标位，域 C 语义）；cell = 全 81 位
- 步数：2000（12K 样本 ≈ 21 epoch；域 C 任务维度远小于域 B，2000 步够判可学习性）
- 判读：train exact 显著 > 10%（随机水平）→ f 可学；test 12 鸭嘴兽外推 > 随机
  （注意 n=12 噪声 ±14pp，外推判据按全组合正确数报告，不单独抠单点）
用法: python m0_4_train_domainC.py [--steps 2000] [--out 输出目录]
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import yaml

from models.losses import ACTLossHead
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

DATA = "../data/domain-c-combo"
BS = 128
EVAL_EVERY = 200


def load_eval_set(data_dir, n=2000, split="test", seed=0):
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


def eval_result(model, inputs, labels, ids):
    """K=1 D=16 标准 eval；域 C exact = 位 0+1 都正确"""
    emod = model
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
    both = (pred[:, 0] == labels[:, 0]) & (pred[:, 1] == labels[:, 1])
    exact = both.float().mean().item()
    cell = (pred == labels).float().mean().item()
    x_acc = (pred[:, 0] == labels[:, 0]).float().mean().item()
    y_acc = (pred[:, 1] == labels[:, 1]).float().mean().item()
    return exact, cell, x_acc, y_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="outputs/2026-08-12/m0_4_domainC")
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
    model = ACTLossHead(m, "stablemax_cross_entropy")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                            betas=(0.9, 0.95), fused=True)

    train_in = np.load(f"{DATA}/train/all__inputs.npy", mmap_mode="r")
    train_lb = np.load(f"{DATA}/train/all__labels.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    eval_in, eval_lb, eval_ids = load_eval_set(DATA, n=12)      # 12 鸭嘴兽全用
    tr_in, tr_lb, tr_ids = load_eval_set(DATA, n=2000, split="train")

    with torch.device("cuda"):
        carry = model.initial_carry({
            "inputs": torch.zeros(BS, 81, dtype=torch.int32, device="cuda"),
            "labels": torch.zeros(BS, 81, dtype=torch.int32, device="cuda"),
            "puzzle_identifiers": torch.zeros(BS, dtype=torch.int32, device="cuda")})

    results = []
    t0 = time.time()
    for step in range(0, args.steps + 1, EVAL_EVERY):
        if step > 0:
            for _ in range(EVAL_EVERY):
                idxb = rng.choice(len(train_in), BS, replace=False)
                batch = {
                    "inputs": torch.tensor(np.asarray(train_in[idxb], dtype=np.int32),
                                           device="cuda"),
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
            m2 = TinyRecursiveReasoningModel_ACTV1(arch_cfg)
            state = {k.removeprefix("model."): v for k, v in torch.load(ckpt_path,
                                                                        map_location="cuda").items()}
            m2.load_state_dict(state, assign=True)
            m2.eval()
            with torch.device("cuda"):
                m2.to("cuda")
            exact, cell, x_acc, y_acc = eval_result(m2, eval_in, eval_lb, eval_ids)
            tr_exact, tr_cell, _, _ = eval_result(m2, tr_in, tr_lb, tr_ids)
            results.append({"step": step, "exact": exact, "cell": cell,
                            "train_exact": tr_exact, "x_acc": x_acc, "y_acc": y_acc,
                            "loss": float(loss.item() / BS)})
            print(f"step {step}: duckbill_exact={exact:.3f} train_exact={tr_exact:.3f} "
                  f"x={x_acc:.3f} y={y_acc:.3f} loss={loss.item()/BS:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            json.dump(results, open(f"{args.out}/result.json", "w"), indent=2)
            del m2
            torch.cuda.empty_cache()

    print("done", flush=True)


if __name__ == "__main__":
    main()
