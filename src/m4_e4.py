"""E4 组合泛化（鸭嘴兽测试）：双轨 vs 单轨（M2 缺口 · v0.33 B1 降档声明 n=3）

- 任务：域 C 组合域 f: x=(c*(a+1))%10, y=(a*(c+1))%10（非线性交互，平凡线性组合
  外推基线 acc=0.083）；train 24 组合 × 500 = 12K；test 12 全新组合零次推断
- 变量（§4 E4 预注册）：组合表征机制——高维特征保留（记忆侧）vs 低维压缩（行动侧）
  - 单轨（对照）：标准 TRM（特征嵌入 → 递归动力学 → lm_head 读出）= 通用组合能力基线
  - 双轨（机制）：标准 TRM + 在输入位 2 注入**高维向量加法组合** v = E[c]+E[a]
    （脑科学认知制图：神经向量加法；记忆侧保留高维组合，行动侧标准读出），
    其余架构/协议与单轨逐位一致（唯一变量 = 注入）
- 判据（v0.33 已登记）：双轨零次推断 > 单轨 且 > 平凡基线（0.083）；n=3/臂，
  d_min≈3.1（B1 降档 · 探索性口径）；报告完整分布 + 中位数 + 配对 d（同种子对）
- 距离梯度：test 12 组合按"特征值新颖度"分两层——group0 两特征值均已见（7 组合，
  仅配对未见）/ group1 一个特征值未见 c=2（5 组合）；报告分层准确率（n 小，探索性曲线）
- 协议：与 m0_4_train_domainC.py 一致（batch 128 · lr 1e-4 · fused AdamW ·
  mlp_t=True pos_encodings="none" · ACTLossHead · v0.23 组合分层采样（2026-08-18 修复））
- §3.5：每次运行自动写 run_meta.json（config + git commit + seed + 数据版本）
用法: python m4_e4.py --arm single|dual --seed N [--steps 6000] [--lr 1e-4] [--out 输出目录]
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch
import yaml

from models.losses import ACTLossHead
from models.recursive_reasoning.trm import (TinyRecursiveReasoningModel_ACTV1,
                                            TinyRecursiveReasoningModel_ACTV1_Inner,
                                            TinyRecursiveReasoningModel_ACTV1InnerCarry)

DATA = "../data/domain-c-combo"
BS = 128
EVAL_EVERY = 500


class DualTrackInner(TinyRecursiveReasoningModel_ACTV1_Inner):
    """双轨 inner：在输入嵌入位 puzzle_emb_len+2 注入高维向量加法组合
    v = embed_scale * (E[c] + E[a])（记忆侧保留），其余与单轨 inner 完全一致。"""

    def forward(self, carry, batch):
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        feats = batch["inputs"][:, :2].to(torch.int32)  # (c, a) 原始 token
        e = self.embed_tokens(feats)                    # [B, 2, H]（未缩放）
        v = self.embed_scale * (e[:, 0] + e[:, 1])      # 与 _input_embeddings 同尺度
        input_embeddings = input_embeddings.clone()
        input_embeddings[:, self.puzzle_emb_len + 2] = v
        seq_info = dict(cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None)
        z_H, z_L = carry.z_H, carry.z_L
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class DualTrackModel(TinyRecursiveReasoningModel_ACTV1):
    """双轨模型：外层 ACT 接口不变，inner 换成 DualTrackInner。"""

    def __init__(self, config_dict: dict):
        super().__init__(config_dict)
        self.inner = DualTrackInner(self.config)


def make_model(arm, arch_cfg):
    if arm == "dual":
        return DualTrackModel(arch_cfg)
    return TinyRecursiveReasoningModel_ACTV1(arch_cfg)


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
    """K=1 D=16 标准 eval；exact = 位 0+1 都正确（x,y 两坐标位）；返回逐样本 both-correct"""
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
    both = ((pred[:, 0] == labels[:, 0]) & (pred[:, 1] == labels[:, 1])).float()
    exact = both.mean().item()
    cell = (pred == labels).float().mean().item()
    x_acc = (pred[:, 0] == labels[:, 0]).float().mean().item()
    y_acc = (pred[:, 1] == labels[:, 1]).float().mean().item()
    return exact, cell, x_acc, y_acc, both


def load_groups(meta):
    """test 12 组合按'距训练组合的最小替换距离'分层（v0.6.2 距离梯度）。

    替换距离 = min_{train 组合 (c,a)} ([c≠c*] + [a≠a*])（特征值替换数）。
    2026-08-19 修正：原'特征值未见'定义失效——train 实际覆盖全部 6 色×6 动物值
    （train 含 (2,10)，c=2 亦在训练），12 测试组合均为'两值均见仅配对未见'，
    最小替换距离全部 = 1 → 梯度在此冻结数据上退化为单层；按诚实报告降级。
    返回 {dist: [样本下标]}（当前仅 dist=1 层，d* 无定义）。
    """
    train_pairs = [(c, a) for c, a in meta["train_combos"]]
    groups = {}
    for i, (c, a) in enumerate(meta["test_combos"]):
        d = min(int(c != tc) + int(a != ta) for tc, ta in train_pairs)
        groups.setdefault(d, []).append(i)
    return groups


def write_run_meta(out, args, commit):
    meta = {
        "experiment": f"E4 组合泛化（鸭嘴兽 · {args.arm} 臂）",
        "script": "m4_e4.py",
        "git_commit": commit,
        "config": {
            "arm": args.arm, "steps": args.steps, "lr": args.lr,
            "batch_size": BS, "weight_decay": 0.1, "betas": [0.9, 0.95],
            "optimizer": "AdamW fused", "eval_every": EVAL_EVERY,
            "sampling": "v0.23 组合分层采样（2026-08-18 修复版）",
            "seed": args.seed,
            "data": "../data/domain-c-combo（2026-08-12 冻结）",
            "env": "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128; DISABLE_COMPILE=1",
        },
        "judge": "双轨 duckbill > 单轨 且 > 平凡基线 0.083；n=3 B1（d_min≈3.1，探索性）；配对 d（同种子对）",
        "run_meta_written_by": "m4_e4.py（§3.5 自动落盘）",
    }
    os.makedirs(out, exist_ok=True)
    json.dump(meta, open(os.path.join(out, "run_meta.json"), "w"), indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["single", "dual"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", default=None, help="续训 checkpoint 路径（2026-08-19 加，§3.5 断点续跑）")
    ap.add_argument("--start-step", type=int, default=0)
    args = ap.parse_args()
    if args.out is None:
        args.out = f"outputs/2026-08-18/m4_e4/{args.arm}/seed{args.seed}"
    os.makedirs(args.out, exist_ok=True)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    write_run_meta(args.out, args, commit)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    meta = json.load(open(f"{DATA}/dataset.json"))
    arch_cfg = yaml.safe_load(open("config/arch/trm.yaml", encoding="utf-8"))
    arch_cfg["puzzle_emb_ndim"] = arch_cfg["hidden_size"]
    arch_cfg.pop("name", None)
    arch_cfg.pop("loss", None)
    arch_cfg.update(dict(batch_size=BS, vocab_size=meta["vocab_size"], seq_len=meta["seq_len"],
                         num_puzzle_identifiers=meta["num_puzzle_identifiers"], causal=False,
                         mlp_t=True, pos_encodings="none"))

    with torch.device("cuda"):
        m = make_model(args.arm, arch_cfg)
    model = ACTLossHead(m, "stablemax_cross_entropy")
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location="cuda"), assign=True)
        print(f"resumed from {args.resume}", flush=True)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                            betas=(0.9, 0.95), fused=True)

    train_in = np.load(f"{DATA}/train/all__inputs.npy", mmap_mode="r")
    train_lb = np.load(f"{DATA}/train/all__labels.npy", mmap_mode="r")
    combo_id = np.asarray(train_in[:, 0]) * 12 + np.asarray(train_in[:, 1])
    cids = np.unique(combo_id)
    buckets = {int(cid): np.where(combo_id == cid)[0] for cid in cids}
    N_PER, N_EXTRA = BS // len(cids), BS - (BS // len(cids)) * len(cids)

    def stratified_batch():
        idx = []
        for cid in rng.permutation(cids):
            b = buckets[int(cid)]
            idx.extend(b[rng.choice(len(b), N_PER, replace=False)])
        if N_EXTRA:
            idx.extend(rng.choice(len(train_in), N_EXTRA, replace=False))
        return np.asarray(idx)

    eval_in, eval_lb, eval_ids = load_eval_set(DATA, n=12)
    tr_in, tr_lb, tr_ids = load_eval_set(DATA, n=2000, split="train")
    groups = load_groups(meta)

    with torch.device("cuda"):
        carry = model.initial_carry({
            "inputs": torch.zeros(BS, 81, dtype=torch.int32, device="cuda"),
            "labels": torch.zeros(BS, 81, dtype=torch.int32, device="cuda"),
            "puzzle_identifiers": torch.zeros(BS, dtype=torch.int32, device="cuda")})

    results = []
    if args.resume and os.path.exists(f"{args.out}/result.json"):
        results = json.load(open(f"{args.out}/result.json"))  # 续跑合并前段曲线（§3.5）
    t0 = time.time()
    for step in range(args.start_step, args.steps + 1, EVAL_EVERY):
        if step > args.start_step:
            for _ in range(EVAL_EVERY):
                idxb = stratified_batch()
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
            m2 = make_model(args.arm, arch_cfg)
            state = {k.removeprefix("model."): v for k, v in torch.load(ckpt_path,
                                                                        map_location="cuda").items()}
            m2.load_state_dict(state, assign=True)
            m2.eval()
            with torch.device("cuda"):
                m2.to("cuda")
            exact, cell, x_acc, y_acc, both = eval_result(m2, eval_in, eval_lb, eval_ids)
            tr_exact, tr_cell, _, _, _ = eval_result(m2, tr_in, tr_lb, tr_ids)
            both_np = both.detach().cpu().numpy()
            # 距离梯度：按最小替换距离分层（当前冻结数据退化为单层 dist=1，见 load_groups）
            dist_acc = {}
            for d, idxs in groups.items():
                dist_acc[f"dist{d}_acc"] = float(both_np[idxs].mean()) if idxs else None
            results.append({"step": step, "duckbill_exact": exact, "cell": cell,
                            "train_exact": tr_exact, "x_acc": x_acc, "y_acc": y_acc,
                            **dist_acc, "loss": float(loss.item() / BS)})
            dist_str = " ".join(f"d{d}={v:.3f}" if v is not None else f"d{d}=-" 
                                for d, v in sorted(dist_acc.items()))
            print(f"step {step}: duckbill={exact:.3f} {dist_str} "
                  f"train={tr_exact:.3f} x={x_acc:.3f} y={y_acc:.3f} loss={loss.item()/BS:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            json.dump(results, open(f"{args.out}/result.json", "w"), indent=2)
            del m2
            torch.cuda.empty_cache()

    print("done", flush=True)


if __name__ == "__main__":
    main()
