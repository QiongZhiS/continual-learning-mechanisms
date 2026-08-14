"""M1 逆向模型训练（v0.16 S3 预注册判别 · 第 2 步）

GCML 式方向场学习（只用最终**正确**的 rollout 轨迹）：
- 样本: (z_L^t, target_t = (z_L^48 − z_L^t)/‖·‖_F)，t = 0..46，权重 = 剩余距离 ‖z_L^48 − z_L^t‖_F
- 条件: PuzzleEncoder = 冻结 TRM token 嵌入（step_4000）+ 单头注意力池化 + 长度特征
- 架构: per-position 共享 MLP（~0.5M 参数），输入 concat(z_L[p], prepr, ctx(mean/meansq 池化))
- 损失: 加权余弦；val 用未加权 mean(1−cos)（10% 谜题留出，无泄漏）
- 增强: 输入加噪 σ=0.2（匹配 eval 噪声分布）；训练随机抽 24/97 位置（per-position 权重共享）
- 早停: val patience 4，max 20 epochs，TF32
用法: python m1_inv_model.py [--data outputs/2026-08-13/inv_data] [--out outputs/2026-08-13/inv_model]
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from torch import nn

from eval_ptrm import load_model

DATA_DIR = "outputs/2026-08-13/inv_data"
OUT_DIR = "outputs/2026-08-13/inv_model"
CKPT = "outputs/2026-08-10/m1_domainB/step_4000"
PAD = 12
H = 512
P = 97  # 81 token + 16 puzzle-emb 位置
D = 48
SIGMA_AUG = 0.2
SEED = 0
SUB_POS = 24  # 训练时每样本随机抽位置数


class PuzzleEncoder(nn.Module):
    """冻结 TRM token 嵌入上的单头注意力池化 + 长度特征 → [B, 257]"""

    def __init__(self, H=512, out=256):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, H))
        self.q = nn.Linear(H, 256)
        self.k = nn.Linear(H, 256)
        self.v = nn.Linear(H, 256)
        self.out = nn.Linear(256, out)

    def forward(self, emb):  # emb [B, 81, 512] fp32（冻结，含 PAD 位）
        B = emb.shape[0]
        x = torch.cat([self.cls.expand(B, 1, -1), emb], 1)  # [B, 82, 512]
        q = self.q(x[:, :1])                       # [B, 1, 256]
        a = (q @ self.k(x).transpose(-1, -2)) / 16.0
        a = a.softmax(-1)
        ctx = a @ self.v(x)                        # [B, 1, 256]
        return self.out(ctx.squeeze(1))            # [B, 256]


class DirModel(nn.Module):
    """per-position 共享 MLP：输入 (z_L[p], prepr, ctx) → Δz[p]"""

    def __init__(self, H=512, repr_dim=257, ctx_dim=64, hid=256):
        super().__init__()
        self.ctx_net = nn.Sequential(nn.Linear(H * 2, ctx_dim), nn.SiLU())
        self.net = nn.Sequential(
            nn.Linear(H + repr_dim + ctx_dim, hid), nn.SiLU(),
            nn.Linear(hid, H),
        )

    def forward(self, zL, prepr):
        # zL [B, P, H] fp32; prepr [B, repr_dim] fp32
        B, Pp, _ = zL.shape
        m = zL.mean(1)
        ctx = self.ctx_net(torch.cat([m, m * m], -1))          # [B, ctx]
        x = torch.cat([zL,
                       prepr.unsqueeze(1).expand(B, Pp, -1),
                       ctx.unsqueeze(1).expand(B, Pp, -1)], -1)
        return self.net(x)                                      # [B, P, H]


def puzzle_repr(model, inputs, enc):
    """inputs [B, 81] int64 → [B, 257] fp32（冻结嵌入 + 注意力池化 + 长度）"""
    emb = model.inner.embed_tokens(inputs.to(torch.int32)).float()  # [B, 81, 512]
    with torch.no_grad():
        emb = emb.detach()
    r = enc(emb)                                             # [B, 256]
    length = (inputs != PAD).sum(1, keepdim=True).float() / inputs.shape[1]
    return torch.cat([r, length], -1)                        # [B, 257]


def assemble(data_dir, model, enc, max_correct=1200, save=True):
    """从 chunk npz 组装样本集（含 puzzle_repr）。返回 dict + 路径"""
    cache = os.path.join(data_dir, "samples.npz")
    if os.path.exists(cache) and save:
        d = np.load(cache)
        print(f"loaded cached samples: {cache}", flush=True)
        return d

    files = sorted(glob.glob(os.path.join(data_dir, "chunk_*.npz")))
    X, T, W, INP = [], [], [], []
    for f in files:
        d = np.load(f)
        for pi in range(len(d["correct"])):
            if not d["correct"][pi]:
                continue
            traj = d["traj"][:, pi].astype(np.float32)      # [48, 97, 512]
            fin = traj[-1]
            diff = fin[None] - traj[:-1]                    # [47, 97, 512]
            nrm = np.linalg.norm(diff.reshape(47, -1), axis=1)
            nrm_safe = np.where(nrm < 1e-8, 1.0, nrm)
            X.append(traj[:-1].astype(np.float16))
            T.append((diff / nrm_safe[:, None, None]).astype(np.float16))
            W.append(nrm)
            INP.append(d["inputs"][pi])
            if len(X) >= max_correct:
                break
        if len(X) >= max_correct:
            break
    X = np.ascontiguousarray(np.concatenate(X, 0))          # [n_corr*47, P, H]
    T = np.ascontiguousarray(np.concatenate(T, 0))
    W = np.concatenate(W).astype(np.float32)
    INP = np.asarray(INP, dtype=np.int16)
    print(f"assembled X {X.shape} T {T.shape} W {W.shape} inputs {INP.shape}", flush=True)

    # puzzle_repr（GPU 分块）
    reprs = []
    enc.cuda().eval()
    with torch.inference_mode():
        for s in range(0, len(INP), 256):
            b = torch.tensor(INP[s:s + 256], device="cuda")
            reprs.append(puzzle_repr(model, b, enc).cpu().numpy())
    reprs = np.concatenate(reprs, 0).astype(np.float32)      # [N_corr, 257]
    if save:
        np.savez(cache, X=X, T=T, W=W, inputs=INP, reprs=reprs)
        print(f"saved cache -> {cache}", flush=True)
    return {"X": X, "T": T, "W": W, "inputs": INP, "reprs": reprs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--aug-sigma", type=float, default=SIGMA_AUG)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    torch.set_float32_matmul_precision("high")  # TF32
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    meta = json.load(open("../data/domain-b-rpn/dataset.json", encoding="utf-8"))
    model = load_model(CKPT, "config/arch/trm.yaml", meta["vocab_size"],
                       meta["seq_len"], meta["num_puzzle_identifiers"])
    model.eval()

    enc = PuzzleEncoder().cuda()
    d = assemble(args.data, model, enc)
    X, T, W, reprs = d["X"], d["T"], d["W"], d["reprs"]
    M, Pp, Hh = X.shape
    assert (Pp, Hh) == (P, H), (Pp, Hh)
    n_puzzles = X.shape[0] // 47
    pidx = np.repeat(np.arange(n_puzzles), 47)      # 每谜题 47 个样本 (t=0..46)
    rng = np.random.default_rng(args.seed)
    pperm = rng.permutation(n_puzzles)
    n_val = max(1, n_puzzles // 10)
    val_p = set(pperm[:n_val].tolist())
    train_mask = np.array([p not in val_p for p in pidx])
    print(f"samples {M} puzzles {n_puzzles} train {train_mask.sum()} val {M - train_mask.sum()}",
          flush=True)

    dir_model = DirModel().cuda()
    n_param = sum(p.numel() for p in dir_model.parameters())
    print(f"DirModel params: {n_param}", flush=True)

    opt = torch.optim.AdamW(list(dir_model.parameters()) + list(enc.parameters()),
                            lr=args.lr, weight_decay=1e-4, fused=True)

    def loss_and_cos(dpred, target, w):
        # dpred [B, P, H], target 单位向量, w [B] 权重（未归一）
        dnorm = torch.linalg.vector_norm(dpred, dim=(-2, -1))       # [B]
        cos = (dpred * target).sum((-2, -1)) / (dnorm + 1e-8)
        wb = w / w.mean()
        return (wb * (1 - cos)).mean(), cos

    X_t = torch.tensor(X)       # CPU fp16
    T_t = torch.tensor(T)
    W_t = torch.tensor(W)
    repr_t = torch.tensor(reprs)

    def run_batch(idx, train):
        x = X_t[idx].float().to("cuda", non_blocking=True)   # [B, P, H]
        t = T_t[idx].float().to("cuda", non_blocking=True)
        w = W_t[idx].to("cuda")
        prepr = repr_t[pidx[idx]].to("cuda")                 # [B, 257]
        if train and args.aug_sigma > 0:
            x = x + args.aug_sigma * torch.randn_like(x)
        if train:
            # 随机抽位置（ctx 用全量 z_L 计算，成本低）
            Bs = x.shape[0]
            pos = torch.argsort(torch.rand(Bs, Pp, device="cuda"), dim=1)[:, :SUB_POS]
            x_s = x.gather(1, pos.unsqueeze(-1).expand(Bs, SUB_POS, H))
            dp = dir_model(x_s, prepr)
            t_s = t.gather(1, pos.unsqueeze(-1).expand(Bs, SUB_POS, H))
        else:
            dp = dir_model(x, prepr)
            t_s = t
        loss, cos = loss_and_cos(dp, t_s, w)
        return loss, cos

    tr_idx = np.where(train_mask)[0]
    val_idx = np.where(~train_mask)[0]
    curves = {"train_loss": [], "val_loss": [], "val_cos": [], "epoch_sec": []}
    best_val, best_state, patience = 1e9, None, 0
    t0 = time.time()
    for ep in range(args.epochs):
        ep0 = time.time()
        dir_model.train(); enc.train()
        perm = rng.permutation(tr_idx)
        tot_loss, nb = 0.0, 0
        for s in range(0, len(perm), args.batch):
            idx = perm[s:s + args.batch]
            opt.zero_grad(set_to_none=True)
            loss, _ = run_batch(idx, train=True)
            loss.backward()
            opt.step()
            tot_loss += loss.item()
            nb += 1
        dir_model.eval(); enc.eval()
        with torch.inference_mode():
            vl, tc, cnt = 0.0, 0.0, 0
            for s in range(0, len(val_idx), 1024):
                loss, cos = run_batch(val_idx[s:s + 1024], train=False)
                n = len(val_idx[s:s + 1024])
                vl += loss.item() * n
                tc += cos.mean().item() * n
                cnt += n
        vl /= cnt
        tc /= cnt
        curves["train_loss"].append(round(tot_loss / nb, 4))
        curves["val_loss"].append(round(vl, 4))
        curves["val_cos"].append(round(tc, 4))
        curves["epoch_sec"].append(round(time.time() - ep0, 1))
        print(f"ep {ep}: train_loss={curves['train_loss'][-1]} "
              f"val_loss={curves['val_loss'][-1]} val_cos={curves['val_cos'][-1]} "
              f"({curves['epoch_sec'][-1]}s)", flush=True)
        if vl < best_val:
            best_val, patience = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in
                          dir_model.state_dict().items()} | \
                         {f"enc.{k}": v.detach().cpu().clone() for k, v in
                          enc.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                print(f"early stop at ep {ep}", flush=True)
                break

    torch.save(best_state, os.path.join(args.out, "inv_model.pt"))
    config = {"params": n_param, "lr": args.lr, "batch": args.batch,
              "epochs_done": len(curves["val_loss"]), "best_epoch": curves["val_loss"].index(min(curves["val_loss"])),
              "best_val_loss": best_val, "aug_sigma": args.aug_sigma,
              "sub_pos": SUB_POS, "P": P, "H": H, "D": D, "repr_dim": reprs.shape[1],
              "n_correct_puzzles": n_puzzles, "n_samples": int(M),
              "total_seconds": round(time.time() - t0, 1)}
    json.dump(config, open(os.path.join(args.out, "train_config.json"), "w"), indent=2)
    json.dump(curves, open(os.path.join(args.out, "loss_curves.json"), "w"), indent=2)
    print("saved ->", os.path.join(args.out, "inv_model.pt"), flush=True)
    print(json.dumps(config, indent=2), flush=True)


if __name__ == "__main__":
    main()
