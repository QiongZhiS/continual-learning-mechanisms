"""M0.3b adapter 可微调性验证：LoRA 挂 TRM 全部 CastedLinear（SwiGLU gate_up/down + lm_head/q_head）

冻结底座，只训 LoRA（rank 8），微调 1000 步，K=1 D=16 eval 对比基线 0.33。
判据：test exact 显著提升 → adapter 在 TRM 上可微调 → E3-D 夜间增量微调可行。

训练用 ACTLossHead-wrapped 模型（batch_size=BS 构造，loss 从 wrapped forward 取）；
eval 重建 unwrapped 模型（eval_ptrm.load_model + ptrm_infer，M0.3a 已验证路径）。
checkpoint 存 wrapped state_dict 但剔除 LoRA 参数 → unwrapped 加载时无 unexpected keys。
"""
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

CKPT = "checkpoints/Sudoku-extreme-1k-aug-100-ACT-torch/TinyRecursiveReasoningModel_ACTV1 precise-muskox/step_23437"
DATA = "../data/sudoku-extreme-1k-aug-100"
OUT = "outputs/2026-08-10/m0_3_lora"
BS = 128
TOTAL = 1000
LR = 3e-4
RANK = 16
ALPHA = 16

os.makedirs(OUT, exist_ok=True)
meta = json.load(open(f"{DATA}/test/dataset.json"))


class LoraCasted(torch.nn.Module):
    """零初始化 B → 初始行为 = 原 CastedLinear，只新增低秩通路（dtype cast 照抄原层）"""
    def __init__(self, linear, r, alpha):
        super().__init__()
        self.linear = linear
        self.scale = alpha / r
        dev = linear.weight.device  # 参数必须与权重同设备（注入发生在模型 load 到 cuda 之后）
        self.lora_A = torch.nn.Parameter(torch.randn(linear.weight.shape[1], r,
                                                     device=dev) * 0.01)
        self.lora_B = torch.nn.Parameter(torch.zeros(r, linear.weight.shape[0], device=dev))
        for p in linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        dtype = x.dtype
        return self.linear(x) + (x @ self.lora_A.to(dtype) @ self.lora_B.to(dtype)) * self.scale


def make_model():
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
    model.load_state_dict(torch.load(CKPT, map_location="cuda"), assign=True)  # checkpoint 自带 model. 前缀
    return model


# ---- LoRA 注入（替换全部 CastedLinear，不是 nn.Linear）----
model = make_model()
n_lin = n_param_lora = 0
for name, mod in list(model.named_modules()):
    if isinstance(mod, CastedLinear):
        parent = model
        for p in name.split(".")[:-1]:
            parent = getattr(parent, p)
        setattr(parent, name.split(".")[-1], LoraCasted(mod, RANK, ALPHA))
        n_lin += 1
        n_param_lora += RANK * (mod.weight.shape[1] + mod.weight.shape[0])
        print(f"lora: {name} ({mod.weight.shape[1]}->{mod.weight.shape[0]})", flush=True)
n_param_all = sum(p.numel() for p in model.parameters())
print(f"{n_lin} CastedLinear layers, LoRA params {n_param_lora} "
      f"({n_param_lora / n_param_all:.2%} of total)", flush=True)

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=LR, weight_decay=0.0, fused=True)

train_in = np.load(f"{DATA}/train/all__inputs.npy", mmap_mode="r")
train_lb = np.load(f"{DATA}/train/all__labels.npy", mmap_mode="r")
rng = np.random.default_rng(1)
inputs, labels, ids, idx = load_test_data(DATA, 200, 0)

# 占位 batch 必须含 labels：ACTLossHead 从 new_carry.current_data["labels"] 取 loss 目标
with torch.device("cuda"):
    carry = model.initial_carry({
        "inputs": torch.zeros(BS, 81, dtype=torch.int32, device="cuda"),
        "labels": torch.zeros(BS, 81, dtype=torch.int32, device="cuda"),
        "puzzle_identifiers": torch.zeros(BS, dtype=torch.int32, device="cuda")})

model.train()
t0 = time.time()
for step in range(TOTAL):
    idxb = rng.choice(len(train_in), BS, replace=False)
    batch = {
        "inputs": torch.tensor(np.asarray(train_in[idxb], dtype=np.int32), device="cuda"),
        "labels": torch.tensor(np.asarray(train_lb[idxb], dtype=np.int32), device="cuda"),
        "puzzle_identifiers": torch.zeros(BS, dtype=torch.int32, device="cuda"),
    }
    opt.zero_grad()
    carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
    (loss / BS).backward()
    opt.step()
    if (step + 1) % 200 == 0:
        print(f"step {step+1}: loss={loss.item()/BS:.4f} ({time.time()-t0:.0f}s)", flush=True)

# ---- 存 checkpoint：剔除 LoRA 参数 + key 重映射（xxx.linear.weight → xxx.weight），unwrapped 加载零 unexpected ----
state = {k.replace(".linear.", "."): v for k, v in model.state_dict().items()
         if not k.endswith("lora_A") and not k.endswith("lora_B")}
ckpt_path = f"{OUT}/lora_{TOTAL}"
torch.save(state, ckpt_path)
model.eval()
del model
torch.cuda.empty_cache()

# ---- eval：重建 unwrapped 模型（removeprefix）----
emod = __import__("eval_ptrm").load_model(
    ckpt_path, "config/arch/trm.yaml", meta["vocab_size"], meta["seq_len"],
    meta["num_puzzle_identifiers"])
emod.config.halt_max_steps = 16
pred, _ = ptrm_infer(emod, inputs, ids, 1, 16, 0.0)
em = (pred == labels).all(dim=1).float().mean().item()
ca = (pred == labels).float().mean().item()
print(f"LoRA rank={RANK} {TOTAL} steps: exact={em:.4f} cell={ca:.4f} "
      f"(baseline 0.3300, {time.time()-t0:.0f}s)", flush=True)
json.dump({"ckpt": CKPT, "rank": RANK, "lr": LR, "steps": TOTAL,
           "exact": em, "cell": ca, "baseline_exact": 0.33},
          open(f"{OUT}/result.json", "w"), indent=2)
