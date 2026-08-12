"""M0.3b supplementary check: LoRA anti-forgetting quick test — retention after subset fine-tuning

With a single domain, "old knowledge" cannot be constructed (no second domain); approximate out-of-distribution:
fine-tune only on train[:2000], then measure performance on the **out-of-subset** train[10000:10500] (the pre-fine-tuning knowledge surface).
- LoRA (rank16 lr3e-4) subset 500 steps <- freshly trained
- control 1: baseline 23437 (not fine-tuned)
- control 2: human-label_500 (full-parameter same-subset fine-tune 500 steps, existing ckpt)
Criterion: on out-of-subset data, LoRA retention loss < full fine-tuning -> the adapter's continual-learning property holds."""
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
HL_CKPT = "outputs/2026-08-10/m0_3_distill/human-label_500"
DATA = "../data/sudoku-extreme-1k-aug-100"
OUT = "outputs/2026-08-10/m0_3_lora_holdout"
BS = 128
STEPS = 500
LR = 3e-4
RANK = 16
ALPHA = 16
SUB = 2000            # fine-tuning subset = train[:SUB]
HO_LO, HO_HI = 10000, 10500  # out-of-subset holdout range

os.makedirs(OUT, exist_ok=True)
meta = json.load(open(f"{DATA}/test/dataset.json"))

# test set (200 puzzles seed 0, same source as all of M0) + holdout set (out-of-subset train)
inputs, labels, ids, idx = load_test_data(DATA, 200, 0)
train_in = np.load(f"{DATA}/train/all__inputs.npy", mmap_mode="r")
train_lb = np.load(f"{DATA}/train/all__labels.npy", mmap_mode="r")
train_pid = np.load(f"{DATA}/train/all__puzzle_identifiers.npy", mmap_mode="r")
ho_in = torch.tensor(np.asarray(train_in[HO_LO:HO_HI], dtype=np.int32), device="cuda")
ho_lb = torch.tensor(np.asarray(train_lb[HO_LO:HO_HI], dtype=np.int32), device="cuda")
ho_pid = torch.tensor(np.asarray(train_pid[HO_LO:HO_HI], dtype=np.int32), device="cuda")


class LoraCasted(torch.nn.Module):
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
    model.load_state_dict(torch.load(CKPT, map_location="cuda"), assign=True)
    return model


def eval_unwrapped(ckpt_path, in_ten, lb_ten, pid_ten):
    emod = __import__("eval_ptrm").load_model(
        ckpt_path, "config/arch/trm.yaml", meta["vocab_size"], meta["seq_len"],
        meta["num_puzzle_identifiers"])
    emod.config.halt_max_steps = 16
    pred, _ = ptrm_infer(emod, in_ten, pid_ten, 1, 16, 0.0)
    return ((pred == lb_ten).all(dim=1).float().mean().item(),
            (pred == lb_ten).float().mean().item())


# ---- LoRA subset fine-tuning (the only freshly trained part) ----
model = make_model()
for name, mod in list(model.named_modules()):
    if isinstance(mod, CastedLinear):
        parent = model
        for p in name.split(".")[:-1]:
            parent = getattr(parent, p)
        setattr(parent, name.split(".")[-1], LoraCasted(mod, RANK, ALPHA))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=LR, weight_decay=0.0, fused=True)
rng = np.random.default_rng(7)
with torch.device("cuda"):
    carry = model.initial_carry({
        "inputs": torch.zeros(BS, 81, dtype=torch.int32, device="cuda"),
        "labels": torch.zeros(BS, 81, dtype=torch.int32, device="cuda"),
        "puzzle_identifiers": torch.zeros(BS, dtype=torch.int32, device="cuda")})
model.train()
t0 = time.time()
for step in range(STEPS):
    idxb = rng.choice(SUB, BS, replace=False)
    batch = {
        "inputs": torch.tensor(np.asarray(train_in[idxb], dtype=np.int32), device="cuda"),
        "labels": torch.tensor(np.asarray(train_lb[idxb], dtype=np.int32), device="cuda"),
        "puzzle_identifiers": torch.zeros(BS, dtype=torch.int32, device="cuda"),
    }
    opt.zero_grad()
    carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
    (loss / BS).backward()
    opt.step()
state = {k.replace(".linear.", "."): v for k, v in model.state_dict().items()
         if not k.endswith("lora_A") and not k.endswith("lora_B")}
lora_path = f"{OUT}/lora_sub{SUB}_{STEPS}"
torch.save(state, lora_path)
print(f"LoRA sub{SUB} {STEPS} steps done ({time.time()-t0:.0f}s)", flush=True)
del model
torch.cuda.empty_cache()

# ---- three models x two eval sets ----
rows = {}
for tag, ck in [("baseline_23437", CKPT), ("lora_sub2000", lora_path),
                ("full_sub2000(human-label)", HL_CKPT)]:
    te, tc = eval_unwrapped(ck, inputs, labels, ids)
    he, hc = eval_unwrapped(ck, ho_in, ho_lb, ho_pid)
    rows[tag] = {"test_exact": te, "test_cell": tc,
                 "holdout_exact": he, "holdout_cell": hc}
    print(f"{tag}: test exact={te:.4f} | holdout exact={he:.4f} "
          f"(cell {tc:.4f}/{hc:.4f})", flush=True)

json.dump({"sub": SUB, "holdout_range": [HO_LO, HO_HI], "steps": STEPS, "rows": rows},
          open(f"{OUT}/result.json", "w"), indent=2)
print("saved -> " + f"{OUT}/result.json", flush=True)
