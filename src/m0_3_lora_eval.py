"""临时 eval：lora_1000 ckpt（LoraCasted 包裹 → key 带 .linear. 前缀）→ unwrapped 模型

key 重映射: `xxx.linear.weight/bias` → `xxx.weight/bias`（lora_A/B 已在保存时剔除）
"""
import json
import yaml
import torch

from eval_ptrm import load_test_data, ptrm_infer
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

CKPT = "outputs/2026-08-10/m0_3_lora/lora_1000"
DATA = "../data/sudoku-extreme-1k-aug-100"

meta = json.load(open(f"{DATA}/test/dataset.json"))
arch_cfg = yaml.safe_load(open("config/arch/trm.yaml", encoding="utf-8"))
arch_cfg["puzzle_emb_ndim"] = arch_cfg["hidden_size"]
arch_cfg.pop("name", None)
arch_cfg.pop("loss", None)
arch_cfg.update(dict(batch_size=1, vocab_size=meta["vocab_size"], seq_len=meta["seq_len"],
                     num_puzzle_identifiers=meta["num_puzzle_identifiers"], causal=False,
                     mlp_t=True, pos_encodings="none"))
with torch.device("cuda"):
    model = TinyRecursiveReasoningModel_ACTV1(arch_cfg)
state = torch.load(CKPT, map_location="cuda")
state = {k.removeprefix("model.").replace(".linear.", "."): v for k, v in state.items()}
model.load_state_dict(state, assign=True)
model.eval()
inputs, labels, ids, idx = load_test_data(DATA, 200, 0)
model.config.halt_max_steps = 16
pred, _ = ptrm_infer(model, inputs, ids, 1, 16, 0.0)
em = (pred == labels).all(dim=1).float().mean().item()
ca = (pred == labels).float().mean().item()
print(f"lora_1000 (lr=1e-3): exact={em:.4f} cell={ca:.4f} (baseline 0.3300)", flush=True)
