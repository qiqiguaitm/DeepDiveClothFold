"""1-GPU smoke test: load policy, build multi-domain dataset, run 1 forward+backward."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from multi_domain_dataset import LeRobotEE6DDataset, MultiDomainDataset, build_weighted_sampler

print("=== Build datasets ===")
ROOT = "/data/shared/ubuntu/workspace/dataset_ee6d"
PROMPT = "Flatten and fold the cloth."
datasets = [
    LeRobotEE6DDataset(f"{ROOT}/Kai0_official/Task_A/base", domain_id=19, task_prompt=PROMPT),
    LeRobotEE6DDataset(f"{ROOT}/Task_A/vis_v2_merged", domain_id=20, task_prompt=PROMPT),
]
print("kai_base:", len(datasets[0]), "vis:", len(datasets[1]))

multi = MultiDomainDataset(datasets)
print("multi total:", len(multi))

# Tokenizer for task instruction
tok = AutoTokenizer.from_pretrained("facebook/bart-large")

def collate(batch):
    out = {}
    for k in batch[0].keys():
        if k == "task":
            tokens = tok([s["task"] for s in batch], padding="max_length", max_length=50, truncation=True, return_tensors="pt")
            out["task"] = [s["task"] for s in batch]
            out["observation.language.tokens"] = tokens["input_ids"]
        elif isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([s[k] for s in batch])
    return out

# Mini-batch
loader = DataLoader(multi, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate)
print("Loading batch...")
t0 = time.time()
batch = next(iter(loader))
print(f"loaded in {time.time()-t0:.1f}s")
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")
    else:
        print(f"  {k}: {type(v).__name__} ({len(v)})")

print()
print("=== Load policy ===")
t0 = time.time()
device = torch.device("cuda:0")
model = XVLAPolicy.from_pretrained("/data/shared/ubuntu/workspace/xvla_ckpts").to(device)
n = sum(p.numel() for p in model.parameters())
print(f"loaded {n/1e6:.0f}M params in {time.time()-t0:.1f}s on {device}")

print()
print("=== Forward + backward ===")
batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
t0 = time.time()
loss, log_dict = model.forward(batch_gpu)
print(f"forward: {time.time()-t0:.2f}s, loss={loss.item():.4f}")
t0 = time.time()
loss.backward()
print(f"backward: {time.time()-t0:.2f}s")
print(f"grad sample: input_proj weight grad norm = {next(p.grad.norm().item() for p in model.parameters() if p.grad is not None and p.numel() > 1000):.4f}")
print("=== SMOKE PASS ✅ ===")
