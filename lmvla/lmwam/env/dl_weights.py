#!/usr/bin/env python
# 下载 LaWAM 复现所需 HF 权重(hf-mirror + huggingface_hub, 见 docs/download_methods.md)
# 训练数据集(libero_merged/robotwin_merged)eval 不需要, 故不在此下载。
import os, time, sys
# 清代理 + hf-mirror (download_methods.md 铁律)
for k in ["http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","all_proxy"]:
    os.environ.pop(k, None)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import snapshot_download

ROOT = os.path.dirname(os.path.abspath(__file__))
JOBS = [
    # (repo_id, repo_type, local_dir(相对 repo root))
    ("Qwen/Qwen3-VL-2B-Instruct",            "model", "results/Checkpoints/qwen3_weights"),
    ("facebook/dinov3-vitb16-pretrain-lvd1689m","model","weights/dinov3-vitb16-pretrain-lvd1689m"),
    ("jialei02/lawam_pretrain",              "model", "results/Checkpoints/pretrain/lawam_pretrain"),
    ("jialei02/lawam_libero_sft_release",    "model", "results/Checkpoints/libero/lawam_libero_sft_release"),
    ("jialei02/lawam_robotwin_sft_release",  "model", "results/Checkpoints/robotwin/lawam_robotwin_sft_release"),
]

def dl(repo, rtype, dst):
    full = os.path.join(ROOT, dst)
    os.makedirs(full, exist_ok=True)
    for attempt in range(1, 9):
        try:
            print(f"[{time.strftime('%H:%M:%S')}] ↓ {repo} → {dst} (try {attempt})", flush=True)
            snapshot_download(repo_id=repo, repo_type=rtype, local_dir=full,
                              max_workers=8, resume_download=True)
            print(f"[{time.strftime('%H:%M:%S')}] ✅ done {repo}", flush=True)
            return True
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] ⚠ {repo} try{attempt} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
            time.sleep(min(30, 5*attempt))
    print(f"[{time.strftime('%H:%M:%S')}] ❌ GIVE UP {repo}", flush=True)
    return False

if __name__ == "__main__":
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    results = {}
    for repo, rtype, dst in JOBS:
        if only and not any(o in repo for o in only):
            continue
        results[repo] = dl(repo, rtype, dst)
    print("\n===== SUMMARY =====", flush=True)
    for r, ok in results.items():
        print(f"  {'✅' if ok else '❌'} {r}", flush=True)
