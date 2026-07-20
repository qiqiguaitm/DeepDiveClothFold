import os, time
for k in ["http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","all_proxy"]:
    os.environ.pop(k, None)
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
from huggingface_hub import snapshot_download
for a in range(1,10):
    try:
        print(f"try {a}", flush=True)
        snapshot_download('jialei02/libero_merged_no_noops_20hz', repo_type='dataset',
                          local_dir='dataset/libero_merged_no_noops_20hz', max_workers=8)
        print("✅ LIBERO dataset done", flush=True); break
    except Exception as e:
        print(f"try{a} fail: {type(e).__name__}: {str(e)[:150]}", flush=True); time.sleep(min(30,5*a))
