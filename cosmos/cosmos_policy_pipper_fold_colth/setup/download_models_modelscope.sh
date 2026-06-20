#!/usr/bin/env bash
# Download Cosmos checkpoints from ModelScope (nv-community mirror) WITHOUT proxy.
# Per user guidance: ModelScope-first, proxy-bypassed; HF-via-proxy only as fallback.
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; export no_proxy='*'
RUNS=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/cosmos_policy_pipper_fold_colth_runs
MD="$RUNS/models"; mkdir -p "$MD"
L="$RUNS/logs/ms_download.log"
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$L"; }

# ModelScope HF-compatible resolve URL
ms_url(){ echo "https://modelscope.cn/models/$1/resolve/master/$2"; }

get(){ # repo  filepath  outdir
  local repo="$1" fp="$2" out="$3"; mkdir -p "$out"
  local f="$out/$(basename "$fp")"
  log "get $repo/$fp"
  while true; do
    wget -c -q --timeout=180 --tries=5 "$(ms_url "$repo" "$fp")" -O "$f"
    local rc=$?
    local sz; sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
    log "  $(basename "$fp"): $((sz/1048576))MB rc=$rc"
    [ "$rc" -eq 0 ] && { log "  done $(basename "$fp")"; return 0; }
    log "  retry in 20s"; sleep 20
  done
}

R_POL=nv-community/Cosmos-Policy-ALOHA-Predict2-2B
R_BASE=nv-community/Cosmos-Predict2-2B-Video2World
R_TOK=nv-community/Cosmos-Predict2.5-2B

# 1) warm-start policy ckpt + its stats + aloha t5 embeddings
get "$R_POL" "Cosmos-Policy-ALOHA-Predict2-2B.pt"   "$MD/Cosmos-Policy-ALOHA-Predict2-2B"
get "$R_POL" "aloha_dataset_statistics.json"        "$MD/Cosmos-Policy-ALOHA-Predict2-2B"
get "$R_POL" "aloha_t5_embeddings.pkl"              "$MD/Cosmos-Policy-ALOHA-Predict2-2B"
get "$R_POL" "config.json"                          "$MD/Cosmos-Policy-ALOHA-Predict2-2B"
# 2) base WFM (referenced at config import time)
get "$R_BASE" "model-480p-16fps.pt"                 "$MD/Cosmos-Predict2-2B-Video2World"
get "$R_BASE" "config.json"                         "$MD/Cosmos-Predict2-2B-Video2World"
get "$R_BASE" "model_index.json"                    "$MD/Cosmos-Predict2-2B-Video2World"
# 3) Wan VAE tokenizer
get "$R_TOK"  "tokenizer.pth"                        "$MD/Cosmos-Predict2.5-2B"

log "ALL MODELSCOPE DOWNLOADS DONE"
ls -la "$MD"/*/ | tee -a "$L"
