#!/usr/bin/env bash
# 16-GPU sharded eval across b0 (gpu0-7, shards 0-7) + b1 (gpu0-7, shards 8-15), then aggregate.
# Do NOT export conda LD_LIBRARY_PATH here (breaks ssh); shard.sh sets it per-process.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY LD_LIBRARY_PATH; export no_proxy='*'
SHARD=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_policy/eval/shard.sh   # scripts
D=/mnt/pfs/p46h4f/cosmos/wam_fold_policy_runs/reports                            # outputs/logs
B1="ssh -p 429 -o BatchMode=yes -o StrictHostKeyChecking=no -o ServerAliveInterval=30 root@120.48.99.93"
NUM=16
ARGS="--out_dir $D --no_lpips --max_full_windows 100000 --max_win_per_ep 4 --n_metric_eps ${NMETRIC:-20} --n_viz_eps ${NVIZ:-10}"
rm -f "$D"/shards/shard_*.json "$D"/episodes/*.mp4 "$D"/episodes/*.png "$D"/shard_*.log 2>/dev/null
mkdir -p "$D/shards" "$D/episodes"
echo "=== launch 16 shards $(date) | $ARGS ==="
# b0: gpu g -> shard g
for g in 0 1 2 3 4 5 6 7; do
  nohup bash "$SHARD" "$g" "$g" "$NUM" $ARGS > "$D/shard_${g}.log" 2>&1 &
done
# b1: gpu g -> shard 8+g
$B1 "for g in 0 1 2 3 4 5 6 7; do nohup bash $SHARD \$g \$((8+g)) $NUM $ARGS > $D/shard_b1_\$g.log 2>&1 & done; echo b1_shards_launched" 2>&1 | grep -v OpenSSL | tail -1
# wait for all 16 shard json
echo "=== waiting for 16 shard files ==="
for t in $(seq 1 240); do
  n=$(ls "$D"/shards/shard_*.json 2>/dev/null | wc -l)
  echo "[$(date +%H:%M:%S)] shards done: $n/16"
  [ "$n" -ge 16 ] && break
  sleep 60
done
echo "=== aggregate -> report.html ==="
bash "$SHARD" 0 0 "$NUM" $ARGS --aggregate > "$D/aggregate.log" 2>&1
grep -E "aggregate\]|cosmos mae" "$D/aggregate.log" | tail -3
echo "=== EVAL_16GPU_DONE ==="
