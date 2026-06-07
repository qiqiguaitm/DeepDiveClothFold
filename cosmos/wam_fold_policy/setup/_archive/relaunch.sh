#!/usr/bin/env bash
pkill -9 -f "uvbin/uv sync" 2>/dev/null
pkill -9 -f "do_sync.sh" 2>/dev/null
sleep 4
# clean orphan uv tmp extraction dirs + stale locks (free space, avoid contention)
find /mnt/pfs/p46h4f/cosmos/uv_cache_root -maxdepth 1 -name ".tmp*" -exec rm -rf {} + 2>/dev/null
find /mnt/pfs/p46h4f/cosmos/uv_cache_root -name ".lock" -delete 2>/dev/null
sleep 1
echo "remaining uv: $(ps -eo cmd | grep -c '[u]vbin/uv sync')" > /mnt/pfs/p46h4f/cosmos/wam_fold_policy_runs/reports/relaunch_state.txt
nohup bash /mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_policy/setup/_archive/do_sync.sh > /mnt/pfs/p46h4f/cosmos/wam_fold_policy_runs/reports/sync_throttle.log 2>&1 &
echo "launched sync_throttle pid $!" >> /mnt/pfs/p46h4f/cosmos/wam_fold_policy_runs/reports/relaunch_state.txt
