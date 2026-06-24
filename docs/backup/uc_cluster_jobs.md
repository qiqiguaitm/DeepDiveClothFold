# uc01/02/03 任务提交 + 3-Host HSDP/FSDP 集群

> uc 集群训练任务提交 — 含 gf0 SSH 管理路径, 单机 8 GPU 启动, 以及 uc01+uc02+uc03 24 GPU RDMA 3-host HSDP/FSDP 集群训练。
>
> **同 submission 子目录**:
> - `volc_ml_platform.md` — Volc Platform 基础
> - `gf0_control_plane.md` — gf0 统一控制平面
>
> **上级**: `../overview.md` / `../ssh_and_credentials.md` (uc 互信拓扑)

---

### 5.6.d gf0 经 SSH 管理 uc01/02/03 训练任务 (2026-05-21 起) ⭐

uc 集群 (uc01/02/03) **没有 volc/qzcli 这类 job 提交系统**, 训练是直接 `python scripts/train.py ...` + nohup 跑 PID。所以"任务管理"实质就是: SSH 进 uc 机器 → 启 / 杀进程 / tail 日志 / 拉 ckpt。

**gf0 → uc01/02/03 SSH 拓扑** (2026-05-21 实测):
```
gf0:~/.ssh/config 已配置 uc01, uc02, uc03 alias → 走内网直连 (公网 IP)
不需要本地跳板; gf0 ssh uc01 "<cmd>" 直接执行
```

#### 5.6.d.1 启动训练 (gf0 远程拉起 uc 任务)

```bash
# 示例: gf0 远程在 uc02 启 task_a_new_smooth_800 训练
ssh gf0 'ssh uc02 "
  cd /data/shared/ubuntu/workspace/deepdive_kai0/kai0 &&
  source .venv/bin/activate &&
  nohup python scripts/train.py pi05_flatten_fold_a_new_smooth_800_new_norm \
    --exp-name task_a_new_smooth_800_new_norm \
    --num-workers 64 \
    --overwrite \
    > /data/shared/ubuntu/logs/train_smooth_800.log 2>&1 &
  echo PID=\$!
"'

# 本地一键 helper (~/.bashrc on laptop)
alias uc-launch='ssh gf0 ssh'
# 用法: uc-launch uc02 "cd ... && nohup python ..."
```

#### 5.6.d.2 任务状态 / 监控

```bash
# 查看 uc 机器上跑的训练进程
ssh gf0 'ssh uc02 "ps aux | grep train.py | grep -v grep | awk \"{print \\\$2, \\\$10, \\\$11, \\\$12}\""'

# tail 训练日志 (在 uc 本机)
ssh gf0 'ssh uc02 "tail -F /data/shared/ubuntu/logs/train_smooth_800.log"'

# 查 GPU 利用率
ssh gf0 'ssh uc02 "nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader"'

# 杀任务 (优雅)
ssh gf0 'ssh uc02 "pkill -SIGTERM -f train.py.*smooth_800; sleep 5; pkill -9 -f train.py.*smooth_800"'
```

#### 5.6.d.3 跨集群 dashboard (gf0 集中视图)

脚本(已抽离为真实文件): **`train_scripts/dashboard_all.sh`** — 5min 轮询火山 jobs + uc01/02/03 训练进程/GPU util, 写 `logs/all_resources.txt`。

启动:
```bash
ssh gf0 'nohup bash /vePFS/tim/workspace/deepdive_kai0/train_scripts/dashboard_all.sh \
  > /tmp/dashboard.log 2>&1 &'
```

本地查看:
```bash
ssh gf0 "cat /vePFS/tim/workspace/deepdive_kai0/logs/all_resources.txt"
# 或 alias daboard='ssh gf0 cat /vePFS/.../all_resources.txt'
```

#### 5.6.d.4 数据 / Ckpt 跨集群同步 (经 gf0 中转)

uc 集群与 vePFS 不直接共享, 跨集群数据流动经 gf0 中转:

```
[uc02 数据] → gf0 (rsync 公网, 内网通 OK)
[gf0 vePFS] → 火山 robot-task vePFS (cnsh, 同 vePFS, 自动可见)
[gf0 vePFS] → TOS → gf3 vePFS-North-E (Robot-North-H20)
[火山 ckpt 在 vePFS] → 反向同上 → uc02 / sim01 部署
```

#### 5.6.d.5 控制平面命令汇总

| 操作 | 命令模板 |
|---|---|
| **火山 提交** | `ssh gf0 "cd /vePFS/.../kai0 && python train_scripts/kai/volc/submit_yaml.py <yaml>"` |
| **火山 list** | `ssh gf0 "mlp job list --state Running"` |
| **火山 stop** | `ssh gf0 "mlp job stop --id t-..."` |
| **火山 logs** | `ssh -t gf0 "mlp job logs --id t-... --follow"` |
| **uc 启动** | `ssh gf0 'ssh ucNN "<launch cmd>"'` |
| **uc 进程 list** | `ssh gf0 'ssh ucNN "ps aux \| grep train.py \| grep -v grep"'` |
| **uc 杀进程** | `ssh gf0 'ssh ucNN "pkill -SIGTERM -f train.py.*<name>"'` |
| **uc GPU 状态** | `ssh gf0 'ssh ucNN "nvidia-smi --query-gpu=... --format=csv,noheader"'` |
| **uc 日志 tail** | `ssh gf0 'ssh ucNN "tail -F /data/shared/.../logs/<exp>.log"'` |
| **跨集群 dashboard** | `ssh gf0 "cat /vePFS/.../logs/all_resources.txt"` |

#### 5.6.d.6 本地 alias 一站式包装 (`~/.bashrc` on laptop)

```bash
# 火山控制
alias vsubmit='ssh gf0 "cd /vePFS/tim/workspace/deepdive_kai0 && python train_scripts/kai/volc/submit_yaml.py"'
alias vlist='ssh gf0 "mlp job list --state Running --page-size 30"'
alias vget='ssh gf0 "mlp job get -o json --id"'
alias vstop='ssh gf0 "mlp job stop --id"'
alias vlog='ssh -t gf0 "mlp job logs --follow --instance-name worker-0 --id"'

# uc 控制 (gf0 → ucXX SSH)
uc() { ssh gf0 "ssh $1 \"${@:2}\""; }
# 用法:
#   uc uc02 "nvidia-smi -L"
#   uc uc02 "ps aux | grep train | grep -v grep"
#   uc uc02 "tail -F /data/shared/.../logs/X.log"

# 集中 dashboard
alias dashboard='ssh gf0 "cat /vePFS/tim/workspace/deepdive_kai0/logs/all_resources.txt"'
```

---


---

## 12. 3-Host HSDP/FSDP 集群训练 (uc01 + uc02 + uc03) ⭐ (2026-05-12)

**硬件**: 三台一致 — 8× A800-SXM4-80GB (NVLink 200 GB/s), 124 核 Xeon 8358P, 1.7 TB RAM, 4× Mellanox ConnectX-6 (200 Gb/s RoCEv2 each)

**关键能力**: 24 GPU 集群训练，RDMA + GPU Direct RDMA (GDR) 启用后跨主机带宽 ~800 Gb/s。

### 12.1 网络架构 (易误判)

| 网卡 | 用途 | 关键事实 |
|---|---|---|
| `eth0` (10.x.x.x) | 管理 / 公网 | virtio_net, MTU 1452, **慢，仅控制面** |
| `eth1-4` (192.168.{1-4}.x/24) | **训练通信** | **mlx5_core, 200 Gbps, RoCEv2, MTU 4200** |

**判断方法**:
```bash
# 网卡驱动 + 速率
for n in eth1 eth2 eth3 eth4; do
  echo -n "$n: "
  ethtool -i $n 2>/dev/null | grep "^driver:"
  cat /sys/class/net/$n/speed 2>/dev/null  # 应为 200000
done

# RoCE GID (NCCL 用 v2 / IPv4-mapped)
show_gids | head -20  # GID INDEX 3 = RoCEv2 IPv4 mapped

# nvidia-peermem 内核模块 (GDR 必需)
lsmod | grep nvidia_peermem
```

**主机间 IP 拓扑** (上 4 个 NIC 各独立 /24，平面无 router):
- uc01: 192.168.1.2, 192.168.2.2, 192.168.3.2, 192.168.4.2
- uc02: 192.168.1.3, 192.168.2.3, 192.168.3.3, 192.168.4.3
- uc03: 192.168.1.4, 192.168.2.4, 192.168.3.4, 192.168.4.4

### 12.2 NCCL 配置 — 必须启用 RDMA + GDR

❌ **错误（之前用过的，速度只有 ~26 Gbps × 4 NIC ≈ 100 Gbps）**:
```bash
export NCCL_IB_DISABLE=1           # ❌ 关 IB 走 TCP socket
export NCCL_NET_TYPE=Socket        # ❌
export NCCL_NET_GDR_LEVEL=0        # ❌ 关 GDR
```

✅ **正确（RDMA + GDR，~800 Gbps）**:
```bash
unset NCCL_IB_DISABLE NCCL_NET_TYPE NCCL_NET_GDR_LEVEL NCCL_NET_GDR_READ
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_IB_GID_INDEX=3              # RoCEv2 IPv4 mapped
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_P2P_LEVEL=NVL               # 节点内 NVLink P2P
export NCCL_SOCKET_IFNAME=eth1          # 控制面 bootstrap
export NCCL_DEBUG=INFO                  # 第一次跑确认 transport
# 不要手动设 NCCL_MAX_NCHANNELS / NCCL_BUFFSIZE — 让 NCCL 自适应
```

**验证 NCCL 真用 IB**（log 应出现）:
```
NET/IB : Made virtual device [0..3] name=mlx5_0..3 speed=200000 ndevs=1
Using [0]mlx5_0:1/RoCE [1]mlx5_1:1/RoCE [2]mlx5_2:1/RoCE [3]mlx5_3:1/RoCE
NET/IB : GPU Direct RDMA Enabled for HCA 0..3
Channel XX/0 : N[i] -> M[i] [send] via NET/IB/X/GDRDMA
```

### 12.3 JAX/XLA 配置

```bash
# JAX distributed (3 host)
export JAX_COORDINATOR_ADDRESS=192.168.1.2:15830  # uc01 via Mellanox eth1
export JAX_NUM_PROCESSES=3
export JAX_PROCESS_INDEX=$PROC  # 0 (uc01) / 1 (uc02) / 2 (uc03)
export JAX_ENABLE_EMPTY_ARRAYS=true   # HSDP 必需

# 持久化编译缓存 (本地, 缓存按 HLO hash 索引)
# train.py:420 已设 jax_compilation_cache_dir = ~/.cache/jax
export JAX_COMPILATION_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_COMPILATION_CACHE_MIN_COMPILE_TIME_SECS=1

# 留余地给 NCCL RDMA buffer (默认 0.95 太激进会导致 NCCL alloc 失败)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

# 不要设 XLA_FLAGS='--xla_gpu_enable_command_buffer='（空值会禁用 CUDA Graph）
# 也不要传 COLLECTIVE token (不存在，会 flag parse failed → Aborted)
unset XLA_FLAGS  # 用 XLA 默认即可
```

### 12.4 Mesh / FSDP 选择 ⚠️ 关键

| 方案 | `fsdp_devices` | mesh | 编译时间 | rate (pi05) | 备注 |
|---|---|---|---|---|---|
| HSDP `[3,8]` | 8 | `[dp=3, fsdp=8]` | **30-45 分钟首次（命中后秒级）** | **~1.0 s/it** | 节点内 sharded, 节点间 replica。命中 cache 后最快 |
| 全 FSDP `[1,24]` | 24 | `[fsdp=24]` | **5-10 分钟首次** | ~1.2 s/it | 没有 mesh 转换 → SPMD partition 简单 |
| 单机 `[8]` | 8 | `[fsdp=8]` | 5-10 分钟 | 0.5-0.7 s/it | 备选，不需要多机 |

**HSDP 巨坑 ⚠️**: SPMD partitioner 在 mesh `[24]→[3,8]T(1,0)` 转换时如果命中"Involuntary full rematerialization"慢路径，会**死锁 50-100+ 分钟**。
- 触发条件: HLO 缓存未命中（weight_loader 路径变化 / batch / 模型架构改了）
- 症状: master CPU 满载 600-800%，但 ~/.cache/jax 不写新文件、日志 30 分钟+ 不动、NCCL clique init `for 10 seconds and may be stuck`
- 解决: 切到全 FSDP (`fsdp_devices=24`)，或者重用已编译过的相同 HLO

**HLO 缓存命中条件** (全部一致才命中):
- batch_size, mesh (fsdp_devices + num_processes)
- 模型架构 (pi05 / pi0)
- `weight_loader` 路径（**包括 ckpt path 字符串**，会进 HLO closure）
- dataset action_dim
- JAX / XLA 版本

### 12.5 共享存储 (NFS on uc01)

```
uc01 /etc/exports: /data/cluster_ckpt 192.168.1.0/24(rw,sync,no_subtree_check,no_root_squash)
uc02/uc03 /etc/fstab: 192.168.1.2:/data/cluster_ckpt /cluster_ckpt nfs vers=4,hard,intr,timeo=600,rsize=1048576,wsize=1048576
```

**用途**:
- Orbax CheckpointManager 跨主机一致性 (POSIX, 必须共享)
- 数据集 (~115 GB 训练数据集放 NFS 训练时实测 GPU 99% util, NFS 没成为瓶颈)

> ⚠️ **2026-06-02 校正**: 上面 `/data/cluster_ckpt` 专用 ckpt 导出 **2026-05-18 重装后未重新 mount**(uc01/uc03 上 `/cluster_ckpt` df 为空, `/data/cluster_ckpt` 只是个本地空目录)。当前可靠的跨主机共享盘是 **workspace NFS**:`10.60.135.47:/data/shared/ubuntu/workspace` 挂在三机同一路径 `/data/shared/ubuntu/workspace`。**多机 orbax ckpt 用 `--checkpoint-base-dir /data/shared/ubuntu/workspace/multinode_ckpts`**。⚠️ 别用默认 `kai0/checkpoints`(uc 上 symlink → 节点本地 `local_ckpts`, 多机会崩, 见 §12.11 坑 9)。

**带宽**: write ~219 MB/s, read ~2 GB/s (单 stream NFSv4 over TCP)，跨 host 直传走 RoCE NIC eth1。

### 12.6 集群训练启动脚本模板

脚本(已抽离为真实文件): **`train_scripts/kai/launch/run_cluster_3host.sh`** — uc01(master/proc0)拉起 uc02/uc03 worker, 经 RDMA eth1 协调的 3-host 24-GPU 启动模板。改 `CONFIG`/`EXP_NAME` 后 `bash run_cluster_3host.sh`。

关键点(脚本内已固化, 详见上文 §12.2 NCCL / §12.3 JAX 配置):
- NCCL: `NCCL_IB_HCA=mlx5_0..3` + `NCCL_IB_GID_INDEX=3` + RDMA/GDR(不要设 `NCCL_IB_DISABLE`)。
- JAX: `JAX_COORDINATOR_ADDRESS=192.168.1.2:15830`(uc01 eth1)+ `JAX_NUM_PROCESSES=3` + 各机 `JAX_PROCESS_INDEX`。
- env: `KAI0_DATA_ROOT`、`OPENPI_DATA_HOME`、`unset XLA_FLAGS`、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.85`、`WANDB_MODE=offline`。

### 12.7 配置同步 (必做)

`config.py` 必须 3 host 一致 — worker 各自从本地读，不会自动从 master 拉:

```bash
# 改完 config.py 后:
CFG=/home/tim/workspace/deepdive_kai0/kai0/src/openpi/training/config.py
scp $CFG tim@192.168.1.3:$CFG
scp $CFG tim@192.168.1.4:$CFG
# verify
ssh tim@192.168.1.3 "grep -c '<new_config_name>' $CFG"
ssh tim@192.168.1.4 "grep -c '<new_config_name>' $CFG"
```

### 12.8 自建数据集时常见陷阱

**陷阱 A: parquet schema 不一致** (跨数据源合并)
- Task_A/base: 7 列标准 (`observation.state, action, timestamp, frame_index, episode_index, index, task_index`)
- Task_A/advantage: 12 列 (多 `progress_gt, stage_progress_gt, relative_advantage, absolute_value, absolute_advantage`)
- 合并后 `load_dataset("parquet", ...)` 会 `CastError: column names don't match`
- **修复**: 重写所有非标准 parquet 只 `select(KEEP_COLS)`

**陷阱 B: episode_index 重新编号必须改 parquet 列**
- 简单 rename 文件不够 — parquet 内 `episode_index` 和 `index` (running counter) 列必须同步重写
- mp4 可以直接 hardlink (同 `/dev/vdb` 分区秒级)

**陷阱 C: Orbax CheckpointManager metadata hash 不一致**
- 症状: `AssertionError: sync_global_devices name mismatch ('CheckpointManager:save_root_metadata') Expected: X; got: Y`
- 原因: NFS metadata stale + 上次启动残留 — 3 host `os.listdir()` 看到不同内容
- 修复: 先 `rm -rf $checkpoint_dir; sync; sleep 1` 再启动

**陷阱 D: train.py 不自动算 norm_stats**
- train.py:438 只 `shutil.copy(data.repo_id / 'norm_stats.json', ckpt_dir)`
- 必须先 `python scripts/compute_norm_states_fast.py --config-name <name>` 算好
- LeRobotDataset 在 init 时不验证 norm_stats，但 Normalize transform 会用

**陷阱 E: dataloader KeyError: 1 大量出现**
- 这是 lerobot 内部 retry 容错日志（`TransformedDataset.__getitem__` retry 50 次）
- 多数情况是 advantage 数据集 ~10% 视频缺失或 LeRobot timestamp lookup 失败
- 不是致命错误，会 skip + 重抽 — 但**会让日志极难看**，掩盖真正错误

**陷阱 F: 数据本地存储 / NFS 选择**
- 训练数据集放 NFS（uc01 export 到 uc02/uc03）实测 GPU 99% util — NFS 不是瓶颈
- 反之放 uc01 `/data/shared/...`（其他 host 没有）会导致 worker fail
- 大数据集（115GB+）合并时 hardlink mp4 + rewrite parquet → 几秒到几分钟

### 12.9 实测性能基线 (3-host 24 GPU)

| 配置 | mesh | 首次编译 | 步速 | ETA 50k |
|---|---|---:|---:|---:|
| pi05 HSDP, batch=120, fsdp=8 | `[3,8]` | 5-50 分钟* | **1.0 s/it** | 14 小时 |
| pi05 全 FSDP, batch=120, fsdp=24 | `[1,24]` | **8 分钟** | 1.2 s/it | 16.7 小时 |

\* HSDP 首次编译时长波动大: 命中缓存秒级；不命中可能 30-45 分钟，最坏死锁 50+ 分钟需要切 mesh

### 12.10 故障排查手册

| 症状 | 可能原因 | 修复 |
|---|---|---|
| 编译 30+ 分钟没出 Step, master 满载, cache 不写 | HSDP SPMD partitioner 死锁 | 切 `fsdp_devices=24` 全 FSDP |
| `Fatal: Check failed: tsl::Flags::Parse` | XLA_FLAGS 错误关键字 | `unset XLA_FLAGS` |
| `AssertionError: sync_global_devices ... CheckpointManager:save_root_metadata` | ckpt dir 残留 / NFS stale | `rm -rf $checkpoint_dir; sync` 再启 |
| `CastError: column names don't match` | parquet schema 不一致 | 重写非标准 parquet 只保 7 标准列 |
| NCCL `NET/Socket` 出现（不是 `NET/IB`） | `NCCL_IB_DISABLE=1` 错设 | unset, 改用 `NCCL_IB_HCA=mlx5_0..3` |
| `Shutdown barrier failed, 2/3 tasks reached` | 1 个 host 进程先死了 | 看那个 host 的 worker log 找根因 |
| GPU mem 满但 util 0% 长时间 | XLA 编译中 (正常) 或卡死 | check master CPU + ~/.cache/jax mtime |

### 12.11 2-host 16-GPU JAX 实战经验 (2026-06-01, A_0522_0526_raw, 8 坑全记录) ⭐

cnbj 16卡被别人任务占满 (只剩 6 卡, gang-scheduling 凑不齐 2 节点) → 迁 uc **2 节点 16 卡** (uc01 被 X3.C-100k 占, 用另外 2 个空闲节点)。一次跑通踩了 8 个坑, 逐一记录:

| # | 坑 | 现象 | 修复 / 教训 |
|---|---|---|---|
| 1 | **ssh alias↔eth1 错位** | `ssh uc02` 实际连到 eth1=.1.4 的机器 (文档 uc02=.1.3) | `~/.ssh/config` 三机 HostName 是乱的, 已修正对齐文档约定 **ucN eth1=192.168.1.{N+1}** (uc01=.1.2/uc02=.1.3/uc03=.1.4) + 注释。**操作前先 `ssh <a> "ip a show eth1"` 核对真身** |
| 2 | **JAX coordinator 时序** | proc1 起后 5min `DEADLINE_EXCEEDED` at `train.py:434` distributed.initialize | **必须 proc0(coordinator) 先起 + `ss -tlnp\|grep 15830` 确认监听, 再起 proc1**。proc1 默认 init timeout 300s, coordinator 起晚就崩。我先起 proc1 又因 SSH reset proc0 起晚 → 超时 |
| 3 | **ICMP 被滤 ≠ 不通** | `ping .1.2` 0 received, 误判网络断 | **用 TCP 测连通, 不用 ping**: python socket connect 到 `<peer>:15830` 成功 = JAX coordinator + NCCL 可用。eth1 RoCE 网 ICMP 被过滤但 TCP/RDMA 正常 |
| 4 | **init ckpt 截断** | `FAILED_PRECONDITION: Truncated Zstd-compressed stream` 读 mixed_1_clean params | TOS 同步 init 不完整 (d/chunk 2401B / ocdbt 9.2G)。**按 size 校验不按文件数**: 完整 `params/ocdbt.process_0` ≈ **22G**。重传到 22G 才对 |
| 5 | **tosutil 仅在某节点** | `/home/ubuntu/tosutil: No such file` | tosutil 只装在某一台 (本轮在 .1.3 机)。alias 改后路径机器变了。**NFS 共享 → 从有 tosutil 的机器跑同步即可** (落 NFS 全机可见) |
| 6 | **多机 orbax sync race** | `AssertionError: sync_global_devices name mismatch (CheckpointManager:save_root_metadata)` | stale ckpt metadata + 残留进程。**清 ckpt 目录 + 训练加 `--overwrite`** (非 `--resume`) |
| 7 | **残留进程干扰** | 反复重启后每机 `pgrep` 2 个 train.py | 旧崩溃/hanging 进程没杀净 (SSH 失败致 cleanup 没确认)。但 **JAX 若形成干净 `process N/2` 组就无害** (僵尸不参与协调); 真冲突才需彻底 kill |
| 8 | **uc 外网 SSH 过载** | 高频 ssh 后 `kex_exchange_identification: Connection reset` | **我连续 ssh 把外网 22 端口打过载了**。修复: ① 停手让它 backoff ② **log 在 NFS 共享, 从更稳的节点 (如 X3.C 所在机) 读同一 log**, 不必连卡死的节点 ③ 低频 + 长 sleep 间隔 |
| **9** 🔴 | **多机 ckpt 落到节点本地盘 → 第一次 save 必崩** (2026-06-02 复盘, 真正杀死该 run 的坑) | 训练到 step 2000 首个 ckpt save 时 proc1 `ValueError: [process_index=1] Timed out waiting for array_metadatas base directory creation ... timeout=600s, primary_process=0` → JAX `Shutdown barrier` → 两节点 `Fatal Python error: Aborted`, **无任何 finalized ckpt** | **根因**: `kai0/checkpoints` 在 uc 是 **symlink → `/data/shared/ubuntu/local_ckpts`**, 而 `/data`=**节点本地 `/dev/vdb`** (uc01/uc03 是两块不同物理盘)。多机 orbax 要求所有进程写**同一个共享目录**: proc0(primary, uc03)在自己本地盘建 `array_metadatas`, proc1(uc01)在自己本地盘等同名目录 → 永远等不到 → 600s 超时。**单机训练无此问题**(本地 symlink 就够), **只有多机才暴露**。**修复**: 多机训练**必须** `--checkpoint-base-dir <真正共享 NFS>`, 走 workspace NFS `/data/shared/ubuntu/workspace/multinode_ckpts`(`10.60.135.47` 导出, uc01/uc03 挂载点一致)。⚠️ §12.5 的 `/data/cluster_ckpt` 专用 ckpt NFS **2026-05-18 重装后未重新 mount**, 别用; 用 workspace NFS。 |
| **10** 🔴 | **节点间 JAX 编译缓存不对称 → 跨主机 clique init 死锁** (2026-06-02, 换节点重跑时连挂 3 次) | 跨主机 NCCL **`Connected all rings, use ring PXN 0 GDR 1` 成功** (RDMA/GDR 都通), 但随即 `E rendezvous.cc: initialize clique for rank N ... All 8 threads joined ... leader has not marked the rendezvous as completed. Leader can be deadlocked inside the rendezvous callback`, 卡死无 Step, GPU mem 已载但 util 0%。**`run_id` 跨多次重试完全相同** = 确定性, 非偶发网络 | **根因**: `train.py:420` 设 `jax_compilation_cache_dir=~/.cache/jax` (**节点本地**)。换节点重跑时, 跑过该 config 的老节点 (uc03) cache **命中**→秒进 clique, 没跑过的新节点 (uc02) cache **未命中**→还在编译, 两边到达跨主机 clique rendezvous 的时刻错开 → XLA leader 回调死锁。排除项: NCCL/RDMA/GDR 全 OK、GPU 无 Xid、/dev/shm nccl 残段清了也没用。**修复**: 多机起前 **所有节点一起清 `rm -rf ~/.cache/jax`** (对称冷编译→同步到达) — 清完第一次就过 clique 出 `Step 0`。更稳的做法: 把编译缓存放共享 NFS 让各机一致。**判据**: 见到 `Connected all rings` 后若 >1min 不出 Step 且刷 rendezvous 警告 = 中招。 |

**2 节点启动 (非 uc01 master 也行)** — 本轮 coordinator = 空闲节点之一 (eth1 .1.4), 不用必须 uc01。每节点独立从控制端 ssh 起进程 (不需节点间 ssh), 各设 `JAX_COORDINATOR_ADDRESS=<proc0 eth1>:15830` / `JAX_NUM_PROCESSES=2` / `JAX_PROCESS_INDEX=0|1` + §12.2 NCCL RDMA env + `--fsdp-devices 16` + **`--checkpoint-base-dir /data/shared/ubuntu/workspace/multinode_ckpts`(坑 9, 多机必加)**。脚本见 `train_scripts/kai/launch/run_raw_uc16.sh`(proc0 先 + 确认监听 :15830 + proc1)。

**验证稳定 (修正 2026-06-02)**: ⚠️ **`Step N: loss` 同步下降只证明前向/反向 + NCCL 通, 不证明 ckpt 能落盘**。多机真正的稳定判据是 **熬过第一次 ckpt save**(本配置 step 2000)—— 初版误判: 见到 Step100 loss 0.268→0.118 就宣布"稳定", 实际 run 在 step 2000 首个 save 因坑 9 崩了, 无可恢复 ckpt。**新判据**: 两节点 `process 0/2`+`process 1/2` + `Step N: loss` 同步 **且** `Saving checkpoint at step 2000` 后 ckpt 目录出现 finalized `2000/`(非 `*.orbax-checkpoint-tmp-*`)+ 训练继续到 step 2100+。
