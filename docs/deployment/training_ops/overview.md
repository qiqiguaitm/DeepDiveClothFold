# 训练服务器知识库 — Overview

> ⚠️ **uc01/02/03 已彻底停用 (2026-05-18 退役)** — 下文服务器全景/用途分工里的 uc 行仅作历史保留;现役训练机为 gf0/gf3 + Volc 集群。uc 历史归档见 [`../../backup/`](../../backup/README.md)。

> 本文档是 `training_servers_knowledge_base.md` 拆分后的"总览层"。包含服务器全景、各机用途分工、单机快速启动模板、常见运维、性能基线。
>
> **同 series 文档**:
> - `storage_and_env.md` — 文件结构 / ckpt 规范 / Python 栈 / env vars
> - `ssh_and_credentials.md` — SSH 速查 / 用户 / TOS 凭据 / uc 互信
> - `data_sync_tos.md` — TOS 枢纽 + 跨服务器数据同步 + ckpt 回流
> - `submission/volc_ml_platform.md` — 经 Volc ML Platform 提集群任务
> - `submission/gf0_control_plane.md` — gf0 作为统一控制平面
> - ~~`submission/uc_cluster_jobs.md`~~ → [`../../backup/uc_cluster_jobs.md`](../../backup/uc_cluster_jobs.md) (uc 已停用)

---

## 1. 服务器全景

> **当前 active**: gf0 + gf3 + uc01/02/03 = **5 台**。
> 三大集群: **gf 华东**(gf0, vePFS-cnsh) / **gf3 华北**(火山 ML 队列 `Robot-North-H20`, vePFS-cnbj) / **uc**(独立, lsyncd 部分镜像)。

| 维度 | **gf0** | **gf3** | **uc01** | **uc02** | **uc03** |
|---|---|---|---|---|---|
| **状态** | active | active | active | active | active |
| **GPU** | 8× A100-80GB | 1× H20-SXM5-96GB | 8× A800-80GB | 8× A800-80GB | 8× A800-80GB |
| **GPU arch** | sm_80 | sm_90 (Hopper) | sm_80 | sm_80 | sm_80 |
| **驱动 / CUDA driver** | 535.129.03 / 12.2 | 535.161.08 / 12.4 | 550.144.03 / 12.4 | 550.144.03 / 12.4 | 550.144.03 / 12.4 |
| **CUDA toolkit** | 12.8 | 12.8 | 12.4 | 12.4 | 12.4 |
| **CPU** | Xeon 8336C, 112c | 180c | Xeon 8358P, 124c | 同 uc01 | 同 uc01 |
| **RAM** | 1.8 TiB | 223 GB | ~1.7 TiB | ~1.7 TiB | ~1.7 TiB |
| **/dev/shm** | 1.3 TB | 159 GB | (待测) | (待测) | (待测) |
| **OS** | Debian-velinux1u1 | Ubuntu 22.04.5 (velinux1u2) | Ubuntu 22.04 | 同 uc01 | 同 uc01 |
| **Hostname** | `di-20260312174527-n5dw4` | `di-20260520161021-qd9b4` | `10-60-135-47` | `10-60-204-66` | (uc03) |
| **IP / 入口** | 跳板 `14.103.44.161:55555` (反向隧道) | `124.174.16.237:7888` 直连 root | `117.50.196.104` 直连 | `106.75.68.254` 直连 | `117.50.217.231` 直连 |
| **本地 SSH 别名** | `ssh -p 55555 tim@14.103.44.161` | `ssh -p 7888 root@124.174.16.237` | `uc01` (bashrc) | `uc02` (bashrc) | `uc03` (bashrc) |
| **共享 FS** | /vePFS (gpfs cnsh, 50T) | /vePFS-North-E (gpfs cnbj, 50T, **与同队列其它节点共享**) | **无** (本机独立) | **无** | **无** |
| **本机大盘** | (overlay 99G, 用 vePFS) | NVMe 3.5T (`/dev/nvme0n1`) | `/data/shared` 4TB ext4 | 同 uc01 | 同 uc01 + `/nix` 3.5T NVMe |
| **InfiniBand** | (无) | 200 Gb/s RDMA (节点间, 集群训练用) | 4× Mellanox CX-6 200 Gb/s RoCEv2 | 同 uc01 | 同 uc01 |
| **多机训练** | 单机 | 单机 / volc 集群提交多节点 (16-56 卡, §5.6.b) | uc01+02+03 HSDP/FSDP (§13) | 同 | 同 |
| **Python / venv** | 3.11 | 3.12 | 3.12 | 同 | 同 |

> **快速归类规则**:
> - **gf0 (华东)** = vePFS 共享, 长跑单机训练
> - **gf3 (华北)** = 火山 ML 队列 `Robot-North-H20` 中的一个 H20 单卡机, 适合 dev + smoke test; 真正多卡集群训练通过 volc submit job 启 2-7 个 `ml.hpcpni3ln.45xlarge` 节点 (16-56 卡)
> - **uc 集群** = 自有机房, 完全独立, 200 Gb/s RoCEv2, 3 机 HSDP/FSDP (§13)

---


---

## 5. 训练快速启动 (按机器)

### 5.1 通用启动模板 (适配任一 gf 机)

```bash
ssh tim@<host>
cd ~/workspace/deepdive_kai0/kai0   # uc01/uc02
# 或 /vePFS/tim/workspace/deepdive_kai0/kai0   # gf0

# Step 1: 计算 norm_stats (新建 dataset 时必做)
.venv/bin/python scripts/compute_norm_states_fast.py --config-name <config_name>

# Step 2: 启动训练 (JAX 全参微调)
nohup bash train_scripts/kai/launch/run_<config>_gf<N>.sh > /tmp/train_<config>.log 2>&1 &
disown $!
```

### 5.2 通用 Launcher 模板

```bash
#!/bin/bash
set -euo pipefail

export PATH=/home/tim/miniconda3/bin:/home/tim/.local/bin:$PATH
export PYTHONUNBUFFERED=1
export KAI0_DATA_ROOT=<see table 3.3>
export OPENPI_DATA_HOME=<see table 3.3>
export PYTORCH_CKPT_BASE=<see table 3.3>
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_DATASETS_CACHE=/home/tim/.cache/huggingface/datasets
export WANDB_MODE=offline
export LD_LIBRARY_PATH=...   # 见 3.3 LD_LIBRARY_PATH

cd <KAI0_DATA_ROOT>
.venv/bin/python scripts/train.py <config_name> --exp_name=<exp_name> --resume
```

### 5.3 Resume 行为

- `--resume` (推荐): 从 `<KAI0_DATA_ROOT>/checkpoints/<config>/<exp_name>/` 找最大 step 的 ckpt resume; 若无 ckpt, fallback 到 `weight_loader` 指定的 init params 冷启
- ⚠️ **永远不要用 `--overwrite`**: 该 flag rmtree 整个 exp 目录, 导致**所有 ckpt 不可逆丢失** (历史教训: 2026-04-24 误用导致 5k ckpts 全失)

### 5.4 数据集放本地加速

```bash
# Stop training first if running
# Copy to /dev/shm (tmpfs, ~3 GB/s read)
mkdir -p /dev/shm/<dataset>
cp -rL /vePFS/.../<dataset> /dev/shm/           # gf0

# Edit config.py: change repo_id to /dev/shm/<dataset>/base
# Restart training
```

**实测 gf1 v3 用 /dev/shm 后**: 步速 5.5 → 3.16 s/step (1.74× 加速), GPU util 80% idle → 100% busy。

### 5.5 自动打包 best ckpt (训练 END 后)

`train_scripts/util/auto_pack_on_end.sh` (or `/tmp/auto_pack_on_end.sh`):
- 监控训练 log 中 `[train] === END` marker
- 解析 inline-eval, 选 best step (lowest MAE@1)
- tar 打包 `params + _CHECKPOINT_METADATA + assets/` (不含 `train_state/`)

```bash
nohup bash /tmp/auto_pack_on_end.sh \
  /tmp/train_<exp>.log \
  <ckpt_root>/<config>/<exp_name> \
  <out_tar_path> \
  > /tmp/auto_pack_<exp>.run.log 2>&1 &
disown $!
```

### 5.5b gf3 单卡 smoke 启动 (Volc 集群训练前的健康验证)

```bash
ssh -p 7888 root@124.174.16.237
bash /vePFS-North-E/vis_robot/workspace/deepdive_kai0/train_scripts/kai/launch/run_gf3_smoke.sh
# log: /vePFS-North-E/vis_robot/logs/gf3_smoke_*.log
```

`run_gf3_smoke.sh` 在 H20 单卡上跑 `pi05_flatten_fold_a_new_pure_1200` config (tracked variant), 用 `A_new_pure_200` 数据集 + `pi05_base` init, FSDP=1, batch=16, `inline_eval_every=1` (eval @ save_interval=2000)。验收:看到 `Step 0` 不报错 + `Step N` loss 下降即证明环境通; 第一次 `inline_eval` (~step 2000) 给出 val MAE 即完整通。


---

## 7. 常见运维 / 故障排查

### 7.1 GPU 利用率低 (训练慢)

| 症状 | 可能原因 | 排查 / 解决 |
|---|---|---|
| GPU util 0% / 99% 周期性切换, 平均 20% | dataloader I/O 瓶颈 | 检查 `top` 看 `pt_data+` workers CPU; 数据放 `/dev/shm` |
| GPU util 99% 但步速慢 | 训练计算密集 (无瓶颈) | 正常, 不需修复 |
| 步速波动大 (3-15 s/step) | vePFS I/O 不稳 / NCCL 同步抖动 | 看 buff/cache 是否积累 (`free -h`) |

### 7.2 vePFS 满 (gf0)

vePFS 99% used (50T / ~533G 余量). 注意:
- 不要再多放训练 ckpt (每个 12-30 GB)
- 老 ckpt 主动清理 / 打包到 TOS
- 检查命令: `df -hT /vePFS`

### 7.3 训练崩溃 / GPU 占用未释放

```bash
# 找进程
pgrep -af 'pi05_flatten_fold' | head

# 优雅停止
kill -SIGTERM <pid>
sleep 10
ps -p <pid>   # 验证已停

# 强制停止 (慎用, 可能损坏 ckpt)
kill -SIGKILL <pid>

# 验证 GPU 释放
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

### 7.4 Locale warning (uc01/uc02)

每次 SSH 都会 `setlocale: LC_ALL: cannot change locale (zh_CN.UTF-8)`. 无功能影响, 可加 `export LC_ALL=C.UTF-8` 到 `~/.bashrc`。

---


---

## 9. 各机当前用途分工 (2026-05 状态)

| 机器 | 主用途 | 典型负载 |
|---|---|---|
| **gf0** | Task_A 全参 fine-tune (主战, 华东) | 50k step 长训, vePFS 数据 |
| ~~gf1~~ | 已退役 (2026-05-06) | — |
| **gf3** | dev + smoke (单卡 H20) / volc 多卡集群训练入口 (华北) | 单卡 H20 跑小 batch smoke; 集群提交 2-7 节点 16-56 卡训练 |
| **uc01** | Advantage Estimator / AWBC 训练 + 3-host HSDP/FSDP (§13) | 数据本地, 24 GPU 集群训练 |
| **uc02** | 同 uc01 (3-host 集群成员) | 同 |
| **uc03** | 同 uc01 (3-host 集群成员, 原 gf4) | 同 |

---


---

## 11. 实测性能基线 (参考)

| 配置 | 机器 | 步速 (s/step) | 备注 |
|---|---|---:|---|
| pi05 全参 fine-tune, batch=128, fsdp=8, vePFS data | gf0 | **2.0** | 基准, 数据热 cache |
| 同上 | gf1 | 5.5 | vePFS 数据冷, dataloader bound |
| 同上, data on /dev/shm | gf1 | **3.16** | 修复后, GPU 100% util |
| 同上 | uc01/uc02 | (待测) | 期望 ~2-3 s/step |
| pi05 全参 fine-tune, batch=128, fsdp=24, HSDP 3-host | uc01+02+03 | (见 §13) | NCCL+IB+GDR ~800 Gbps |
| pi05 全参 fine-tune, **batch=16, fsdp=1**, vePFS-North-E data | **gf3** (单 H20) | **2.9** | 2026-05-20 smoke: `run_gf3_smoke.sh`, JAX 0.5.3 + CUDA 12.8, Hopper sm_90 |
| pi05 全参 fine-tune, batch=128, fsdp=16, 2-host RDMA | **Robot-North-H20** 2 节点 | (待测) | 2 × `ml.hpcpni3ln.45xlarge`, 提交 `gf3_cluster_smoke_16gpu.yaml` |

inline-eval 时间 (200 frames 采样):
- 17 val ep: 660s
- 22 val ep: 850s
- 40 val ep: 1525s
- 57 val ep: 2300s
- 60 val ep: 1170s (gf0 mixed_173, val 集不同导致差异)

---


---

## 13. 修订历史

| 日期 | 内容 |
|---|---|
| 2026-05-21 | **统一控制平面 (gf0)**: §5.6.c 新增 gf0 作为火山 ML Platform (Robot-North-H20 + robot-task) + uc01/02/03 唯一管理入口 + §5.6.d gf0→uc SSH 远程控制详细; §6 整体重构为 TOS 为中心枢纽架构 (sim01 = source, TOS = exchange, gf0/gf3/uc = consumers); §2.3/§2.4 修正重复 section number; §13 与 §12 顺序调换 (修订历史置末尾) |
| 2026-05-20 | **新增 gf3 (火山华北 H20 单卡)**: §1 表格扩到 5 台; §2/§3/§4 增 gf3 行/列; §5.5b 单卡 launcher (`run_gf3_smoke.sh`) + §5.6.b cn-beijing 16 卡 yaml (`gf3_cluster_smoke_16gpu.yaml`); §11 加 gf3 单卡 2.9 s/it 基线; `submit_yaml.py` 加 `Robot-North-H20` queue 映射 (cn-beijing, q-20260516104642-khch9, cn-beijing-e); `setup_env.sh` 加 `profile=gf3`. gf3 venv 通过 uc01 → TOS → gf3 + path 重写方案 (GitHub HTTPS 跨 region 失败) |
| 2026-05-20 | **删除 js01-04 服务器全部条目** (集群停用); §1/§2/§3/§4/§5/§6/§7/§9/§11 表格还原为 4 台 (gf0+uc01/02/03); §14 (js 集群章节) 整体移除; §6.4/§6.5 (js 内部 + 跨集群同步) 整体移除 |
| 2026-05-18 | uc01/02/03 重装 (mining 入侵), 改用 ubuntu 账户; gf1 条目从文档删除 (已退役) |
| 2026-05-13 | (已并入 5-20 移除) §2/§3/§4/§5/§6/§7/§9/§11 全部扩展加入 js 集群行; §6.4/§6.5 新增 js 内部 + 跨集群同步; §7.5 新增 JuiceFS 元数据延迟坑 |
| 2026-05-12 | (已并入 5-20 移除) 添加 §14: js01-04 集群; §1 全景表扩到 8 台 |
| 2026-05-12 | 添加 section 13: 3-host HSDP/FSDP 集群训练 + RDMA + GDR + NCCL 配置 + 坑 |
| 2026-05-02 | 初版: 整合 gf0/gf1/uc01/uc02, 含 v3 /dev/shm 加速实测 |

后续更新: 添加 uc01/uc02 实际训练性能基线 / sim01 ↔ gf 集群网络拓扑细节。

---

