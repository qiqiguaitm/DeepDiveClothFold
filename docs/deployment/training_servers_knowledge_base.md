# 训练服务器知识库 (gf0 / gf3 / uc01 / uc02 / uc03)

> ⚠️ **2026-05-08 更新: gf2/3/4 → uc01/02/03 重命名; uc03 (原 gf4) 加入训练**
> ⚠️ **2026-05-11 更新: 日期 leaf 命名统一为 `YYYY-MM-DD-v2` (见 §2.3 末)**
> 🔴 **2026-05-18 更新: uc01/02/03 因挖矿木马入侵全系统重装** (详见 `docs/security/2026-05-16_rvn_miner_incident.md`)。**uc 集群不再创建 `tim` 用户, 改用 `ubuntu` 用户作为开发账户**。SSH 用 `ssh ubuntu@<IP>` (本地 alias `uc01/02/03` 已改 User=ubuntu)。3 台间内网 SSH 互信已配 (见 §4.4)。**gf / sim 集群 仍用 tim 账户, 不受影响**。
> 🗑️ **2026-05-18 更新: gf1 (2026-05-06 退役) 已从本文档彻底移除条目**。历史 ckpt @ /vePFS 仍可经 gf0 访问 (vePFS 仍是 gf0/gf1 共享盘)。具体 gf1 #25 best ckpt (MAE@1=0.0104 task_a_new_pure_1200_new_norm step 38000) 已通过 TOS 拉到 sim01 `/data1/DATA_IMP/checkpoints/task_a_new_pure_1200_new_norm_best_step38000/`。
> 🗑️ **2026-05-20 更新: js01-04 服务器全部停用, 不再启用; 相关章节 / 条目从本文档移除**。
> 🆕 **2026-05-20 更新: 新增 gf3 (火山华北 H20 单卡机器, `124.174.16.237:7888`)**。属于火山 ML Platform `Robot-North-H20` 队列 (`ml.hpcpni3ln.45xlarge` × 7, 共 56 H20 GPUs)。本机 gf3 是其中 1 节点 (单卡); 集群训练 (16/24/32 卡) 通过 volc 队列提交多节点 job (见 §5.6.b)。vePFS 是 `vepfs-cnbj875793a96d6b` (华北 region, 与 gf0 的华东 vePFS 隔离), 挂载到 `/vePFS-North-E/vis_robot/`。
>
> **当前 active 服务器: gf0, gf3, uc01, uc02, uc03** (5 台)。



> **作用**: 4 台 GPU 训练服务器的全方位参考 — 硬件、文件结构、环境、连接方式、训练命令、机器间差异、常见运维。
> **更新日期**: 2026-05-20
> **关联文档**:
> - [`gf2_gf3_deployment.md`](./gf2_gf3_deployment.md) — uc01/uc02 详细部署记录
> - [`sim01_deployment.md`](./sim01_deployment.md) — sim01 推理机部署
> - [`checkpoints_layout.md`](./checkpoints_layout.md) — ckpt 文件结构规范

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

## 2. 文件结构

### 2.1 工作目录路径速查

| 服务器 | 工作目录 | 实际存储 |
|---|---|---|
| gf0 | `/vePFS/tim/workspace/deepdive_kai0/` (= `/home/tim/workspace/deepdive_kai0` 软链) | gpfs cnsh 跨机共享 |
| gf3 | `/vePFS-North-E/vis_robot/workspace/deepdive_kai0/` | gpfs cnbj, 与同队列其它节点共享 |
| uc01 | `/home/ubuntu/workspace/deepdive_kai0/` → `/data/shared/ubuntu/workspace/deepdive_kai0/` (2026-05-18 后) | 本机 4TB ext4 |
| uc02 | 同 uc01 (各自独立, 不共享) | 同 uc01 |
| uc03 | 同 uc01 (各自独立) | 同 uc01 + 本机 `/nix` 3.5T NVMe |

### 2.2 Checkpoint 本地存储规范 ⭐ (2026-05-04 重要更新)

> **核心原则**: 每台服务器的 ckpt 写到独立的本地路径, 不跨机同步, 重启不丢失。

**统一路径**: 每台机器都使用 `/home/tim/local_ckpts/` 作为 ckpt 根目录 (其中是 symlink 还是 real dir 因机器而异)。

| Server | `/home/tim/local_ckpts/` 实现 | 物理后端 | 容量 | 持久性 |
|---|---|---|---|---|
| gf0 | symlink → `/vePFS/tim/gf0_local_ckpts/` | /vePFS (50T 共享 FS) | 看 /vePFS 余量 | ✓ 持久 |
| uc01 | 真实 dir | /dev/vda2 (492G ext4) | ~290G 可用 | ✓ 持久 |
| uc02 | 真实 dir | /dev/vda2 (492G ext4) | ~410G 可用 | ✓ 持久 |
| uc03 | 真实 dir | /dev/vda2 (492G ext4) | (待测) | ✓ 持久 |

**为何不放 `/dev/shm` (RAM)**:
- 重启数据丢失, 训练 ckpt 不能容忍
- /dev/shm 适合 dataset (可从源重建), 不适合 ckpt (训练成果)

**为何 gf0 没用 `/home/tim` 真实 dir**:
- gf0 上 `/home/tim` 在 overlay (~99G, 已 95% 用) — 没空间存 ckpt
- 唯一持久 + 大容量选项是 `/vePFS` (slow but persistent)
- 所以统一用 `/home/tim/local_ckpts` (symlink) → /vePFS 子目录

**怎么让训练写到 local_ckpts**:

openpi 默认把 ckpt 写到 `<KAI0_DATA_ROOT>/checkpoints/<config>/<exp>/`。我们用 **per-exp 软连接**, 在 launcher 启动训练前 pre-create 链接:

```bash
# 在 launcher 里:
CONFIG=pi05_flatten_fold_<your_config>
EXP=<your_exp_name>
LOCAL_DIR=/home/tim/local_ckpts/$CONFIG/$EXP
WORKSPACE_DIR=$KAI0_DATA_ROOT/checkpoints/$CONFIG/$EXP

mkdir -p "$LOCAL_DIR"
mkdir -p "$(dirname "$WORKSPACE_DIR")"
[ -e "$WORKSPACE_DIR" ] && [ ! -L "$WORKSPACE_DIR" ] && {
    echo "WARN: $WORKSPACE_DIR exists as real dir, please move first"
    exit 1
}
ln -sfn "$LOCAL_DIR" "$WORKSPACE_DIR"

# 然后正常启训练:
.venv/bin/python scripts/train.py $CONFIG --exp_name=$EXP --resume
```

`ln -sfn` (`-n` = no-deref existing symlink) 确保 idempotent, 重复 launcher 启动不出错。

**lsyncd 兼容性 (uc01/uc02)**:
- uc01/uc02 之间有 lsyncd 双向 mirror `/data/shared/` 目录
- `/home/tim/local_ckpts` 在 `/dev/vda2` 不在 lsyncd scope, 不会被同步 ✓
- 而 `/home/tim/workspace` 是 symlink → `/data/shared/...` 在 lsyncd 范围, 千万 **不要直接写 ckpt 到** `<kai0>/checkpoints/<config>/<exp>` 真实目录 (旧 bug 多次因此损坏)

**keep_period 设置**:
- 100k step 训练: `keep_period=10000` (保留 10 个) 比 `2_000` (保留 50 个) 减少 5× 占用
- 50k step: `keep_period=10000` (保留 5 个) 大约 165GB; 默认 `2_000` 时 825GB 可能撑爆 /dev/vda2

**已知 ckpt 路径**:

| 实验 | 当前 ckpt 真实路径 | 所有者 |
|---|---|---|
| uc01 实验1 | `/home/tim/local_ckpts/pi05_flatten_fold_mix_b6000_p1200_init_mixed_1/task_a_mix_base6000_pure1200_new_norm_base_mixed_1` | uc01 |
| uc02 实验2 | `/home/tim/local_ckpts/pi05_flatten_fold_mix_b6000_p1200_init_pi05_base/task_a_mix_base6000_pure1200_new_norm_base_pi0.5` | uc02 |
| gf0 实验3 | `/vePFS/tim/gf0_local_ckpts/pi05_flatten_fold_mix_b6000_p1200_init_pi05_base_100k/task_a_mix_base6000_pure1200_new_norm_base_pi0.5_100000` | gf0 |

### 2.3 数据集 / Checkpoint 目录约定 (传统 view)

```
deepdive_kai0/
├── kai0/                              # 主代码 (uv venv at .venv/)
│   ├── .venv/                         # Python 3.11/3.12 (uv 管理)
│   ├── src/openpi/                    # openpi 主代码
│   ├── scripts/                       # train.py / compute_norm_states_fast.py / ...
│   ├── checkpoints/                   # 训练 ckpt 落地
│   │   ├── Task_A/mixed_1/            # MA-merged init 模型 (paper-grade)
│   │   │   ├── _CHECKPOINT_METADATA
│   │   │   ├── norm_stats.json
│   │   │   └── params/                # ~12 GB JAX/Flax 权重
│   │   └── pi05_flatten_fold_*/<exp_name>/  # 各训练 exp 的 ckpts
│   └── data/                          # 数据集软链入口
│       └── Task_A/
│           ├── vis_base/              # → 真实/模拟采集数据集
│           ├── kai0_base/             # → HF 官方 kai0 base
│           ├── kai0_dagger/           # → HF 官方 kai0 dagger
│           ├── kai0_advantage/        # → HF 官方 advantage (uc01/uc02 only)
│           └── self_built/            # 用户构建的混合数据集
│               ├── A_pure_1200/{base,val}/
│               ├── A_new_pure_1200/{base,val}/
│               ├── mix_apr28_450/{base,val}/
│               └── ...
├── train_scripts/                     # 训练 launcher / 数据脚本
│   ├── data/
│   │   ├── build_task_a_*.py          # 数据集构建脚本
│   │   └── compute_delta_norm_stats_fast.py
│   └── launch/
│       ├── run_*_gf0.sh
│       └── run_gf2.sh / run_gf2_adv_est.sh
├── docs/                              # 文档
├── setup_env.sh                       # KAI0_DATA_ROOT / OPENPI_DATA_HOME 自动配置
└── install.sh                         # 一键安装环境
```

### 2.4 数据集源 (按机器)

#### gf0 (共享 vePFS 华东)
```
/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/
  base/                # 自建 (来自 visrobot01)
  dagger/              # 自建
  vis_base/<date>/     # 按日期分子集 (~310-644 ep)
  kai0_base/, kai0_dagger/
  self_built/A_pure_1200, A_new_pure_1200, mix_apr28_450, ...

/vePFS/visrobot01/KAI0/Task_A/base/<date>/  # 原始采集 (跨用户共享)
```

#### gf3 (共享 vePFS 华北)
```
/vePFS-North-E/vis_robot/dataset/KAI0/Task_<X>/                      # 数据集 (从 TOS 同步)
/vePFS-North-E/vis_robot/base_init_ckpts/extracted/pi05_base/params/ # init weights (从 TOS pi05_base.tar 解压)
/vePFS-North-E/vis_robot/checkpoints/<config>/<exp>/                 # 训练输出
/vePFS-North-E/vis_robot/logs/                                       # 训练日志
/vePFS-North-E/vis_robot/workspace/deepdive_kai0/                    # 代码 (从 gf0 scp tarball)
/vePFS-North-E/vis_robot/workspace/.uv_python/                       # uv-managed Python (self-contained)
/vePFS-North-E/vis_robot/venv/                                       # 原始 venv.tar / uvpython.tar 缓存
```

> **跨 region 同步**: gf3 (cn-beijing) 不能直连 gf0/uc01/sim01 (cn-shanghai), 一切通过 TOS `tos://transfer-shanghai/...` 中转 (跨 region 走 TOS 后端骨干)。pi05_base.tar (12.3G) + 数据子集 ~17G 总同步 ≈ 4-6 分钟。

#### uc01 / uc02 / uc03 (独立 4TB ext4)
```
/data/shared/dataset/KAI0/Task_<X>/base/         # 自建 (rsync from /vePFS)
/data/shared/dataset/Kai0_official/Task_A/      # HF 官方 base/dagger/advantage
~/workspace/deepdive_kai0/kai0/data/Task_<X>/   # symlinks 指向上述路径
```

#### 日期 leaf 命名约定: `YYYY-MM-DD-v2` (2026-05-11 起)

历史上 `base/` 下日期 leaf 直接是 `YYYY-MM-DD`. 4-23 ~ 4-30 的数据被处理后另存为 `YYYY-MM-DD-v2`. **2026-05-11 起统一**: 所有新采集直接写 `YYYY-MM-DD-v2`, 不再区分"原始"与"处理后".

- **写入**: `web/data_manager/backend/app/layout.py:new_task_subset_root()` 给今日日期附加 `-v2`. 受影响调用方: `recorder.py::start_recording` (web UI 采集).
- **读取**: `_DATE_RE = r"^\d{4}-\d{2}-\d{2}(?:-v\d+)?$"` 同时匹配两种, `path_to_compound()` 把 `-v2` 保留在 task_id 中 (e.g. `Task_A_2026-05-11-v2`).
- **历史**: 2026-05-11 一次性把 5-06 ~ 5-09 (sim01 / TOS / uc01-uc03 共 4 端) 全部 `mv old → old-v2`. 期间 uc02/uc03 因 lsyncd `--update` 不删旧, 手动 rm 残留. Task_PP/5-09 当时仅 sim01 有, 下次 sync 直接以 -v2 上 TOS.

### 2.5 临时 / 加速存储 (按机器)

| 路径 | gf0 | gf3 | uc01/uc02/uc03 |
|---|---|---|---|
| `/dev/shm` (tmpfs RAM) | **1.3 TB** ⭐ 训练数据可加速 | 159 GB | 大 (具体大小待测) |
| `/tmp` | overlay ~99GB | overlay ~100GB | overlay ~99GB |
| 本机 NVMe | (无独立) | **3.5 TB** (`/dev/nvme0n1`) | uc03: `/nix` 3.5T NVMe |
| 跨机/跨节点共享 | `/vePFS` 50T gpfs cnsh | `/vePFS-North-E` 50T gpfs cnbj | (无) |
| TOS (cn-shanghai) | tosutil/rclone (本地有 AK/SK) | tosutil (复用 cnsh AK/SK, 跨 region 走骨干) | tosutil/rclone |

---

## 3. 环境 (Python 栈)

### 3.1 venv 路径

| 机器 | venv 路径 | Python |
|---|---|---|
| gf0 | `/vePFS/tim/workspace/deepdive_kai0/kai0/.venv` → `/home/tim/.kai0_venv` (本地 symlink) | 3.11 |
| gf3 | `/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/.venv` (**self-contained on vePFS**, 跨节点共享) | 3.12.13 |
| uc01 | `/home/tim/workspace/deepdive_kai0/kai0/.venv` (uv 管理, 真实 dir) | 3.12 |
| uc02 | 同 uc01 (本地独立) | 3.12 |
| uc03 | 同 uc01 (本地独立) | 3.12 |

> **注意 (gf0 / gf3 区别)**:
> - **gf0**: vePFS 上的 `.venv` 是 symlink, 真实 venv 在本机 `/home/tim/.kai0_venv` (不跨机)
> - **gf3**: `.venv` 完全 self-contained 在 vePFS 上 — `python` 二进制 + uv-managed Python tree 都在 `/vePFS-North-E/vis_robot/workspace/.uv_python/cpython-3.12.13-linux-x86_64-gnu/` 下, `pyvenv.cfg home =` 也指 vePFS 路径。这样 volc 集群任意新节点 mount vePFS-North-E 后 `source .venv/bin/activate` 即可直接用, **无需在每节点重装**。
>
> **gf3 venv 构建路径 (2026-05-20)**: 由于 GitHub HTTPS 在 cn-beijing 跨 region 极不稳 (lerobot git fetch 反复 TLS stream cancel), 直接在 gf3 跑 `uv sync` 失败。改为: ① uc01 上 `tar` 现成 `.venv` (8.2 GB) + uv-managed Python (104 MB) 上传到 TOS `from_uc01/gf3/`; ② gf3 拉取后解压 + sed 重写 hardcoded 路径 (`/data/shared/ubuntu/workspace/deepdive_kai0/kai0` → `/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0`, `/home/ubuntu/.local/share/uv` → `/root/.local/share/uv`); ③ 后续再把 uv-managed Python 也搬到 vePFS, .venv 重 symlink. 一键脚本: `/root/gf3_install_venv.sh` (副本: `train_scripts/launch/gf3_install_venv.sh`)。**全过程 6 分半**。

### 3.2 关键依赖 (各机基本一致)

- **JAX** 0.5.3 + cuda12 (含 GPU)
- **PyTorch** 2.7.1+cu126 (uc01/uc02) / 与之兼容版本 (gf0)
- **Flax** 0.10.2 / orbax-checkpoint 0.11.13
- **openpi** (editable, in `kai0/src/openpi/`)
- **lerobot** (HF 库) / transformers / sentencepiece
- **tos** 2.9.0 (Volcengine, 用于 TOS 文件传输)

### 3.3 环境变量 (`setup_env.sh` 自动设置)

| 变量 | gf0 (`profile=gf`) | gf3 (`profile=gf3`) | uc01/02/03 (`profile=default`) |
|---|---|---|---|
| `KAI0_DATA_ROOT` | `/vePFS/tim/workspace/deepdive_kai0/kai0` | `/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0` | `$HOME/workspace/deepdive_kai0/kai0` |
| `OPENPI_DATA_HOME` | `/vePFS/tim/workspace/openpi_cache` | `/vePFS-North-E/vis_robot/openpi_cache` | `$HOME/.cache/openpi` |
| `PYTORCH_CKPT_BASE` | `/vePFS/tim/workspace/openpi_cache/modelscope_cache/lerobot` | `/vePFS-North-E/vis_robot/openpi_cache/modelscope_cache/lerobot` | `$HOME/.cache/openpi/modelscope_cache/lerobot` |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | 0.9 (set per-launcher) | 0.85-0.9 (单卡 0.9, 集群 0.85 留 NCCL buffer) | 同 |
| `WANDB_MODE` | `offline` (无外网) | `offline` | `offline` |
| `LD_LIBRARY_PATH` | 含 `/usr/local/cuda-12.8/...` + `/home/tim/.cuda_compat` | 由 venv 内 `nvidia/*/lib` 提供 (launcher 自动 append) | 含 `/usr/local/cuda-12.4/...` |
| `TORCH_CUDA_ARCH_LIST` | (default) | `"9.0"` (Hopper) | `"8.0"` (设在 `~/.bashrc`) |

> **gf3 profile 识别**: `setup_env.sh` 通过 `[[ -d /vePFS-North-E/vis_robot ]]` 探测 (火山华北节点 hostname 形如 `di-YYYYMMDDHHMMSS-xxxxx`, 不固定, 用文件系统探测更稳)。

### 3.4 已知的机器特定 workaround

| 现象 | 解决 |
|---|---|
| gf0 vePFS (历史与 gf1 共享, gf1 已退役) | 在 gf0 单机操作 |
| uc01/uc02 HF 下载 429 限流 | 单机优先 + retry, 然后 rsync 到另一机 |

---

## 4. 连接方式 / 用户信息

### 4.1 SSH 速查

```bash
# gf0 (从 sim01 / 任意公网机)
ssh -p 55555 tim@14.103.44.161   # gf0 (反向隧道经 14.103.44.161 跳板)

# gf3 (火山华北 H20 单卡机, root 直连)
ssh -p 7888 root@124.174.16.237  # gf3, 密码 tim (建议改 key-based)

# uc01 / uc02 / uc03 (2026-05-18 重装后, 直连, ubuntu 账户 key-based)
ssh ubuntu@117.50.196.104   # uc01
ssh ubuntu@106.75.68.254    # uc02
ssh ubuntu@117.50.217.231   # uc03
# (旧: sshpass -p tim ssh tim@... — 已废弃, tim 用户在 uc 上不存在)

# 也可在 ~/.bashrc 设别名:
alias gf3='ssh -p 7888 root@124.174.16.237'
alias uc01='ssh ubuntu@117.50.196.104'   # 2026-05-18 后, key-based, 无需密码
alias uc02='ssh ubuntu@106.75.68.254'
alias uc03='ssh ubuntu@117.50.217.231'
```

### 4.2 用户

- **gf0/sim01**: 用户名 `tim`, 密码 `tim` (有密码 sudo)
- **gf3** (火山华北 H20): 用户名 **`root`**, 密码 `tim`。`/root/code/{README*,demo_project}` 是火山初始 demo, 我们的项目在 `/vePFS-North-E/vis_robot/` 下
- **uc01/02/03** (2026-05-18 重装后): 用户名 **`ubuntu`** (不再创建 tim), key-based 登录, 强密码已设
  - cloud-init pre-seed 了本地 dev pubkey + 团队 key (yihaochen / qiqiguaitm / tim@ipc01 等) 到 `/home/ubuntu/.ssh/authorized_keys`
  - 3 台 uc 间 ubuntu 用户 ed25519 互信已配 (详见 §4.4)
  - **⚠️ 重要安全**: 重装后应立刻**禁 SSH 密码登录** (`PasswordAuthentication no` in `/etc/ssh/sshd_config`) 避免被爆破 (上次事件 2026-05-15 即由此引发, 见 `docs/security/2026-05-16_rvn_miner_incident.md`)
- gf0: 反向隧道无密码 key-based

### 4.3 TOS 凭据 / Bucket

- Bucket: `transfer-shanghai` @ `tos-cn-shanghai.volces.com` (region `cn-shanghai`)
- 读凭据: hardcoded 在 `train_scripts/data/from_tos_file.py` (公开)
- 写凭据: `VOLC_TOS_AK` / `VOLC_TOS_SK` env vars 或 `tosutil` 配置

> **完整 TOS 数据同步架构 (sim01 是源 → TOS 枢纽 → 各训练服务器) 见 §6**。本节仅记录凭据/bucket 信息。

### 4.4 uc 集群 SSH 互信拓扑 (2026-05-18 重装后)

**3 台 uc 间 ubuntu 用户 ed25519 互信** (cloud-init 已 pre-seed):

```
                  ┌────────────┐
                  │  本地 dev   │  (id_rsa, qiqiguaitm@sina.com 等)
                  │   tim@*     │
                  └─────┬──────┘
                        │ pubkey 进 3 server ubuntu authorized_keys
                        ▼
        ┌───────────────────────────────┐
        │  uc01 ubuntu@10-60-135-47     │
        │  uc02 ubuntu@10-60-204-66     │  彼此 ed25519 互信 (6 个方向已通)
        │  uc03 ubuntu@10-60-253-225    │
        └───────────────────────────────┘
```

**各 host ubuntu ed25519 pubkey** (2026-05-18 验证):

| Host | Pubkey (前缀 + comment) |
|---|---|
| uc01 | `AAAAC3NzaC1lZDI1NTE5AAAAIF+mEiKsU8Q2fiXWl9fG/6J+THe9+vMZKjvICm0srfLb ubuntu@10-60-135-47` |
| uc02 | `AAAAC3NzaC1lZDI1NTE5AAAAIPOYAi7KHrboT1M1AVXiulnVlyzAmJAa3HKzXaNDfc0n ubuntu@10-60-204-66` |
| uc03 | `AAAAC3NzaC1lZDI1NTE5AAAAILQdFOvow28O9HalNIPUCElD/im+FHxQCiP9N2yVtWYD ubuntu@10-60-253-225` |

**测试命令**:
```bash
ssh uc01 hostname                              # 本地 → uc01 (key-based)
ssh uc01 'ssh ubuntu@10.60.204.66 hostname'    # uc01 → uc02 (内网)
```

**与 gf / sim 集群隔离**: uc 的 ubuntu 用户 SSH key **未推到** gf/sim。跨集群 SSH 仍走 tim 用户旧互信。本地 dev 是唯一同时拥有 tim (gf/sim) + ubuntu (uc) 访问的入口。

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
nohup bash train_scripts/launch/run_<config>_gf<N>.sh > /tmp/train_<config>.log 2>&1 &
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
bash /vePFS-North-E/vis_robot/workspace/deepdive_kai0/train_scripts/launch/run_gf3_smoke.sh
# log: /vePFS-North-E/vis_robot/logs/gf3_smoke_*.log
```

`run_gf3_smoke.sh` 在 H20 单卡上跑 `pi05_flatten_fold_a_new_pure_1200` config (tracked variant), 用 `A_new_pure_200` 数据集 + `pi05_base` init, FSDP=1, batch=16, `inline_eval_every=1` (eval @ save_interval=2000)。验收:看到 `Step 0` 不报错 + `Step N` loss 下降即证明环境通; 第一次 `inline_eval` (~step 2000) 给出 val MAE 即完整通。

### 5.6 Volc ML Platform 提交基础 (YAML + SDK 模式)

> ℹ️ **当前推荐工作流见 §5.6.c (gf0 统一管理)**。本节仅记录 YAML 格式 + Python SDK monkey-patch 等底层细节, 用于自定义脚本开发。日常提交直接用 §5.6.c 的 `vsubmit` alias 即可。

Volc 火山引擎 ML Platform 提供按量付费 H20/A100 节点（机房代号 `cn-shanghai` 与 `cn-beijing`）。任务通过 OpenAPI 提交，代码 + 数据走挂载的 vePFS。

**前置：**
- vePFS workspace 已有最新代码（`/vePFS/tim/workspace/deepdive_kai0/...`），volc 节点 boot 时挂载 `MountPath: /vePFS`
- 凭证：`VOLC_AK` / `VOLC_SK` 写入 `~/.volc_creds`（mode 0600）+ `~/.bashrc` 加 source guard
- SDK：`volcengine-python-sdk`（uv pip install），SDK 5.0.27 有 deserializer KeyError bug，必须 monkey-patch

**YAML 模板**（`xvla/scripts/*.yaml`）：

```yaml
TaskName: "xvla-stage1-kai-warmup-16gpu"
ImageUrl: "visincept-cn-shanghai.cr.volces.com/grasp/h2r:1.0"
ResourceQueueName: "robot-task"          # → q-20251204185107-fvnpx (A100 80G)
Framework: "PyTorch"
TaskRoleSpecs:
  - RoleName: "worker"
    RoleReplicas: 2                       # 节点数 (16 GPU = 2×8)
    Flavor: "ml.hpcpni2.28xlarge"        # A100×8 + RDMA
ActiveDeadlineSeconds: 172800             # 48h hard timeout
Storages:
  - Type: "Vepfs"
    VepfsId: "vepfs-cnsh075262e1f815"
    MountPath: "/vePFS"
CacheType: "Cloudfs"
Envs:
  - {Name: HF_HUB_OFFLINE, Value: "1"}
  - {Name: NCCL_DEBUG, Value: "WARN"}
  - {Name: XLA_PYTHON_CLIENT_MEM_FRACTION, Value: "0.85"}
  - {Name: JAX_ENABLE_EMPTY_ARRAYS, Value: "true"}
Entrypoint: |
  exec >> /vePFS/.../logs/$(date -u +%Y%m%d_%H%M%S)_node${MLP_ROLE_INDEX:-0}.log 2>&1
  if ! ldconfig -p | grep -q libavutil; then apt-get install -y -qq ffmpeg; fi
  cd /vePFS/tim/workspace/deepdive_kai0/kai0
  source .venv/bin/activate
  export JAX_COORDINATOR_ADDRESS="${MLP_WORKER_0_HOST}:15830"
  export JAX_NUM_PROCESSES="${MLP_WORKER_NUM:-2}"
  export JAX_PROCESS_INDEX="${MLP_ROLE_INDEX:-0}"
  exec python -u scripts/train.py <config_name> --exp-name <exp_name> --no-wandb-enabled --overwrite
```

**提交 (绕开 SDK 反序列化 bug)：**

```python
import os, json, yaml
import volcenginesdkcore
from volcenginesdkmlplatform20240701.api.ml_platform20240701_api import MLPLATFORM20240701Api
import volcenginesdkcore.interceptor.interceptors.deserialized_response_interceptor as drm

# Monkey-patch broken deserializer (SDK 5.0.27 KeyError: '.models')
def safe_intercept(self, ctx):
    if ctx.request.preload_content:
        try: ctx.response.result = json.loads(ctx.response.http_response.data)
        except: ctx.response.result = {}
    return ctx
drm.DeserializedResponseInterceptor.intercept = safe_intercept

cfg = volcenginesdkcore.Configuration()
cfg.ak, cfg.sk = os.environ['VOLC_AK'], os.environ['VOLC_SK']
cfg.region, cfg.client_side_validation = 'cn-shanghai', False
volcenginesdkcore.Configuration.set_default(cfg)
api = MLPLATFORM20240701Api(volcenginesdkcore.ApiClient(cfg))

# Parse YAML and submit
y = yaml.safe_load(open('xvla/scripts/stage1_kai_warmup_16gpu.yaml').read())
QID = {'robot-task': 'q-20251204185107-fvnpx', 'Robot-East-H20': 'q-20260516104437-2ml4v'}
body = {
    'Name': y['TaskName'],
    'ResourceConfig': {
        'ResourceQueueId': QID[y['ResourceQueueName']],
        'MaxRuntimeSeconds': int(y.get('ActiveDeadlineSeconds', 86400)),
        'Roles': [{'Name': r['RoleName'], 'Replicas': int(r['RoleReplicas']),
                   'Resource': {'InstanceTypeId': r['Flavor'], 'ZoneId': 'cn-shanghai-a'}}
                  for r in y['TaskRoleSpecs']],
    },
    'RuntimeConfig': {
        'Framework': y.get('Framework', 'Custom'),
        'Image': {'Url': y['ImageUrl'], 'Type': 'Prebuild'},
        'Command': y['Entrypoint'],
        'Envs': [{'Name': e['Name'], 'Value': str(e['Value']),
                  'IsPrivate': bool(e.get('IsPrivate', False))} for e in y.get('Envs', [])],
    },
    'StorageConfig': {
        'Storages': [{'Type': s['Type'], 'MountPath': s['MountPath'],
                      'Config': {'Vepfs': {'Id': s['VepfsId'], 'SubPath': s.get('SubPath', '')}}}
                     for s in y['Storages']],
        **({'CacheType': y['CacheType']} if y.get('CacheType') else {}),
    },
}
r = api.create_job(body)
print('task_id:', r['Result']['Id'])  # e.g. t-20260520225742-jv6jk
```

**Stop / Get：**

```python
api.stop_job({'Id': 't-20260520225742-jv6jk'})
r = api.get_job({'Id': 't-20260520225742-jv6jk'})
print(r['Result'].get('State'))    # Running / Success / Failed / Stopped
```

**封装好的 helper** (用 submit_yaml.py CLI 形式, 处理 dry-run + 错误): `train_scripts/volc/submit_yaml.py`。

> **Queue ID + 容量速查见 §5.6.c.2** (附实测可用 / 已用 GPU 数)。

### 5.6.b 16-卡 H20 集群训练 YAML 配置要点

> ℹ️ **集群训练当前走 §5.6.c (vsubmit 一键提交)**。本节列出 YAML 中针对 16-GPU H20 集群的关键字段差异 (与 8 GPU 单节点对比), 用于自定义 YAML 编写。

模板: `train_scripts/volc/gf3_cluster_smoke_16gpu.yaml` (2 节点 × 8 H20 = 16 GPU, FSDP=16)。

```yaml
ImageUrl:        "visincept-cn-beijing.cr.volces.com/grasp/h2r:1.0"   # cn-beijing CR (与 cn-shanghai CR 是不同 region)
ResourceQueueName: "Robot-North-H20"                                   # auto: cn-beijing / cn-beijing-e
TaskRoleSpecs:
  - RoleName: "worker"
    RoleReplicas: 2
    Flavor: "ml.hpcpni3ln.45xlarge"                                    # 8× H20-SXM5-96GB, RDMA
Storages:
  - Type: "Vepfs"
    VepfsId: "vepfs-cnbj875793a96d6b"                                  # 华北 vePFS, 与 gf3 共享
    MountPath: "/vePFS-North-E"
```

提交:
```bash
source ~/.volc_creds
python train_scripts/volc/submit_yaml.py train_scripts/volc/gf3_cluster_smoke_16gpu.yaml
# 或 dry-run:
python train_scripts/volc/submit_yaml.py train_scripts/volc/gf3_cluster_smoke_16gpu.yaml --dry-run
```

**vePFS 与 .venv self-containment (关键)**: gf3 上的 `.venv` 是经 sed 重写并把 uv-managed Python 一并搬到 vePFS 的版本 (见 §3.1 注解), volc 集群任一新节点 mount `vepfs-cnbj875793a96d6b` 后 `source .venv/bin/activate` 直接可用, **无需在每节点重装**。

**JAX 多机协调**: entrypoint 用 volc 提供的 `MLP_WORKER_0_HOST` / `MLP_WORKER_NUM` / `MLP_ROLE_INDEX` 设 `JAX_COORDINATOR_ADDRESS=$MLP_WORKER_0_HOST:15830` (port 15830, **不要用 `MLP_WORKER_0_PORT=2222` — 那是 SSH 端口冲突**)。

**注意事项 (同 5.6 通用):**

- ckpt 写入 `/vePFS-North-E/<...>/checkpoints/<config>/<exp_name>/<step>/`, vePFS 本地立即可见
- multi-host orbax 保存可能 race, 用 `--overwrite` 或 node-0 预清理
- 日志走 vePFS 共享, `logs/cluster_smoke_*_node${MLP_ROLE_INDEX}.log`, gf3 上 tail 即可
- 任务列表 / GUI: `https://console.volcengine.com/ml-platform/region:ml-platform+cn-beijing/task`

### 5.6.c gf0 作为**所有训练资源的统一控制平面** (2026-05-21 起, 推荐) ⭐

**核心原则**: gf0 是**全部 5 个训练资源**的唯一控制 / 监控入口:

```
              ┌──── 火山 cn-beijing  Robot-North-H20 (56 H20) ── via mlp/volc API
              │
              ├──── 火山 cn-shanghai robot-task       (28 A100) ── via mlp/volc API
   [gf0] ─────┤
              ├──── uc01 (8 A800)  ── via ssh (本地 SSH config 内网联通)
              ├──── uc02 (8 A800)  ── via ssh
              └──── uc03 (8 A800)  ── via ssh
```

本地 (laptop) / 用户终端只需 `ssh gf0 "<command>"`, 无需关心底层是哪个集群。

**为什么是 gf0**:
1. ✅ **唯一稳定通公网 + 火山的机器** (uc / 本地有时网络抖动, gf3 是计算节点)
2. ✅ **凭据集中** — AK/SK 只在 gf0 上, 不分散到多台 (减少泄露面)
3. ✅ **已有 submit_yaml.py 基础设施** (`train_scripts/volc/submit_yaml.py`)
4. ✅ **共享 vePFS 入口** — gf0 上 `/vePFS` 直接 visible 给 robot-task; Robot-North-H20 任务的 stdout 走 gf3 vePFS, 但 job 管理 (list/stop/logs) 都通过 gf0 经火山 API
5. ✅ **TOS 跨 region 中转** — gf0 也是 cnsh↔cnbj 数据传输的网关
6. ✅ **gf0 → uc SSH 已通** (2026-05-21 实测 — 见 §5.6.d)

#### 5.6.c.1 gf0 一次性设置

**(1) 取火山 AK/SK** (从火山 web console → 访问密钥):
- 推荐子账号 + 读写 ML Platform 权限
- 格式: AK 以 `AKLT` 开头, SK 是 base64 编码字符串 (带 `==` 结尾)

**(2) 在 gf0 上装 mlp CLI** (二进制已在 gf3, 复制过来):
```bash
ssh gf0
scp gf3:/root/.volc/bin/volc ~/.volc/bin/volc
scp gf3:/root/.volc/bin/mlp  ~/.volc/bin/mlp
chmod +x ~/.volc/bin/{volc,mlp}
echo 'export PATH=$HOME/.volc/bin:$PATH' >> ~/.bashrc
```

**(3) gf0 凭据文件** `~/.volc/credentials` (mode 0600):
```ini
[default]
access_key_id     = AKLT<your-ak>
secret_access_key = <your-sk-base64-with-trailing-==>
region            = cn-beijing
```

并在 `~/.bashrc` 加 export 给 Python SDK:
```bash
export VOLC_AK=$(awk -F= '/access_key_id/{gsub(/[ \t]/,"",$2); print $2}' ~/.volc/credentials)
export VOLC_SK=$(awk -F= '/secret_access_key/{gsub(/[ \t]/,"",$2); print $2}' ~/.volc/credentials)
```

> ⚠️ **region 字段不影响 OpenAPI**: volc CLI 走的是 `open.volcengineapi.com` 全 region 入口, 实际 region 由 API call 时 `Credentials(region=...)` 决定 (见 §5.6.c.4)。但**填 `cn-beijing` 比 `cn-shanghai` 更稳** (默认 cn-shanghai region 部分接口可能 401)。

**(4) gf0 Python SDK** (已装, 验证):
```bash
ssh gf0 "python3 -c 'import volcengine, volcenginesdkcore; print(\"OK\")'"
```

如果未装:
```bash
ssh gf0 "pip install --user volcengine volcengine-python-sdk"
```

#### 5.6.c.2 可用队列 (2026-05-21 实测)

> 通过 `ListResourceQueues` 跨 region 查询 + 个人 quota 验证, 当前账号可用 **2 个队列**:

| Region | Queue Name | Queue ID | 节点 × 类型 | Total GPU | Allocated | Free | 单节点 GPU/CPU/MEM | RDMA |
|---|---|---|---|---:|---:|---:|---|---|
| **cn-beijing** | **Robot-North-H20** ⭐ | `q-20260516104642-khch9` | **7 × ml.hpcpni3ln.45xlarge** | **56 H20** | 17 | **39** | 8 × H20-SXM5-96GB / 180 vCPU / 1960 GiB | 4× |
| **cn-shanghai** | **robot-task** | `q-20251204185107-fvnpx` | 1 × ml.pni2.14xlarge + 3 × ml.hpcpni2.28xlarge | **28 A100** | 24 | 4 | 8 × Tesla-A100-80G / 112 vCPU / 1715 GiB | varies |

**核心区别**:
- **Robot-North-H20** (cn-beijing): 新, 大 (56 H20), **多机集群训练首选**, 走 `vepfs-cnbj875793a96d6b` (与 gf3 共享 FS)
- **robot-task** (cn-shanghai): 旧, 小 (28 A100), 已被其他实验占 86%, 走 `vepfs-cnsh075262e1f815` (与 gf0 共享 FS)

#### 5.6.c.3 经 gf0 操作 mlp CLI 速查

> 模式: `ssh gf0 "mlp <cmd>"`. 或先 `ssh gf0` 后在 gf0 终端跑 (避免反复 SSH 认证)。

```bash
# 列出 jobs (全部 region 可见; queue_id 决定实际 region)
ssh gf0 "mlp job list --page-size 30"                                       # all states
ssh gf0 "mlp job list --state Running --page-size 30"
ssh gf0 "mlp job list --state Queueing"
ssh gf0 "mlp job list --resource-queue-id q-20260516104642-khch9"           # 仅 Robot-North-H20
ssh gf0 "mlp job list --resource-queue-id q-20251204185107-fvnpx"           # 仅 robot-task

# 详情 (JSON 格式)
ssh gf0 "mlp job get --id t-XXXXXXXX-XXXXX -o json"

# 停止
ssh gf0 "mlp job stop --id t-XXXXXXXX-XXXXX"

# 日志 (实时 tail)
ssh gf0 "mlp job logs --id t-XXXXXXXX-XXXXX --instance-name worker-0 --follow"

# 拉历史日志到 gf0 local (然后 scp 回本地)
ssh gf0 "mlp job logs --id t-... --instance-name worker-0 > /tmp/job_t-....log"
scp gf0:/tmp/job_t-....log /tmp/  # 拉回本地查看
```

**本地一键 helper alias** (`~/.bashrc` on laptop):
```bash
alias vlist='ssh gf0 "mlp job list --state Running --page-size 30"'
alias vget='ssh gf0"mlp job get -o json --id"'   # 用法: vget t-XXX
alias vstop='ssh gf0"mlp job stop --id"'         # 用法: vstop t-XXX
alias vlog='ssh gf0"mlp job logs --follow --instance-name worker-0 --id"'  # vlog t-XXX
```

#### 5.6.c.4 gf0 上的提交脚本

> 提交脚本统一放在 `/vePFS/tim/workspace/deepdive_kai0/train_scripts/volc/submit_yaml.py` (gf0 vePFS), 已 git 化随代码同步。

> SDK 5.0.27 自带 `KeyError: '.models'` 反序列化 bug, 必须 monkey-patch (见 §5.6 提交脚本)。**或绕过 SDK 直接用 `volcengine.base.Service` 调用 OpenAPI** — 此方式无 bug 且代码更简:

```python
#!/usr/bin/env python3
"""LOCAL submit to Robot-North-H20 via volcengine Service API (no SDK deserializer bug)."""
import os, json, yaml
from volcengine.ApiInfo import ApiInfo
from volcengine.Credentials import Credentials
from volcengine.ServiceInfo import ServiceInfo
from volcengine.base.Service import Service

# Queue → region/zone mapping
QUEUES = {
    "Robot-North-H20": {"id": "q-20260516104642-khch9", "region": "cn-beijing",  "zone": "cn-beijing-e",
                        "vepfs_id": "vepfs-cnbj875793a96d6b", "vepfs_mount": "/vePFS-North-E",
                        "image_cr": "visincept-cn-beijing.cr.volces.com/grasp/h2r:1.0"},
    "robot-task":      {"id": "q-20251204185107-fvnpx",  "region": "cn-shanghai", "zone": "cn-shanghai-a",
                        "vepfs_id": "vepfs-cnsh075262e1f815", "vepfs_mount": "/vePFS",
                        "image_cr": "visincept-cn-shanghai.cr.volces.com/grasp/h2r:1.0"},
}

def submit(yaml_path):
    cfg = yaml.safe_load(open(yaml_path))
    q = QUEUES[cfg["ResourceQueueName"]]
    si = ServiceInfo('open.volcengineapi.com', {'Accept': 'application/json'},
                     Credentials(os.environ['VOLC_AK'], os.environ['VOLC_SK'],
                                 'ml_platform', q["region"]), 10, 30)
    api = {'CreateJob': ApiInfo('POST', '/', {'Action': 'CreateJob', 'Version': '2024-07-01'}, {}, {})}
    svc = Service(si, api)
    body = {
        "Name": cfg["TaskName"],
        "Description": cfg.get("Description", ""),
        "ResourceConfig": {
            "ResourceQueueId": q["id"],
            "MaxRuntimeSeconds": int(cfg.get("ActiveDeadlineSeconds", 86400)),
            "Roles": [{"Name": r["RoleName"], "Replicas": int(r["RoleReplicas"]),
                       "Resource": {"InstanceTypeId": r["Flavor"], "ZoneId": q["zone"]}}
                      for r in cfg["TaskRoleSpecs"]],
        },
        "RuntimeConfig": {
            "Framework": cfg.get("Framework", "Custom"),
            "Image": {"Url": cfg.get("ImageUrl", q["image_cr"]), "Type": "Prebuild"},
            "Command": cfg["Entrypoint"],
            "Envs": [{"Name": e["Name"], "Value": str(e["Value"]),
                      "IsPrivate": bool(e.get("IsPrivate", False))} for e in cfg.get("Envs", [])],
        },
        "StorageConfig": {
            "Storages": [{"Type": "Vepfs", "MountPath": q["vepfs_mount"],
                          "Config": {"Vepfs": {"Id": q["vepfs_id"], "SubPath": ""}}}],
        },
    }
    if cfg.get("CacheType"):
        body["StorageConfig"]["CacheType"] = cfg["CacheType"]
    raw = svc.json('CreateJob', {}, json.dumps(body).encode())
    d = json.loads(raw) if isinstance(raw, str) else raw
    print(json.dumps(d, ensure_ascii=False, indent=2)[:1500])
    if 'Result' in d and d['Result'].get('Id'):
        print(f"\n✅ task_id: {d['Result']['Id']}")
    return d

if __name__ == "__main__":
    import sys
    submit(sys.argv[1])
```

使用方式 (在 gf0 上):
```bash
ssh gf0
# AK/SK 已 export (在 ~/.bashrc 中 source ~/.volc/credentials parse)
cd /vePFS/tim/workspace/deepdive_kai0
python train_scripts/volc/submit_yaml.py train_scripts/volc/<your_task>.yaml
```

本地一键提交 alias (`~/.bashrc` on laptop):
```bash
alias vsubmit='ssh gf0 "cd /vePFS/tim/workspace/deepdive_kai0 && python train_scripts/volc/submit_yaml.py"'
# 用法: vsubmit train_scripts/volc/x1_delta_joint_16gpu.yaml
```

#### 5.6.c.5 经 gf0 状态查询 (Python, 无 SDK bug)

```python
from volcengine.ApiInfo import ApiInfo
from volcengine.Credentials import Credentials
from volcengine.ServiceInfo import ServiceInfo
from volcengine.base.Service import Service
import json, os

def get_svc(region):
    si = ServiceInfo('open.volcengineapi.com', {'Accept': 'application/json'},
                     Credentials(os.environ['VOLC_AK'], os.environ['VOLC_SK'], 'ml_platform', region), 5, 5)
    return Service(si, {
        'ListResourceQueues': ApiInfo('POST', '/', {'Action': 'ListResourceQueues', 'Version': '2024-07-01'}, {}, {}),
        'ListJobs':           ApiInfo('POST', '/', {'Action': 'ListJobs',           'Version': '2024-07-01'}, {}, {}),
        'GetJob':             ApiInfo('POST', '/', {'Action': 'GetJob',             'Version': '2024-07-01'}, {}, {}),
    })

# 查 Robot-North-H20 当前 running
svc = get_svc('cn-beijing')
r = svc.json('ListJobs', {}, json.dumps({"ResourceQueueId": "q-20260516104642-khch9", "PageSize": 30, "State": "Running"}).encode())
for j in json.loads(r)['Result'].get('List', []):
    print(j['Name'], j['Status']['State'], j['CreateTime'])
```

#### 5.6.c.6 数据 / Ckpt 跨网同步策略 ⭐ (gf 服务器是跳板, 与队列 1:1 对应)

**关键事实**: 每个 volc 队列各有 1 个 gf 服务器作为**本地跳板**, 共享同一 vePFS:

| Region | volc 队列 | vePFS ID | Mount Path | **跳板 gf 机** | gf 上路径 |
|---|---|---|---|---|---|
| cn-shanghai | **robot-task** | `vepfs-cnsh075262e1f815` | `/vePFS` | **gf0** | `/vePFS/tim/workspace/deepdive_kai0/` |
| cn-beijing | **Robot-North-H20** | `vepfs-cnbj875793a96d6b` | `/vePFS-North-E` | **gf3** | `/vePFS-North-E/vis_robot/workspace/deepdive_kai0/` |

> **gf 跳板的作用**:
> 1. **数据/代码上传站**: 本地 / uc / sim01 → 经 公网 SSH/rsync 推到 gf 上的 vePFS
> 2. **shared vePFS R/W 入口**: volc 节点挂载 vePFS 后, 代码 + 数据 + ckpt + 日志立即可见
> 3. **任务结果 tail 入口**: volc 节点的 stdout 重定向到 vePFS 上 `logs/`, 在 gf 跳板上 `tail -F` 即可实时看
> 4. **single-GPU smoke test**: gf3 本身是 1 H20 (与 Robot-North-H20 同节点类型), 可先单卡 smoke 验证再走 volc 集群

**同步流程示例**:

**(A) 推送数据到 robot-task (gf0/vePFS-cnsh)**:
```bash
# 本地 (or uc02) → gf0 vePFS
rsync -av /data/tim/data/Task_A/mix_dataset/ \
  gf0:/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/mix_dataset/

# 验证 volc 上看见
ssh gf0 "ls /vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/mix_dataset/ | head"
```

**(B) 推送数据到 Robot-North-H20 (gf3/vePFS-cnbj)**:
```bash
# 本地 (or uc02) → gf3 vePFS
rsync -av --info=progress2 /data/tim/data/Task_A/mix_dataset/ \
  gf3:/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/mix_dataset/

# 验证 volc 上看见
ssh gf3 "ls /vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/mix_dataset/ | head"
```

**(C) 跨 region (cnsh ↔ cnbj) 同步走 TOS**:
```bash
# 因为 vePFS-cnbj 与 vePFS-cnsh 完全隔离, 跨 region rsync 不能直连
# 走火山 TOS 跨 region 复制 (后端骨干, 不走公网)

# 方向 1: gf0 (cnsh) → TOS → gf3 (cnbj)
ssh gf0 "tosutil cp -r /vePFS/tim/.../mix_dataset/ tos://transfer-shanghai/temp/mix_dataset/"
ssh gf3 "tosutil cp -r tos://transfer-shanghai/temp/mix_dataset/ /vePFS-North-E/vis_robot/.../mix_dataset/"
# 速度: pi05_base.tar 12.3G + 数据 17G ≈ 4-6 分钟
```

**Ckpt 反向取回 (volc 完成 → 本地真机部署)**:
```bash
# Robot-North-H20 job 写 ckpt → vepfs-cnbj 上
# 在 gf3 上确认
ssh gf3 "ls /vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/checkpoints/<config>/<exp>/"

# 拉到 sim01 (走 TOS)
ssh gf3 "tosutil cp -r /vePFS-North-E/.../<ckpt>/ tos://transfer-shanghai/checkpoints/<ckpt>/"
ssh sim01 "tosutil cp -r tos://transfer-shanghai/checkpoints/<ckpt>/ /data1/DATA_IMP/checkpoints/<ckpt>/"
```

#### 5.6.c.7 何时用哪个队列?

| 场景 | 推荐队列 | 理由 |
|---|---|---|
| 16-56 GPU 大规模集群训练 | **Robot-North-H20** | 7 节点 × 8 H20 = 56 GPU, 200 Gb/s RDMA, 内置 IB |
| 16 GPU 集群训练 | **Robot-North-H20** (2 节点) | 同上, 39 GPU free |
| 8 GPU 单节点 (smoke / debug) | Robot-North-H20 (1 节点) | 资源最足 |
| 仅 A100 兼容性测试 (历史复现) | robot-task | sm_80 = uc 集群一致 |
| 紧急小任务 (其他队列都满) | robot-task | 4 GPU 余量 |

**默认走 Robot-North-H20**, 因 (a) 资源最足, (b) H20 比 A100 快 ~1.5× (sm_90), (c) 与 gf3 同 vePFS (调试方便)。

#### 5.6.c.8 监控 / 日志统一通过 gf0

**Job 控制日志 (API 层)** — 经 gf0:
```bash
# 实时 tail (mlp 是 streaming, 用 ssh -t 给 TTY 让中断信号正常)
ssh -t gf0 "mlp job logs --id t-... --instance-name worker-0 --follow"

# 拉历史日志到 gf0, 再 scp 回本地
ssh gf0 "mlp job logs --id t-... --instance-name worker-0 > /tmp/job.log"
scp gf0:/tmp/job.log /tmp/
```

**训练 stdout (entrypoint 重定向到 vePFS)** — 直接到对应 vePFS 跳板:
```bash
# Robot-North-H20 任务的训练日志走 vepfs-cnbj (gf3 跳板)
ssh gf3 "tail -F /vePFS-North-E/vis_robot/logs/<exp_name>_*.log"

# robot-task 任务的训练日志走 vepfs-cnsh (gf0 跳板)
ssh gf0 "tail -F /vePFS/tim/workspace/deepdive_kai0/logs/<exp_name>_*.log"
```

**job 状态 dashboard** (gf0 + cron, 可选): 在 gf0 上跑 5 分钟轮询的脚本, 把 running/queueing 任务汇总写到 vePFS, 本地可定期 scp 拉。

```bash
# /vePFS/tim/workspace/deepdive_kai0/train_scripts/volc/dashboard.sh (gf0)
#!/usr/bin/env bash
OUT=/vePFS/tim/workspace/deepdive_kai0/logs/volc_dashboard.txt
while true; do
  date > "$OUT"
  echo "=== Running ==="                                                       >> "$OUT"
  mlp job list --state Running --page-size 30                                  >> "$OUT"
  echo                                                                         >> "$OUT"
  echo "=== Queueing ==="                                                      >> "$OUT"
  mlp job list --state Queueing --page-size 30                                 >> "$OUT"
  sleep 300
done
```
本地一键看:
```bash
ssh gf0 "cat /vePFS/tim/workspace/deepdive_kai0/logs/volc_dashboard.txt"
# 或 alias vdash='ssh gf0 cat /vePFS/.../volc_dashboard.txt'
```

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

`/vePFS/tim/workspace/deepdive_kai0/train_scripts/dashboard_all.sh` (gf0 上):
```bash
#!/usr/bin/env bash
# 5min 轮询: 火山 jobs + uc01/02/03 training procs + GPU util
OUT=/vePFS/tim/workspace/deepdive_kai0/logs/all_resources.txt
while true; do
  {
    date '+=== %Y-%m-%d %H:%M:%S ==='
    echo
    echo '┌─────────── 火山 ML Platform ───────────'
    echo '│ Running:'; mlp job list --state Running --page-size 20 | head -20
    echo '│ Queueing:'; mlp job list --state Queueing --page-size 10 | head -8
    echo
    for h in uc01 uc02 uc03; do
      echo "┌─────────── $h ───────────"
      ssh -o ConnectTimeout=5 $h "
        echo '│ GPU:'; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | head -10
        echo '│ Training procs:'
        ps aux | grep -E 'python.*train\.py' | grep -v grep | awk '{print \"│  PID=\"\$2\" cmd=\"\$11\" \"\$12\" \"\$13}'
      " 2>&1 | grep -v 'Warning\|setlocale'
    done
  } > "$OUT.tmp" && mv "$OUT.tmp" "$OUT"
  sleep 300
done
```

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
| **火山 提交** | `ssh gf0 "cd /vePFS/.../kai0 && python train_scripts/volc/submit_yaml.py <yaml>"` |
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
alias vsubmit='ssh gf0 "cd /vePFS/tim/workspace/deepdive_kai0 && python train_scripts/volc/submit_yaml.py"'
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

## 6. 机器间数据同步 ⭐ (2026-05-21 重构: TOS 为中心枢纽)

### 6.0 总架构: sim01 是源, TOS 是枢纽, 训练服务器都从 TOS 拉

```
                 ┌─────────────────────────┐
                 │   Real Robot 真机采集     │
                 │   (摇操 / DAgger / 部署)   │
                 └─────────────┬───────────┘
                               │ direct write
                               ▼
                 ┌─────────────────────────┐
                 │   sim01 (数据中心)         │
                 │   /data1/DATA_IMP/KAI0/   │
                 │   (raw → cleaned → ready) │
                 └─────────────┬───────────┘
                               │ tosutil upload (~85 MB/s 公网)
                               ▼
        ┌─────────────────────────────────────────────────┐
        │  TOS bucket (cn-shanghai, 中心枢纽 ⭐)            │
        │  tos://transfer-shanghai/KAI0/                  │
        │    ├── dataset/  (训练数据集 canonical 副本)       │
        │    ├── checkpoints/  (各 exp ckpt tar)           │
        │    └── base_init_ckpts/  (pi05_base 等基础模型)   │
        └────────┬────────┬────────┬────────┬─────────────┘
                 │        │        │        │
       pull ▼   ▼   pull ▼   pull ▼   pull ▼  (tosutil download)
              gf0      gf3     uc01    uc02/uc03
              │        │        │       │
        /vePFS/   /vePFS-North-E/  /data/shared/.../kai0/data/
        (cnsh)   /vis_robot/      (各 uc 本地 4TB ext4)
                 (cnbj)
```

**核心原则**:
1. **sim01 = 数据 single source of truth** — 真机数据先到 sim01, 清洗/打包/审核后上传 TOS
2. **TOS = 唯一权威副本** — 所有训练服务器都**从 TOS 拉**, 不互相 P2P 传输
3. **训练服务器是 TOS 的镜像消费者** — 本地 path 与 TOS 路径**保持一致**, 通过定期 resync 保证一致
4. **gf0/gf3 的 vePFS 与火山队列绑定** — 数据落地后 volc job 自动可见 (无需再次传输)
5. **uc 集群本地 SSD** — 拉 TOS 后保留本地, 用于训练 IO 加速

**反模式 (禁止)**:
- ❌ uc02 → uc03 直接 rsync 数据 (应都从 TOS 拉, 保持权威副本)
- ❌ gf0 → gf3 直接传 (走 TOS, 否则破坏中心化)
- ❌ 真机数据先到 gf0 不到 sim01 (绕开 single source)

### 6.1 sim01 数据上传 TOS (数据进入枢纽)

**前置**: sim01 上有 `tosutil` CLI + AK/SK (data_manager 后端常驻配置)。

**单文件上传**:
```bash
ssh sim01
cd /data1/DATA_IMP/KAI0/dataset/Task_A/cleaned/A_new_smooth_800/
tosutil cp -r ./base tos://transfer-shanghai/KAI0/dataset/Task_A/A_new_smooth_800/base
tosutil cp -r ./val  tos://transfer-shanghai/KAI0/dataset/Task_A/A_new_smooth_800/val
```

**整集打包上传 (推荐, 减少小文件 overhead)**:
```bash
# tar 后上传 (大块串行更快)
cd /data1/DATA_IMP/KAI0/dataset/Task_A/
tar -cf - A_new_smooth_800/ | tosutil cp - tos://transfer-shanghai/KAI0/dataset/Task_A/A_new_smooth_800.tar
```

**数据集 manifest** (`sim01:/data1/DATA_IMP/KAI0/MANIFEST.md`):
- 每次上传后更新 manifest 记录: dataset name / 上传日期 / 总大小 / ep 数 / 校验 hash
- TOS 上同名旧版本被覆盖 → manifest 标记版本递进 (`v1`, `v2`, ...)

### 6.2 训练服务器从 TOS 拉数据 (镜像消费)

各服务器 KAI0 数据路径**与 TOS 路径一一对应**, 拉取流程:

| 训练服务器 | KAI0 本地路径 (mirror) | 拉取命令 |
|---|---|---|
| **gf0** | `/vePFS/tim/data/KAI0/...` | `cd /vePFS/tim/data/KAI0 && tosutil cp -r tos://transfer-shanghai/KAI0/dataset/Task_A/<dataset>/ ./dataset/Task_A/` |
| **gf3** | `/vePFS-North-E/vis_robot/dataset/KAI0/...` | `cd /vePFS-North-E/vis_robot/dataset/KAI0 && tosutil cp -r tos://transfer-shanghai/KAI0/dataset/Task_A/<dataset>/ ./dataset/Task_A/` |
| **uc01/02/03** | `/data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/Task_A/<dataset>/` | `cd /data/shared/.../kai0/data && tosutil cp -r tos://transfer-shanghai/KAI0/dataset/Task_A/<dataset>/ ./Task_A/` |

**关键: 路径前缀对齐**:
```
TOS:  tos://transfer-shanghai/KAI0/dataset/Task_A/A_new_smooth_800/
            │                       │
            └── 子路径与服务器本地  ─┴── 完全一致
            
gf0:  /vePFS/tim/data/KAI0/dataset/Task_A/A_new_smooth_800/
gf3:  /vePFS-North-E/vis_robot/dataset/KAI0/dataset/Task_A/A_new_smooth_800/
ucNN: /data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/Task_A/A_new_smooth_800/
```

### 6.3 跨服务器 sync 工作流 (经 gf0 统一发起)

由于 gf0 是统一控制平面 (§5.6.c), TOS sync 命令也从 gf0 发起:

```bash
# gf0 → 火山 / uc 上的数据 sync 都通过 gf0 ssh + 本机 tosutil
ssh gf0 'bash -s' <<'EOF'
# 1. 通知 sim01 准备好数据后, gf0 触发各服务器拉取
DATASET=A_new_smooth_800

# gf0 自己拉 (cnsh vepfs)
cd /vePFS/tim/data/KAI0/dataset/Task_A
tosutil cp -r tos://transfer-shanghai/KAI0/dataset/Task_A/$DATASET ./

# gf3 拉 (cnbj vepfs)
ssh gf3 "cd /vePFS-North-E/vis_robot/dataset/KAI0/dataset/Task_A && \
  tosutil cp -r tos://transfer-shanghai/KAI0/dataset/Task_A/$DATASET ./"

# 各 uc 拉 (uc 本地 SSD)
for u in uc01 uc02 uc03; do
  ssh $u "cd /data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/Task_A && \
    tosutil cp -r tos://transfer-shanghai/KAI0/dataset/Task_A/$DATASET ./"
done

# 验证 (各服务器抽样查目录大小 + ep 数)
for h in gf0 gf3 uc01 uc02 uc03; do
  P=$(case $h in
    gf0) echo /vePFS/tim/data/KAI0/dataset/Task_A/$DATASET;;
    gf3) echo /vePFS-North-E/vis_robot/dataset/KAI0/dataset/Task_A/$DATASET;;
    *)   echo /data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/Task_A/$DATASET;;
  esac)
  echo -n "$h: "; [ "$h" = "gf0" ] && du -sh "$P" 2>/dev/null || \
    ssh $h "du -sh $P 2>/dev/null"
done
EOF
```

### 6.4 Ckpt 回流 (训练完成 → 经 TOS → sim01 部署)

反向链路 (训练产物回到 sim01 真机部署):

```
[训练服务器 ckpt] ──tosutil──→ TOS  ──tosutil──→ sim01 [部署]
   gf0 (vepfs-cnsh)
   gf3 (vepfs-cnbj)
   ucNN (本地)
```

**统一通过 gf0 调度**:
```bash
# 步骤 1: 训练完成后, 在训练所在服务器打包 + 上传 TOS
# 例如 Robot-North-H20 ckpt 在 gf3 vePFS:
ssh gf0 'ssh gf3 "
  cd /vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/checkpoints/<config>/<exp> && \
  tar -cf - <best_step>/ | tosutil cp - tos://transfer-shanghai/KAI0/checkpoints/<exp>_step<N>.tar
"'

# 步骤 2: sim01 从 TOS 拉
ssh sim01 'cd /data1/DATA_IMP/checkpoints && \
  tosutil cp tos://transfer-shanghai/KAI0/checkpoints/<exp>_step<N>.tar ./ && \
  tar -xf <exp>_step<N>.tar'
```

### 6.5 TOS bucket 目录约定 ⭐

```
tos://transfer-shanghai/KAI0/
├── dataset/                       # 训练数据集 (canonical)
│   ├── Task_A/
│   │   ├── kai0_base/            # 官方 base (3055 ep)
│   │   ├── kai0_dagger/          # 官方 dagger (3457 ep)
│   │   ├── vis_base/             # 自采 raw (各日期子目录)
│   │   ├── vis_base_clean_v2/    # 自采清理后 (837 ep)
│   │   ├── A_new_smooth_800/     # 合并后训练 dataset
│   │   ├── A_new_pure_200/
│   │   └── A_new_pure2_1800/
│   ├── Task_E/                   # 扶起倒箱
│   └── Task_P/                   # 抓放盒子
│
├── checkpoints/                   # 训练完成 ckpt (tar 压缩)
│   ├── task_a_new_smooth_800_step49999.tar
│   ├── task_a_new_pure_200_step49999.tar
│   └── ...
│
├── base_init_ckpts/              # 基础模型 (供训练 init 用)
│   ├── pi05_base.tar             # 12.3 GB, 官方 pi0.5
│   ├── mixed_1_clean/            # Task_A 训练好的中间 ckpt
│   └── ...
│
└── external/                     # 第三方资源
    ├── XVLA-Soft-Fold/           # Facebear HF 数据集副本
    └── ...
```

### 6.6 同步频率 + 一致性约定

| 数据类别 | 上传时机 | 拉取时机 | 一致性级别 |
|---|---|---|---|
| **训练数据集** (`dataset/`) | sim01 清洗完成 + manifest 更新 | 训练任务启动前 (各服务器手动拉) | 永久 — 同名版本永远相同 |
| **Ckpt** (`checkpoints/`) | 训练完成 + 验证 MAE 合格 | sim01 部署时 / 其他服务器 init 时 | 永久 + 不变更 |
| **base_init_ckpts/** | 极少更新 (新 pi05_base 出版本时) | 各服务器初始化时一次 | 长期 — 当前版本固定 |
| **external/** | 一次性 (例如 XVLA-Soft-Fold) | 各服务器需要时 | 永久 |

**KEY**: TOS 副本是只追加 (append-only), 不修改不删除。要修改 → 上新版本 (`v2`, `v3`...) + 更新 manifest。

### 6.7 历史方法 (已 deprecated)

- ❌ ~~uc01 ↔ uc02 lsyncd~~ (跨集群一致性靠 TOS 保证, 不再依赖 lsyncd 实时镜像)
- ❌ ~~js01-04 JuiceFS 共享~~ (js 集群已停用)
- ❌ ~~gf0 → sim01 SSH 反向隧道传 ckpt~~ (改走 TOS)
- ⚠️ rsync 直连仅用于**临时小文件** (代码 patch / 配置 / 日志), 数据/ckpt **必须走 TOS**

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

## 8. 训练实验命名约定

```
<config_name>:    pi05_flatten_fold_<dataset_label>
<exp_name>:       <experiment_descriptor>_<version>
ckpt_path:        ${KAI0_DATA_ROOT}/checkpoints/<config>/<exp_name>/<step>/

例:
  config:   pi05_flatten_fold_mix_apr28_450
  exp_name: mix_apr28_450_v1
  ckpt:     /vePFS/.../checkpoints/pi05_flatten_fold_mix_apr28_450/mix_apr28_450_v1/28000/
```

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

## 10. 部署 Ckpt 工作流 (训练 → sim01 推理)

```
训练完成 (gf*)
  ↓
auto_pack_on_end.sh: 选 best step, 打包 params+assets+METADATA
  ↓
TOS upload (gf*) → /transfer-shanghai/KAI0/<name>.tar
  ↓
sim01 from_tos_file.py download
  ↓
sim01 解压 + symlink 到 kai0/checkpoints
  ↓
serve_policy.py 启动推理服务
```

详见 [`sim01_deployment.md`](./sim01_deployment.md) 与 [`checkpoints_layout.md`](./checkpoints_layout.md)。

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

**带宽**: write ~219 MB/s, read ~2 GB/s (单 stream NFSv4 over TCP)，跨 host 直传走 RoCE NIC eth1。

### 12.6 集群训练启动脚本模板 (`/tmp/run_cluster_3host.sh`)

```bash
#!/bin/bash
set -euo pipefail

CONFIG="<your_config_name>"
EXP_NAME="<exp_name>"
COORD_ADDR="192.168.1.2:15830"
LOG_DIR=/home/tim/workspace/deepdive_kai0/logs
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
mkdir -p $LOG_DIR

NCCL_OPTS='
unset NCCL_IB_DISABLE NCCL_NET_TYPE NCCL_NET_GDR_LEVEL NCCL_NET_GDR_READ
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_P2P_LEVEL=NVL
export NCCL_SOCKET_IFNAME=eth1
unset NCCL_MAX_NCHANNELS NCCL_MIN_NCHANNELS NCCL_BUFFSIZE
export NCCL_DEBUG=INFO
'

TRAIN_CMD="cd /home/tim/workspace/deepdive_kai0/kai0 && .venv/bin/python -u scripts/train.py $CONFIG --exp_name=$EXP_NAME --seed=123 --overwrite --no-wandb-enabled"

launch_worker() {
  local TGT=$1 PROC=$2 TAG=$3
  ssh -o StrictHostKeyChecking=no $TGT "
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export PATH=/home/tim/miniconda3/bin:/home/tim/.local/bin:\$PATH
export PYTHONUNBUFFERED=1
export KAI0_DATA_ROOT=/home/tim/workspace/deepdive_kai0/kai0
export KAI0_LOCAL_ROOT=/home/tim/local_ckpts
export OPENPI_DATA_HOME=/home/tim/workspace/openpi_cache
export JAX_COORDINATOR_ADDRESS=$COORD_ADDR
export JAX_NUM_PROCESSES=3
export JAX_PROCESS_INDEX=$PROC
export JAX_ENABLE_EMPTY_ARRAYS=true
export JAX_COMPILATION_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_COMPILATION_CACHE_MIN_COMPILE_TIME_SECS=1
unset XLA_FLAGS
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
$NCCL_OPTS
export WANDB_MODE=offline
mkdir -p $LOG_DIR
nohup bash -c '$TRAIN_CMD' > $LOG_DIR/run_${TAG}_${TIMESTAMP}.log 2>&1 &
echo \"[${TAG} proc${PROC}] pid=\$!\"
disown
"
}

launch_worker "tim@192.168.1.3" 1 "uc02"
launch_worker "tim@192.168.1.4" 2 "uc03"
sleep 5

# uc01 master (local exec — 复制相同 env)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export PATH=/home/tim/miniconda3/bin:/home/tim/.local/bin:$PATH
export PYTHONUNBUFFERED=1
export KAI0_DATA_ROOT=/home/tim/workspace/deepdive_kai0/kai0
export KAI0_LOCAL_ROOT=/home/tim/local_ckpts
export OPENPI_DATA_HOME=/home/tim/workspace/openpi_cache
export JAX_COORDINATOR_ADDRESS=$COORD_ADDR
export JAX_NUM_PROCESSES=3 JAX_PROCESS_INDEX=0
export JAX_ENABLE_EMPTY_ARRAYS=true
export JAX_COMPILATION_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_COMPILATION_CACHE_MIN_COMPILE_TIME_SECS=1
unset XLA_FLAGS
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
unset NCCL_IB_DISABLE NCCL_NET_TYPE NCCL_NET_GDR_LEVEL NCCL_NET_GDR_READ
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23 NCCL_IB_RETRY_CNT=7 NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_P2P_LEVEL=NVL
export NCCL_SOCKET_IFNAME=eth1
unset NCCL_MAX_NCHANNELS NCCL_MIN_NCHANNELS NCCL_BUFFSIZE
export NCCL_DEBUG=INFO
export WANDB_MODE=offline
cd /home/tim/workspace/deepdive_kai0/kai0
nohup .venv/bin/python -u scripts/train.py $CONFIG --exp_name=$EXP_NAME --seed=123 --overwrite --no-wandb-enabled > $LOG_DIR/run_uc01_${TIMESTAMP}.log 2>&1 &
echo "[uc01 proc0] pid=$!"
disown
```

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

