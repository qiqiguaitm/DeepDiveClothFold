# 训练存储布局与 Python 环境

> 文件结构 / ckpt 规范 / 数据集源 / 临时存储 / Python 栈 / 环境变量 / 训练实验命名约定。
>
> **同 series**: `overview.md` (服务器全景) / `ssh_and_credentials.md` / `data_sync_tos.md` / `submission/`

---

## 2. 文件结构

### 2.1 工作目录路径速查

| 服务器 | 工作目录 | 实际存储 |
|---|---|---|
| gf0 | `/vePFS/tim/workspace/deepdive_kai0/` (= `/home/tim/workspace/deepdive_kai0` 软链) | gpfs cnsh 跨机共享 |
| gf3 | `/vePFS-North-E/vis_robot/workspace/deepdive_kai0/` | gpfs cnbj, 与同队列其它节点共享 |
| uc01 | `/home/ubuntu/workspace/deepdive_kai0/` → `/data/shared/ubuntu/workspace/deepdive_kai0/` | uc01 本机 `/dev/vdb` 4 TB ext4 (NFS server) |
| uc02 | 同 uc01 路径 — 经 NFS 共享 uc01 内容 ⭐ | NFS mount `10.60.135.47:/data/shared/ubuntu/workspace`, NFSv4.1 over eth0 |
| uc03 | 同 uc02 — 经 NFS 共享 uc01 内容 ⭐ | 同 uc02; 另有本机 `/nix` 3.5T NVMe (闲置, 未参与 kai0) |

> ⭐ **2026-05-18 重装后**: uc01 `/etc/exports` 把 `/data/shared/ubuntu/workspace` export 给 `10.60.0.0/16`, uc02/03 mount 同路径. 代码/venv/数据集/集群训练 ckpt **均经 NFS 共享**, 不再各自独立 (lsyncd 已停). 单机训练 ckpt 仍走 NFS-内 symlink → 各机本地盘 (§2.2)。

### 2.2 Checkpoint 本地存储规范 ⭐ (2026-05-04 重要更新)

> **核心原则**: 每台服务器的 ckpt 写到独立的本地路径, 不跨机同步, 重启不丢失。

**统一路径**: 每台机器都使用 `/home/tim/local_ckpts/` 作为 ckpt 根目录 (其中是 symlink 还是 real dir 因机器而异)。

| Server | ckpt 根 | 物理后端 | 容量 (实测) | 持久性 |
|---|---|---|---|---|
| gf0 | `/home/tim/local_ckpts/` → `/vePFS/tim/gf0_local_ckpts/` (symlink) | /vePFS (50T 共享 FS) | 看 /vePFS 余量 | ✓ 持久 |
| uc01 | `/data/shared/ubuntu/local_ckpts/` (NFS 上的 symlink 字符串解析到本机) ⭐ | **`/dev/vdb` ext4 (4 TB)** ← 整个 `/data` 都是这块盘, 不是 `/dev/vda2` | uc01: 1.7 T avail (Use 56%) | ✓ 持久 |
| uc02 | 同 uc01 (各机本地 `/dev/vdb`) | 同 | (本机 `df /data` 自查) | ✓ 持久 |
| uc03 | 同 uc01 | 同 | (本机自查) | ✓ 持久 |

> ⭐ **uc 单机 ckpt 实际是"NFS 上的 symlink + 各机本地解析"** (2026-05-28 实测):
> - NFS 上: `/data/shared/ubuntu/workspace/deepdive_kai0/kai0/checkpoints -> /data/shared/ubuntu/local_ckpts` (symlink 字符串)
> - NFS export 范围 only `workspace/`, 不含 `ubuntu/local_ckpts/`
> - 各 host resolve 到自己本机 `/dev/vdb` 上的 `/data/shared/ubuntu/local_ckpts/` (真实 dir, 互不冲突)
> - 详见 `../../backup/uc_cluster_data_sharing_analysis.md §4` (uc 已停用). 历史 `/home/tim/local_ckpts/` 路径已是兼容别名, 当前 launcher 写入路径是 `<KAI0_DATA_ROOT>/checkpoints/...`, 经 NFS symlink 落本机盘。

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

**lsyncd 已废弃 (2026-05-18 重装后)**:
- 历史: uc01/uc02 之间曾有 lsyncd 双向 mirror `/data/shared/` 目录
- 现状 (2026-05-18 重装后): 改为 **uc01 NFS export `/data/shared/ubuntu/workspace`** 给 uc02/03, lsyncd 不再启用
- 仍然需要警惕的事: 千万 **不要直接写 ckpt 到** `<kai0>/checkpoints/<config>/<exp>` 真实目录 — 那是 NFS 路径会占共享空间。一律走 symlink trick (`checkpoints/` 已预置 symlink 到 `local_ckpts/`)

**keep_period 设置**:
- 100k step 训练: `keep_period=10000` (保留 10 个) 比 `2_000` (保留 50 个) 减少 5× 占用
- 50k step: `keep_period=10000` (保留 5 个) 大约 165GB; 默认 `2_000` 时 825GB 可能撑爆数据盘 (uc 用 `/dev/vdb` 4T, 余量见 §2.2 表; gf0 看 /vePFS 配额)

**已知 ckpt 路径**:

| 实验 | 当前 ckpt 真实路径 | 所有者 |
|---|---|---|
| uc01 实验1 | `/home/tim/local_ckpts/pi05_flatten_fold_mix_b6000_p1200_init_mixed_1/task_a_mix_base6000_pure1200_new_norm_base_mixed_1` | uc01 |
| uc02 实验2 | `/home/tim/local_ckpts/pi05_flatten_fold_mix_b6000_p1200_init_pi05_base/task_a_mix_base6000_pure1200_new_norm_base_pi0.5` | uc02 |
| gf0 实验3 | `/vePFS/tim/gf0_local_ckpts/pi05_flatten_fold_mix_b6000_p1200_init_pi05_base_100k/task_a_mix_base6000_pure1200_new_norm_base_pi0.5_100000` | gf0 |

### 2.3 数据集 / Checkpoint 目录约定 (传统 view)

> ⭐ **数据集存放规范 (2026-05-28 明确)**: **所有用户/脚本构建的数据集一律放 `kai0/data/Task_A/self_built/<name>/`**, 不要直接放 `Task_A/` 根目录下。
> - build 脚本 (`train_scripts/kai/data/build_*.py`) 的 `DST` 必须指向 `.../data/Task_A/self_built/<name>`。
> - config.py 中训练 config 的 `repo_id` 也指向 `self_built/<name>`。
> - 例外: 原始采集 (`vis_base/`, 真实本地盘 build 源) + HF 官方 (`kai0_base/`, `kai0_dagger/`, `kai0_advantage/`) 可放 `Task_A/` 根, 因它们不是"构建"产物。
> - 已迁移 (2026-05-28 完成): `vis_v2_full`, `vis_v2_merged`, `vis_v2_merged_val`, `val_kai0_official`, `A_0423_0527` 原直接放 `Task_A/` 根, 现全部移入 `self_built/`; config.py / build 脚本 / volc+xvla yaml 引用已同步更新。
> - **gf0 / gf3 / uc 三端均已按此规范对齐** (2026-05-28): 数据统一在 `kai0/data/Task_A/` 下 `kai0_base/kai0_dagger/kai0_advantage`(官方)+ `vis_base`(原始,仅-v2)+ `self_built/`(构建集);内部软链全部 retarget(dangling=0);config.py 对应各端路径(gf0 `/vePFS`、gf3 `/vePFS-North-E`、uc `/data/shared/.../kai0/data`)分别更新。uc 额外有 `vis_dagger/vis_autonomy/vis_inference`(原始自采,gf0/gf3 无)。各端 NFS/git-pull 机制详见 `../../backup/uc_cluster_data_sharing_analysis.md` (uc 已停用)。

> ⭐ **X-VLA EE6D 数据集存放规范 (2026-05-29 新增)**: pi0/openpi 用的 joint 数据集放 `kai0/data/Task_A/self_built/`(上条);**X-VLA 架构用的 EE6D 20D 数据集 (由 joint 转换而来) 一律放 `xvla/data/self_built/<name>/`**。
> - 该目录由 `xvla/data/self_built/.gitignore` (`*` + `!.gitignore`) 保留文件夹、忽略全部数据内容(重 parquet 不入 git, 仅存盘)。
> - 转换脚本: `train_scripts/xvla/data/joint_to_ee6d.py` (LeRobot parquet 14D joint→20D EE6D) / `convert_xvla_action.py` (hdf5)。⚠️ 转换约定 (rot6d interleaved 排布、gripper 二值化、无需 norm_stats) 见 `train_scripts/xvla/data/README.md`。
> - 训练 config 在 `train_scripts/xvla/launch/xvla_train.py` 的 `CONFIGS` dict, dataset `root` 指向 `xvla/data/self_built/<name>` 绝对路径。
> - 现有: `A_0423_0527` (1085 ep, 由 `kai0/.../self_built/A_0423_0527` joint 转换 + cnsh→uc TOS 传输)。


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
│   └── data/                          # 数据集入口 (gf0 现已全部真实落盘, 无软链)
│       └── Task_A/
│           ├── vis_base/              # build 源; 2026-06-02 v2/v3 分层 (见 data_sync_tos.md §6.8):
│           │   ├── v2/                 #   原始采集 <date>-v2 ×20 (含 depth), sync DST, cron 拉新; build 脚本 SRC_ROOT 指此
│           │   └── v3/                 #   裁投放 <date>-v3 ×20 (no-release: 裁 ep 开头投放静止段, 无 depth, 1956 ep)
│           ├── kai0_base/             # HF 官方 base — 真实目录 (2026-05-28: 原 base 改名而来, 不再是软链)
│           ├── kai0_dagger/           # HF 官方 dagger — 真实目录 (2026-05-28: 原 dagger 改名而来)
│           ├── kai0_advantage/        # HF 官方 advantage — 真实目录 (2026-05-28: 自 self_built/advantage 迁回根; 视频软链→kai0_base)
│           └── self_built/            # ⭐ 所有用户构建数据集一律放这里
│               ├── vis_v2_full/       # 全 v2 日期合并 (build_vis_v2_full.py, 视频软链→vis_base)
│               ├── vis_v2_merged/     # 合并数据集 (build_vis_v2_merged.py, 软链→vis_base)
│               ├── vis_v2_merged_val/ # cross-val hold-out (30 ep, build 出的)
│               ├── val_kai0_official/  # kai0_base+kai0_dagger 各留 15 集的 holdout (inline eval, build_val_kai0_official.py)
│               ├── A_pure_1200/{base,val}/
│               ├── A_new_pure_1200/{base,val}/
│               ├── A_new_pure_200/
│               ├── A_0423_0527/       # 13 dates 排校准漂移 (build_A_0423_0527.py)
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
/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/   # ⭐ 与 gf0 同规范 (2026-05-28 对齐):
    kai0_base/ kai0_dagger/         # HF 官方 base(3055)/dagger(3457) 真实目录 (原 base/dagger 改名)
    vis_base/                       # 原始采集 (原 vis_base_real 改名), build 源
    self_built/                     # 所有构建数据集: vis_v2_full, vis_v2_merged, vis_v2_merged_val, vis_5day_recent, A_new_pure_200, A_new_pure_200_val
                                    #   (vis_v2_merged 为视频底座真实文件; 其余视频软链→ self_built/vis_v2_merged 或 vis_base)
    (gf3 无 advantage; eval_val/kai0_*_val_50 为另置的小验证集, 未被 config 引用)
    # 注: 原 /vePFS-North-E/vis_robot/dataset/KAI0/ 这棵独立树已于 2026-05-28 合并进上面 kai0/data 并删除, 统一单一数据位置
/vePFS-North-E/vis_robot/base_init_ckpts/extracted/pi05_base/params/ # init weights (从 TOS pi05_base.tar 解压)
/vePFS-North-E/vis_robot/checkpoints/<config>/<exp>/                 # 训练输出
/vePFS-North-E/vis_robot/logs/                                       # 训练日志
/vePFS-North-E/vis_robot/workspace/deepdive_kai0/                    # 代码 (从 gf0 scp tarball)
/vePFS-North-E/vis_robot/workspace/.uv_python/                       # uv-managed Python (self-contained)
/vePFS-North-E/vis_robot/venv/                                       # 原始 venv.tar / uvpython.tar 缓存
```

> **跨 region 同步**: gf3 (cn-beijing) 不能直连 gf0/uc01/sim01 (cn-shanghai), 一切通过 TOS `tos://transfer-shanghai/...` 中转 (跨 region 走 TOS 后端骨干)。pi05_base.tar (12.3G) + 数据子集 ~17G 总同步 ≈ 4-6 分钟。

#### uc01 / uc02 / uc03  ⚠️ 已停用

> uc 集群已彻底停用 (2026-05-18 退役)。数据集源/NFS 共享布局归档见 [`../../backup/uc_cluster_reference.md`](../../backup/uc_cluster_reference.md)。

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


---

## 3. 环境 (Python 栈)

### 3.1 venv 路径

| 机器 | venv 路径 | Python |
|---|---|---|
| gf0 | `/vePFS/tim/workspace/deepdive_kai0/kai0/.venv` → `/home/tim/.kai0_venv` (本地 symlink) | 3.11 |
| gf3 | `/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/.venv` (**self-contained on vePFS**, 跨节点共享) | 3.12.13 |
| uc01 | `/data/shared/ubuntu/workspace/deepdive_kai0/kai0/.venv` (真实 dir on NFS server) ⭐ | 3.12 |
| uc02 | 同 uc01 路径, 经 NFS 共享 (跨机 inode 一致, 实测 inode=65539) ⭐ | 3.12 |
| uc03 | 同 uc02 — 经 NFS 共享 ⭐ | 3.12 |

> ⭐ **uc venv 是 NFS 共享而非本地独立** (2026-05-28 实测): `uv add` / `uv sync` 只需在 uc01 上跑**一次**, uc02/03 立即可用。**但 uv-managed Python interpreter** (`~/.local/share/uv/python/cpython-3.12.x/`) 是各机本地 — 因 `.venv/bin/python` 是 symlink 指向绝对路径 `~/.local/share/uv/python/...`, 各机必须先各跑过一次 `uv python install 3.12` 才能解析这个 symlink。

> **注意 (gf0 / gf3 区别)**:
> - **gf0**: vePFS 上的 `.venv` 是 symlink, 真实 venv 在本机 `/home/tim/.kai0_venv` (不跨机)
> - **gf3**: `.venv` 完全 self-contained 在 vePFS 上 — `python` 二进制 + uv-managed Python tree 都在 `/vePFS-North-E/vis_robot/workspace/.uv_python/cpython-3.12.13-linux-x86_64-gnu/` 下, `pyvenv.cfg home =` 也指 vePFS 路径。这样 volc 集群任意新节点 mount vePFS-North-E 后 `source .venv/bin/activate` 即可直接用, **无需在每节点重装**。
>
> **gf3 venv 构建路径 (2026-05-20)**: 由于 GitHub HTTPS 在 cn-beijing 跨 region 极不稳 (lerobot git fetch 反复 TLS stream cancel), 直接在 gf3 跑 `uv sync` 失败。改为: ① uc01 上 `tar` 现成 `.venv` (8.2 GB) + uv-managed Python (104 MB) 上传到 TOS `from_uc01/gf3/`; ② gf3 拉取后解压 + sed 重写 hardcoded 路径 (`/data/shared/ubuntu/workspace/deepdive_kai0/kai0` → `/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0`, `/home/ubuntu/.local/share/uv` → `/root/.local/share/uv`); ③ 后续再把 uv-managed Python 也搬到 vePFS, .venv 重 symlink. 一键脚本: `/root/gf3_install_venv.sh` (副本: `train_scripts/kai/launch/gf3_install_venv.sh`)。**全过程 6 分半**。

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

