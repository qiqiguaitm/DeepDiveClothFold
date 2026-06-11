# uc01/02/03 集群当前数据共享方式分析

> 2026-05-28 现场实测重写; 同日数据集迁移后刷新 §3 (数据从 `dataset/` 迁到 `kai0/data/Task_A/`, 对齐 gf0/gf3)。
>
> **结论速览**: uc 集群当前是 **单 NFS 平面 (uc01 export, 走管理网 eth0) + 单机本地 ckpt (symlink trick 绕 NFS)** 两层架构。文档历史描述的"RDMA NFS B 平面 (`/data/cluster_ckpt` 192.168.1.0/24)"**实测已废弃 / 未启用**。所有共享 (代码 / venv / 数据集 / 集群训练 ckpt) 都走唯一 NFS 平面。**2026-05-28 起数据集统一在 `kai0/data/Task_A/` (kai0_base/dagger/advantage + vis_base + self_built), 与 gf0/gf3 同规范; hourly cron `sync_tos_kai0.sh` 增量同步 TOS → vis_base (见 §3.3)。**

---

## 1. 硬件 / 网络 / 物理盘 (实测)

| 维度 | 值 |
|---|---|
| 节点 | uc01, uc02, uc03 (各 8× A800-80GB) |
| 系统盘 | `/dev/vda2` 492 GB ext4, mount `/` |
| 数据盘 | `/dev/vdb` 4 TB ext4, mount `/data` (uc01: 2.1T used / 1.7T avail) |
| 管理 / 公网 | `eth0` 10.60.x.x/16 (virtio_net, NFS 走此网) |
| RDMA 训练网 | `eth1-4` 192.168.{1-4}.x/24 (mlx5_core, 200 Gbps × 4, RoCEv2) — **当前仅用于 NCCL/RDMA**, 不参与 NFS |
| uc01 IP (mgmt) | `10.60.135.47` |
| uc01 hostname | `10-60-135-47` (账户 `ubuntu`) |

主机间 RDMA 平面 IP (仅 NCCL 用):
```
uc01: 192.168.{1,2,3,4}.2
uc02: 192.168.{1,2,3,4}.3
uc03: 192.168.{1,2,3,4}.4
```

---

## 2. NFS 平面 (唯一)

### 2.1 export 配置 (uc01 = server, 实测)

```
/etc/exports:
  /data/shared/ubuntu/workspace  10.60.0.0/16(rw,sync,no_subtree_check,no_root_squash)
```

> 这是 `/etc/exports` 的**全部内容** — 没有第二条, 没有 `/data/cluster_ckpt`, 没有 192.168.1.0/24 网段的 RDMA NFS。

### 2.2 client mount (uc02/03 实测)

```
10.60.135.47:/data/shared/ubuntu/workspace
  → /data/shared/ubuntu/workspace
  nfs4 vers=4.1, rsize=1048576, wsize=1048576, hard, proto=tcp, timeo=600
```

`findmnt -t nfs,nfs4` 在 uc02 / uc03 上都只显示这一项。

### 2.3 NFS 共享下内容 (uc01 实测, `du -sh`)

| 子目录 | 大小 | 用途 |
|---|---|---|
| `dataset/` | **783 GB** ⭐ | **TOS 同步主数据 (见 §3)** |
| `deepdive_kai0/` | 455 GB | 主代码 (含 `kai0/.venv` 8.2 GB + ckpt symlink — 见 §4) |
| `cluster_ckpts/` | 42 GB | **3-host HSDP/FSDP 集群训练 ckpt (走 NFS, 不是单独 RDMA NFS)** |
| `base_init_ckpts/` | 34 GB | 共享 init ckpt (pi05_base 13G, Task_A_mixed_1 22G) |
| `X-VLA-env/` | 8.5 GB | X-VLA 项目 venv / 环境 |
| `xvla_ckpts/` | 3.3 GB | X-VLA 训练 ckpt |
| `dataset_ee6d/` | 1.8 GB | (Kai0_official + Task_A 子集 + `xvla_soft_fold_action_cache`) |
| `logs/`, `piper_sdk/`, `xvla_scripts/` | <2 GB | 杂项 |

---

## 3. 数据集位置 (2026-05-28 已迁到 kai0/data, 对齐 gf0/gf3) ⭐

### 3.1 实际路径 (kai0/data/Task_A/, NFS 共享)

```
/data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/Task_A/   # ⭐ 与 gf0/gf3 同规范
├── kai0_base/ kai0_dagger/ kai0_advantage/   # HF 官方 (3055/3457/3055) 真实目录
├── vis_base/                  # 原始自采 base, 仅 -v2 (19 日期), build 源 + TOS 同步目标
├── vis_dagger/ vis_autonomy/ vis_inference/  # uc 特有: 原始自采 (gf0/gf3 无)
└── self_built/                # 构建集: A_new_100_5_16_5_18(+val), A_new_pure_200(+val),
                               #   vis_v2_merged(自包含真实)+vis_v2_merged_val, val_kai0_official, eval_val/, xvla_exp1_hard_merged
```

> **历史**: 2026-05-28 前 uc 数据在独立的 `dataset/` 树 (`dataset/KAI0`、`dataset/Kai0_official`、`dataset/Task_A` 分散), 与 gf0/gf3 的 `kai0/data/` 不一致。已全部迁入 `kai0/data/Task_A/` 并对齐命名 + retarget 22647 条跨数据集软链 (dangling=0) + 删冗余 `hf_kai0`(47G)。

### 3.2 残留 dataset/ (未纳入, 保留)

```
/data/shared/ubuntu/workspace/dataset/
├── KAI0/
│   ├── from_tos_file.py / to_tos.py / to_tos_file.py   TOS 传输 helper (AK/SK 硬编码)
│   ├── Task_A/base/{README.md, analysis}               (6 个非-v2 日期 + kai0_official_base 已删)
│   ├── Task_E/H/HP/P/PP/PS/                             其它任务 (未迁)
│   └── task_a_mix_*.tar / task_p_*.tar                  ckpt 中转 tar
└── Kai0_official/                                       仅剩 HF metadata (base/dagger/advantage 已迁 kai0_*)
```

### 3.3 TOS → vis_base 自动同步 (hourly cron) ⭐ (2026-05-28 重写)

uc01 cron `0 * * * * /home/ubuntu/scripts/sync_tos_kai0.sh`:
- **目标改为新路径**: `tos://transfer-shanghai/KAI0/Task_A/base/<date>-v2/` → `kai0/data/Task_A/vis_base/<date>-v2/`
- 与 gf0 的 `train_scripts/kai/data/sync_vis_base_from_tos.sh` 同逻辑: `tosutil cp -r -u`(size/crc 增量, 从不删本地), **只 -v2 日期**, **`-exclude='*top_head_depth*'`** 排除 depth, flock 防重叠。
- **旧脚本已废弃**: 原 `sync_tos_kai0.sh` 把整个 `tos://.../KAI0/` 全量拉到 `dataset/KAI0/`(含 138G `Task_A_kai_official.tar`、depth、所有 task), 与迁移对冲、反复重拉 138G tar — 已重写。
- 只在 uc01 拉, uc02/03 经 NFS 自动可见。

**手动拉/传 (canonical helper, 仍在 `dataset/KAI0/`)**:
```bash
# 拉某日期到 vis_base
ssh uc01 "cd .../kai0/data/Task_A/vis_base && /home/ubuntu/tosutil cp -r -u \
  tos://transfer-shanghai/KAI0/Task_A/base/<date>-v2/ ./ -exclude='*top_head_depth*'"
# 上传 (to_tos.py 文件夹模式)
ssh uc01 "cd .../dataset/KAI0 && python to_tos.py --folder <dir> --tos_prefix KAI0/Task_A/..."
```
> ⚠️ TOS AK/SK 硬编码在 `from_tos_file.py` / `to_tos.py` / `~/.tosutilconfig`, 不读 env。

### 3.4 性能 (实测 GPU 99% util)

数据放 NFS 实测 GPU 99% util — NFS 在数据集消费场景下不是瓶颈 (`submission/uc_cluster_jobs.md §12.5`)。

---

## 4. 单机训练 ckpt: NFS-内 symlink trick (实测)

### 4.1 现状

```
NFS 上 (uc01 server 提供):
  /data/shared/ubuntu/workspace/deepdive_kai0/kai0/checkpoints
    → symlink (NFS 内字符串) → /data/shared/ubuntu/local_ckpts

NFS export 范围 ONLY: workspace/ 子树
  /data/shared/ubuntu/local_ckpts/  ← 不在 export 范围, 各机本地

各 host resolve:
  uc01: /data/shared/ubuntu/local_ckpts → 本机 /dev/vdb (ext4, 4TB)
  uc02: 同 → 各自本机 /dev/vdb
  uc03: 同 → 各自本机 /dev/vdb
```

uc01 / uc02 实测 symlink 字符串完全相同 (`-> /data/shared/ubuntu/local_ckpts`), mtime 都是 `May 18 12:45` (2026-05-18 重装时一次性建立)。

### 4.2 巧妙之处

- NFS 把 symlink **当字符串**传给 client; client 在本机 namespace 解析这个绝对路径。
- 因为 NFS 只 export 了 `workspace/`, 没 export `ubuntu/`, 所以同名兄弟目录 `ubuntu/local_ckpts/` 在各 host 上指向各自 `/dev/vdb` 上的真实 dir。
- 训练 ckpt write 走本地 ext4, 不挤 NFS 带宽, 也不互相覆盖。

### 4.3 注意: 物理盘是 `/dev/vdb` 不是 `/dev/vda2`

`storage_and_env.md §2.2` 旧表写 "uc01 ckpt 走 /dev/vda2 (492G ext4)" — **错**。实测:
- `/dev/vda2` mount `/` (系统盘)
- `/dev/vdb` mount `/data` (4 TB 数据盘, 整个 `/data/...` 包括 `local_ckpts/` 都在这上面)
- uc01 当前 `/data` 余 1.7 T (Use% 56%), 比旧表 "~290G 可用" 宽裕得多

---

## 5. 3-host HSDP/FSDP 集群训练 ckpt: **也走 NFS** (与历史文档不同) ⚠️

### 5.1 实测路径

```
/data/shared/ubuntu/workspace/cluster_ckpts/      42 GB, NFS 共享
├── pi05_flatten_fold_a_new_pure_200_js/
├── pi05_flatten_fold_kai0_official_kai_prompt/
├── xvla_exp1_hard_prompt_merged_uc/
└── xvla_exp1_hard_prompt_mixed_uc/
```

`findmnt -T` 显示 `/dev/vdb ext4` (uc01 上是本地盘, uc02/03 上 mount 是 NFS over eth0)。

### 5.2 RDMA NFS B 平面: **已废弃 / 未启用**

文档 `submission/uc_cluster_jobs.md §12.5` 写:
```
uc01 /etc/exports: /data/cluster_ckpt 192.168.1.0/24(rw,sync,...)
uc02/uc03 /etc/fstab: 192.168.1.2:/data/cluster_ckpt /cluster_ckpt nfs ...
```

**实测推翻**:
- uc01 `/etc/exports` 仅一条 (无 `/data/cluster_ckpt` 项)
- uc01 `/data/cluster_ckpt/` 是本地空目录 (仅 root owned, 没数据)
- uc02 / uc03 上 `/cluster_ckpt` **未 mount** (`findmnt /cluster_ckpt` 空)
- 当前 3-host 集群训练用 NFS 上的 `cluster_ckpts/` (复数, NFS 内) 替代

### 5.3 性能影响评估

走管理网 eth0 NFS 而非 RDMA NFS 的代价:
- Orbax checkpoint write 是周期性低频 (每 `keep_period` step 一次), 不是热路径 → eth0 带宽足够
- 实际训练吞吐 (`submission/uc_cluster_jobs.md §12.9`) pi05 HSDP 1.0 s/it, FSDP 1.2 s/it — 与 ckpt FS 解耦
- 若未来出 Orbax barrier 慢, 可考虑恢复 RDMA NFS, 但目前没有这个症状

---

## 6. 共享方式决策树 (现状)

```
要存什么?
│
├── 代码 / venv / 共享 init ckpt
│   └── 改 uc01 即可 → NFS 平面自动同步
│       路径: /data/shared/ubuntu/workspace/{deepdive_kai0/, base_init_ckpts/, X-VLA-env/}
│
├── TOS 同步的训练数据集 (2026-05-28 起在 kai0/data)
│   └── hourly cron sync_tos_kai0.sh: TOS Task_A/base/<date>-v2 → vis_base (exclude depth)
│       路径: /data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/Task_A/{vis_base, kai0_*, self_built/}
│       只在 uc01 拉 → NFS 自动可见 uc02/03
│
├── 3-host HSDP/FSDP 集群训练 ckpt (Orbax cross-host barrier)
│   └── 写 NFS 共享路径
│       路径: /data/shared/ubuntu/workspace/cluster_ckpts/<exp>/
│
├── 单机训练 ckpt
│   └── 走 NFS symlink trick → 各机本地 /dev/vdb
│       逻辑路径: $KAI0_DATA_ROOT/checkpoints/<config>/<exp>/
│                  = .../kai0/checkpoints → /data/shared/ubuntu/local_ckpts/...
│       物理路径: 各机本机 /dev/vdb ext4 (4TB)
│
└── 跨集群 (uc → 火山 cnsh/cnbj)
    └── rsync 公网到 gf0:/vePFS  或  TOS 中转 (大文件 + 跨 region)
        见 data_sync_tos.md §6
```

---

## 7. 必须知道的注意事项

1. **NFS 只 export `workspace/`, 不是整个 `ubuntu/`** — 这是 symlink trick (§4) 能工作的前提。改 export 范围前请评估对单机 ckpt 的影响。

2. **TOS AK/SK 硬编码在 `dataset/KAI0/{from_tos_file,to_tos}.py`** — AK 是子账号凭据, 不应外泄。Git 化前必须替换为 env var 读取。

3. **千万不要写 ckpt 到 `kai0/checkpoints/<config>/<exp>` 真实路径** — 那是 NFS 路径, 会占共享空间 + 拖慢其他机 I/O。一律走 symlink trick (§4) 或直接写 `/data/shared/ubuntu/local_ckpts/...`。

4. **代码/`config.py` 由 git main 管理 (1-min pull cron)** — uc01 cron `* * * * * /home/ubuntu/scripts/sync_kai0.sh <repo> <log>` 每分钟 `git fetch + reset --hard origin/main` 镜像 GitHub(pull-only)。所以**改 config 必须在 gf0 commit+push**, uc ~1 分钟内自动拿到; 不要直接在 uc 改代码(会被 reset 覆盖)。数据 (gitignore) 不受 reset 影响。

5. **uc01 是 SPOF** — NFS server + 数据集源都在 uc01。uc01 down 同时影响 uc02/03 的 NFS read。重启 uc01 前先停 uc02/03 上的训练。

6. **SSH 密码登录已禁用** (cloud-init 50-cloud-init.conf), 唯一登录 = pubkey。

---

## 8. 校验命令 (留作下次 verify)

```bash
# A. NFS server 配置
ssh uc01 "cat /etc/exports; showmount -e localhost"

# B. NFS client mount
for h in uc02 uc03; do
  ssh $h "findmnt -t nfs,nfs4"
done

# C. 跨机 inode 一致性验证 (代码/数据 真共享)
for h in uc01 uc02 uc03; do
  ssh $h "stat -c '$h: inode=%i %n' /data/shared/ubuntu/workspace/dataset/KAI0/from_tos_file.py"
done
# 期望: 3 行 inode 全相同

# D. local_ckpts symlink trick
for h in uc01 uc02 uc03; do
  ssh $h "
    ls -la /data/shared/ubuntu/workspace/deepdive_kai0/kai0/checkpoints  # 应是 symlink → /data/shared/ubuntu/local_ckpts
    findmnt -T /data/shared/ubuntu/local_ckpts                            # 应是本机 /dev/vdb (不是 NFS)
  "
done

# E. cluster_ckpt 旧路径状态 (应为废弃)
for h in uc01 uc02 uc03; do
  ssh $h "
    ls -la /data/cluster_ckpt 2>&1 | head -3
    findmnt /cluster_ckpt 2>&1 || echo '/cluster_ckpt NOT mounted'
  "
done

# F. 数据盘余量
for h in uc01 uc02 uc03; do
  ssh $h "df -h /data"
done
```

---

## 9. 与历史文档的差异速查

| 文档 | 历史描述 | 实测现状 | 建议修订 |
|---|---|---|---|
| `submission/uc_cluster_jobs.md §12.5` | 双 NFS 平面: 管理网 + RDMA `/data/cluster_ckpt` | 单 NFS 平面 (仅管理网); RDMA NFS 未启用; 3-host ckpt 走 NFS 内 `cluster_ckpts/` | 标注 §12.5 RDMA NFS 部分为"曾计划/未启用", 加新 §12.5b 描述实际 |
| `data_sync_tos.md §6.2/§6.3` | uc 拉到 `~/workspace/deepdive_kai0/kai0/data/Task_A/<dataset>/`; for-loop 各机拉 | 拉到 `dataset/KAI0/...`; 只 uc01 拉 + NFS 共享 | 已在本次修订一并改 |
| `storage_and_env.md §2.2 / §2.4` | `local_ckpts → /dev/vda2 (~290G)`; `/data/shared/dataset/...` | `local_ckpts → /dev/vdb (~1.7T avail)`; 数据在 `/data/shared/ubuntu/workspace/dataset/` | 已在本次修订一并改 |
| memory `reference_uc_cluster_nfs_layout` | "单机训练 ckpt 不上 NFS, `/data/cluster_ckpt` 各机各自本地, 集群训练 init ckpt 需手动 scp" | 单机 ckpt 走 symlink-本地 ✓; cluster_ckpt 路径已迁到 NFS 共享路径 | 更新 memory |
