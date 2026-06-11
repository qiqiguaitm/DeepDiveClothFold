# 数据同步与 Ckpt 部署 (TOS 枢纽)

> TOS 中心枢纽 / sim01 上传 / 训练服务器拉取 / 跨服务器 sync / ckpt 训练→sim01 部署工作流。
>
> **同 series**: `overview.md` / `storage_and_env.md` / `ssh_and_credentials.md` / `submission/`

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

### 6.2 训练服务器从 TOS 拉数据 (镜像消费) ⭐ (2026-05-28 修订)

各服务器 KAI0 数据路径与 TOS 路径**子路径一致** (TOS 端没有 `dataset/` 这层, 本地端 uc 为兼容历史有):

| 训练服务器 | KAI0 本地路径 (mirror) | 拉取命令 |
|---|---|---|
| **gf0** | `/vePFS/tim/data/KAI0/...` | `cd /vePFS/tim/data/KAI0 && tosutil cp -r tos://transfer-shanghai/KAI0/Task_A/<sub>/ ./Task_A/` |
| **gf3** | `/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/...` | `cd /vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data && tosutil cp -r tos://transfer-shanghai/KAI0/Task_A/<sub>/ ./Task_A/` |
| **uc01** (仅 uc01, 经 NFS 自动同步给 uc02/03) ⭐ | `/data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/Task_A/...`(2026-05-28 起,原 `dataset/` 已迁此) | 原始 base 拉到 `vis_base/`:`cd …/kai0/data/Task_A/vis_base && tosutil cp -r tos://transfer-shanghai/KAI0/Task_A/base/<date>-v2/ ./`;官方/构建集对应 `kai0_*` / `self_built/`。同步脚本 `from_tos_file.py/to_tos.py` 仍在 `dataset/KAI0/` |

> ⭐ **uc 集群只在 uc01 拉一次** — `/data/shared/ubuntu/workspace/` 是 uc01 export 的 NFS root (`10.60.0.0/16`, 走管理网 eth0), uc02/03 通过 NFSv4.1 自动看到同一份 (跨机 inode 一致, 2026-05-28 实测). **不要 for-loop 各机各拉一份** — 浪费 3× TOS 带宽, 还会因不同步导致训练读到不同内容。详见 `../../backup/uc_cluster_data_sharing_analysis.md` (uc 已停用)。

**关键: 路径前缀对齐**:
```
TOS:  tos://transfer-shanghai/KAI0/Task_A/base/2026-05-22-v2/
            │                       │
            └── 子路径与服务器本地  ─┴── 完全一致 (但 uc 本地多一层 dataset/)

gf0:  /vePFS/tim/data/KAI0/Task_A/base/2026-05-22-v2/
gf3:  /vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/base/2026-05-22-v2/
uc01: /data/shared/ubuntu/workspace/dataset/KAI0/Task_A/base/2026-05-22-v2/    ← uc02/03 经 NFS 自动可见
```

### 6.3 跨服务器 sync 工作流 (经 gf0 统一发起)

由于 gf0 是统一控制平面 (§5.6.c), TOS sync 命令也从 gf0 发起:

```bash
# gf0 → 火山 / uc 上的数据 sync 都通过 gf0 ssh + 本机 tosutil
ssh gf0 'bash -s' <<'EOF'
# 1. 通知 sim01 准备好数据后, gf0 触发各服务器拉取
SUB=base/2026-05-22-v2

# gf0 自己拉 (cnsh vepfs)
cd /vePFS/tim/data/KAI0/Task_A
tosutil cp -r tos://transfer-shanghai/KAI0/Task_A/$SUB ./

# gf3 拉 (cnbj vepfs)
ssh gf3 "cd /vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A && \
  tosutil cp -r tos://transfer-shanghai/KAI0/Task_A/$SUB ./"

# uc 集群只在 uc01 拉一次, uc02/03 经 NFS 自动同步 ⭐
ssh uc01 "cd /data/shared/ubuntu/workspace/dataset/KAI0/Task_A && \
  tosutil cp -r tos://transfer-shanghai/KAI0/Task_A/$SUB ./"

# 验证 (各服务器抽样查目录大小 + ep 数)
for h in gf0 gf3 uc01; do
  P=$(case $h in
    gf0)  echo /vePFS/tim/data/KAI0/Task_A/$SUB;;
    gf3)  echo /vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/$SUB;;
    uc01) echo /data/shared/ubuntu/workspace/dataset/KAI0/Task_A/$SUB;;
  esac)
  echo -n "$h: "; [ "$h" = "gf0" ] && du -sh "$P" 2>/dev/null || \
    ssh $h "du -sh $P 2>/dev/null"
done

# 确认 uc02/03 也看到了 (经 NFS, 一般 <1s 同步)
for h in uc02 uc03; do
  ssh $h "ls /data/shared/ubuntu/workspace/dataset/KAI0/Task_A/$SUB | head -3"
done
EOF
```

> **uc 端 canonical 同步脚本** (内含 AK/SK 硬编码, 不读 env): `/data/shared/ubuntu/workspace/dataset/KAI0/{from_tos_file.py, to_tos.py, to_tos_file.py}`. 通常 `tosutil cp -r` 已够用; Python 脚本仅在需要调分片并发 (task_num/part_size) 时用。

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

#### 6.5.A 顶层 prefix (按用途分)

```
tos://transfer-shanghai/
├── KAI0/                         ⭐ 主数据 prefix (大写, canonical)
├── kai0/checkpoints/             ckpt 落档 (小写)
├── ckpt/                         早期 ckpt 中转
├── shared_ckpt/                  跨用户共享 ckpt (XVLA / mixed_1_clean 等)
├── shared_sz/                    深圳团队共享
├── Task_A_kai_official.tar       kai0_base+dagger+advantage 官方数据归档 (2026-05-28 起; 原散落的 kai_official_relay/ 已删除合并为此单一 tar)
├── from_uc01/                    uc01 → 任意机器中转 (含 gf3 venv tar)
├── tim/                          用户私有 (临时打包 / 中转)
├── migrate_js/                   js 集群停用前迁移备份
├── migrate_pure200/              early pure200 迁移备份
├── backup_uc_reinstall_20260516/ uc 重装前完整备份
├── graspforge/                   另一项目 (GraspForge)
├── sam3d/                        另一项目 (SAM3D)
└── test_xvla_speed/              性能测试
```

#### 6.5.B KAI0/ 主数据结构 (canonical)

```
tos://transfer-shanghai/KAI0/
├── Task_A/                       ⭐ 主任务 (双臂叠衣)
│   ├── base/
│   │   ├── 2026-04-23-v2/        每个 date dir 结构:
│   │   ├── 2026-04-24-v2/          ├── data/chunk-000/episode_*.parquet
│   │   ├── 2026-04-25-v2/          ├── videos/chunk-000/
│   │   ├── 2026-04-28-v2/          │   ├── hand_left/episode_*.mp4
│   │   ├── 2026-04-29-v2/          │   ├── hand_right/episode_*.mp4
│   │   ├── 2026-04-30-v2/          │   ├── top_head/episode_*.mp4
│   │   ├── 2026-05-06-v2/          │   └── top_head_depth/episode_*.zarr/
│   │   ├── 2026-05-07-v2/          └── meta/{info.json,episodes.jsonl,tasks.jsonl}
│   │   ├── 2026-05-08-v2/
│   │   ├── 2026-05-09-v2/        注: 4 月之前的 date 无 top_head_depth (那时未录 depth);
│   │   ├── 2026-05-16-v2/             5-06 起所有 date 含完整 depth zarr.
│   │   ├── 2026-05-18-v2/        … (append-only, 持续追加)
│   │   ├── 2026-05-22-v2/
│   │   ├── 2026-05-26-v2/
│   │   ├── 2026-05-27-v2/
│   │   ├── 2026-05-28-v2/        ← 当前最新 (20 date dirs, 截至 2026-06-03)
│   │   ├── README.md             ⭐ 数据描述 (per-date 场景表 / 质量评估)
│   │   ├── analysis/             ⭐ 质量分析 csv (Class C 黑名单 / end-snap 清单 等)
│   │   └── kai0_official_base/   官方 base 副本 (3055 ep)
│   ├── autonomy/
│   │   └── <date-v2>/            自主推理录制数据 (同 base 结构)
│   └── dagger/                   DAgger 接管数据 (同 base 结构); 本地 → vis_dagger/v2/ 自动同步 (见 §6.9)
│       ├── 2026-05-29-v2/ (64ep)  2026-06-01-v2/ (32ep)
│       └── 2026-06-02-v2/ (71ep)  2026-06-03-v2/ (24ep)   ← 4 dates, 截至 2026-06-03
│
├── Task_E/  base/<date-v2>/      扶起倒箱
├── Task_H/  base/<date-v2>/      
├── Task_HP/ base/<date-v2>/      
├── Task_P/  base/<date-v2>/      抓放盒子
├── Task_PP/ base/<date-v2>/      抓放 (2 dates: 05-09 202ep / 05-25 204ep, 19GB; 2026-06-02 拉到 gf0 本地)
└── Task_PS/ base/<date-v2>/      
```

#### 6.5.C "-v2" 后缀约定 (2026-05-11 起)

- `<YYYY-MM-DD>-v2` = canonical 命名 (本地权威格式, 全 TOS 路径只用此格式)
- 没有 -v2 后缀的 legacy 目录均为 deprecated, 不应作为新数据落点
- 一个 date 在 TOS 同时只能以 `-v2` 形式存在; 数据修改 → 升 -v3 / -v4, 同名永不内容变更 (append-only)

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

### 6.8 gf0 `vis_base` 自动**完整增量**同步 (cron + tosutil, 每小时) ⭐ (2026-05-28; 2026-06-02 v2/v3 重构)

**目的**: 让 gf0 上的 `kai0/data/Task_A/vis_base/v2/`(vis_v2_* / A_0423_0527 等数据集的 build 源)持续与 TOS 保持最新, 无需手动拉。

> 🔀 **2026-06-02 目录重构 — v2/v3 分层** (三机 gf0 / uc-NFS / gf3 已对齐):
> ```
> vis_base/
> ├── v2/  <date>-v2 ×20   # 原始采集 (含 depth), sync DST, build 源
> └── v3/  <date>-v3 ×20   # 裁投放 (no-release: 裁掉每 ep 开头投放等待静止段), 无 depth, 1956 ep
> ```
> - **v3 = v2 经 `build_no_release.py --per-date` 裁投放**: 逐日期裁掉开头静止段 (motion-onset 检测), 保留原 ep 编号, drop depth (RGB-only)。机理见 [`../../training/history/experiments/data_root_cause_probe_results.md`](../../training/history/experiments/data_root_cause_probe_results.md) §4 (policy idling: 演示停顿被 BC 模仿致真机走停)。裁剪比例: 早期日期 ~2% (节奏紧凑), 后期 ~7.5% (投放等待长)。
> - **sync DST 已从 `vis_base/` 改为 `vis_base/v2/`** —— 历史 build 脚本 (build_vis_v2_full / A_0423_0527 等) 的 SRC_ROOT 也加 `/v2`。
> - ⚠️ **14k+ self_built 软链** (vis_v2_merged/full 等指向 vis_base 绝对路径) 在 mv 时已批量重指到 `/v2/`。

> 🔒 **重构/迁移 SOP — 必须先停 sync (2026-06-03 复盘教训)**: 任何动 vis_base 目录结构 (如本次 v2/v3 分层) 的操作, **必须严格按此顺序**, 否则迁移窗口内 cron 会用旧 DST 拉数据造成残留:
> ```
> ① 停 sync cron        (注释/移除 crontab 行; 各机的 cron 在哪台都要停 — gf0 本机 / uc 在 uc02 / gf3 本机 root)
> ② mv 数据 → v2/        (含批量修复 self_built 指向 vis_base 的绝对路径软链)
> ③ build v3            (build_no_release.py --per-date all, 从 v2 裁投放)
> ④ 改同步脚本 DST → v2  (+ 护栏: DST 必须含 /v2; 自愈清理根扁平残留)
> ⑤ 验证                (帧对齐 / 软链 0 断裂 / 顶层只剩 v2 v3 无扁平)
> ⑥ 重启 sync cron      (此时 DST 已对, 安全)
> ```
> **本次踩坑**: gf0 做对了 (①先停 cron); 但 gf3/uc 漏了①, 迁移时它们的 cron 用旧 DST=`vis_base` 根拉出 **20 个扁平 `<date>-v2` ×15G 重复残留** (无 depth/无软链引用=纯重复), 事后手动删除 + 给 sync 脚本加护栏 (DST 校验 + 根扁平自愈清理, commit `772152b`)。**核心: ①和⑥把整个迁移包在 "sync 停止" 窗口内。**

| 项 | 值 |
|---|---|
| 机器 | **gf0 + uc-NFS(uc01 cron, uc02/03 共享可见) + gf3**(各机本机有 `~/tosutil` + 凭据) |
| 脚本 | `train_scripts/kai/data/sync_vis_base_from_tos.sh` (host-aware, 三机通用) |
| 频率 (base) | crontab `0 * * * *`(gf0) / `17 * * * *`(uc01) / `37 * * * *`(gf3) — 三机错峰 |
| 传输工具 | **tosutil `cp -r -u`**(原生 TOS 客户端, 多线程, 不依赖 FUSE 挂载) |
| 源 → 目标 | 逐日期 `tos://transfer-shanghai/KAI0/Task_A/base/<date>-v2/` → `…/vis_base/v2/<date>-v2/` |
| 排除 | **`-exclude='*top_head_depth*'`**(depth zarr, 见下) |
| 日志 | `logs/vis_base_sync.log`(>5MB 自动轮转) |

**同步策略**: **完整增量** —— 每次遍历 TOS 上**所有** `<date>-v2`(不只新日期), 用 `tosutil cp -r -u` 增量拉取:
- `-u` 按 **size/crc** 跳过未变文件(**非 mtime**, 实测早期日期 760/762 skip、0.24s), 只下载新/变更对象, **从不删除本地** → 保护 vis_v2_*/A_0423_0527 指向 vis_base 的软链。
- 既能接住"旧日期后续追加 episode"的更新, 也能拉全新日期。
- `flock -n` 防重叠(上次未跑完则跳过)。

> **为何排除 depth zarr (`top_head_depth`)**: 深度图存为 zarr, 单日期约 **18.5 万个小 chunk 文件** × 13 个含 depth 的日期 ≈ **240 万对象**。若全量同步, 每轮要比对 240 万对象 → 20-30 分钟, 不适合每小时。而 depth **当前不被 vis_v2_* 训练消费**(只用 RGB: top_head/hand_left/hand_right)。`-exclude='*top_head_depth*'` 后只同步 RGB + parquet + meta(数万对象)。
> - **本地已有的 depth 不会被删**(cp 从不删本地), 只是不再逐轮比对/更新。
> - 若将来需要 depth: 手动 `tosutil cp -r -u .../base/<date>/videos/chunk-*/top_head_depth/ <DST>/<date>/videos/...`, 或单独做低频(每日)depth 同步。
> - 即便排除 depth, tosutil 仍需列举日期前缀下的对象, 含 depth 日期单轮仍偏慢, 整轮约数分钟级(可接受, < 1h 间隔, flock 兜底)。

> **前置归一化 (2026-05-28 一次性完成)**: 早期 10 个日期 (04-23~05-09) 的本地视频目录原为 `observation.images.*` 长名, 而 TOS 是短名 `top_head` —— 已把这 30 个目录改回短名 + retarget 了 vis_v2_*/A_0423_0527 中 7881 条相关软链。归一化后 vis_base 与 TOS 结构一致, 完整 `cp -r -u` 不会产生重复(否则会两套并存)。
>
> **tosutil 与 rsync-over-FUSE 之别**: tosutil 走 TOS API 原生多线程, 不依赖 `/transfer-shanghai` FUSE 挂载在位, 更快更稳。注意 **tosutil 无 `sync` 子命令**(官方文档确认), 用 `cp -r -u` 实现增量。路径映射: `cp -r .../base/<date>/ vis_base/` 会把末级 `<date>` 落在 `vis_base/<date>/`(实测)。
>
> ⚠️ **前置依赖: cron 守护进程必须在运行**。gf0 重装/重启后需 `sudo service cron start`(需 root)。验证: `pgrep -x cron && crontab -l`。tosutil 配置 `~/.tosutilconfig` 内 AK/SK + 路径(从 uc01 拷来后已把 `/home/ubuntu` 改为 `/home/tim`)。

### 6.9 `vis_dagger` 自动同步 (cron + tosutil, 每小时) ⭐ (2026-06-03)

**目的**: DAgger 接管数据 (`TOS Task_A/dagger/`) 持续同步到本地, 与 `vis_base` 同款机制。

**结构 (与 vis_base/v2 对齐)**: TOS dagger 扁平 `dagger/<date>-v2`, 本地按 **v2 数据版本命名空间** 组织:
```
vis_dagger/
└── v2/  <date>-v2    # sync DST; TOS dagger 各日期 → vis_dagger/v2/<date>-v2/
```

| 项 | 值 |
|---|---|
| 机器 | **gf0 + gf3**(2026-06-03; **uc 暂未处理**) |
| 脚本 | `train_scripts/kai/data/sync_vis_dagger_from_tos.sh` (host-aware, 由 base 脚本派生) |
| 频率 (dagger) | crontab `7 * * * *`(gf0) / `47 * * * *`(gf3) — 与各机 base cron 错峰 |
| 源 → 目标 | `tos://transfer-shanghai/KAI0/Task_A/dagger/<date>-v2/` → `…/vis_dagger/v2/<date>-v2/` |
| 策略 | 同 base: `cp -r -u` 完整增量 (size/crc, 只增不删) + `-exclude='*top_head_depth*'` + `--mirror` 手动传播删除 |
| host 探测 | 用 `kai0/data/Task_A/vis_base/v2` 存在性判真实 KAI0 根 (gf3 `/vePFS/tim` 空壳排除) |
| 日志 | `logs/vis_dagger_sync.log` |

**当前 dagger 日期** (2026-06-03, 4 日期 / 2.5G, 三处一致 gf0+gf3+TOS): `2026-05-29-v2` (64 ep) / `2026-06-01-v2` (32) / `2026-06-02-v2` (71) / `2026-06-03-v2` (24)。

> **2026-06-03 重构 SOP** (同 §6.8 v2/v3 教训): 建 `vis_dagger/v2/` → mv 现有 `<date>-v2` 进去 → 改脚本 DST `vis_dagger` → `vis_dagger/v2` (commit `653c655`) → 跑一次追平 (gf0/gf3 各补 06-02 +25 ep & 全新 06-03 24 ep) → 装 cron。**uc 待补**: 同流程 (建 v2 + 装 cron `27 * * * *`)。

---


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

