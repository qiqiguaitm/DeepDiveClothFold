# gf0 作为所有训练资源的统一控制平面

> 2026-05-21 起推荐做法: 经 gf0 控制 Volc + uc01/02/03 + Robot-North-H20 + robot-task 全部训练任务。包含 mlp CLI 速查 / queue mapping / SDK 自动提交。
>
> **同 submission 子目录**:
> - `volc_ml_platform.md` — Volc Platform 基础 YAML/SDK
> - [`../../../backup/uc_cluster_jobs.md`](../../../backup/uc_cluster_jobs.md) — uc01-03 (已停用)
>
> **上级**: `../overview.md` / `../ssh_and_credentials.md` (gf0 SSH 设置)

---

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
3. ✅ **已有 submit_yaml.py 基础设施** (`train_scripts/kai/volc/submit_yaml.py`)
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

#### 5.6.c.2 可用队列 (2026-05-22 实测)

> 通过 `ListResourceQueues` 跨 region 查询 + 个人 quota 验证, 当前账号可用 **2 个队列**:

| Region | Queue Name | Queue ID | 节点 × 类型 | Total GPU | Allocated | Free | 单节点 GPU/CPU/MEM | RDMA |
|---|---|---|---|---:|---:|---:|---|---|
| **cn-beijing** | **Robot-North-H20** ⭐ | `q-20260516104642-khch9` | **7 × ml.hpcpni3ln.45xlarge** | **56 H20** | 25 | **31** | 8 × H20-SXM5-96GB / 180 vCPU / 1960 GiB | 4× |
| **cn-shanghai** | **robot-task** ⭐ | `q-20251204185107-fvnpx` | 1 × ml.pni2.14xlarge + 3 × ml.hpcpni2.28xlarge | **28 A100-80G** | 8 | **20** | 8 × Tesla-A100-80G / 112 vCPU / 1715 GiB | varies |

**核心区别**:
- **Robot-North-H20** (cn-beijing): 新, 大 (56 H20), **多机集群训练首选**, 走 `vepfs-cnbj875793a96d6b` (与 gf3 共享 FS), zone `cn-beijing-e`
- **robot-task** (cn-shanghai): 旧, 中 (28 A100-80G), 2026-05-22 起**已大幅空闲** (20/28 free), 走 `vepfs-cnsh075262e1f815` (与 gf0 共享 FS), zone `cn-shanghai-a`

**当前可启动并发**:
- Beijing: 2 × 16 GPU job 或 1 × 32 GPU job
- Shanghai: 2 × 8 GPU job 或 1 × 16 GPU job ⭐ 新增可用空间

**跨 region 共享性**: 数据需双地各放一份。kai0 base+dagger 已就位 gf0 (cnsh) + gf3 (cnbj); vis_v2_merged 已就位 uc01 NFS + gf0 (cnsh), gf3 (cnbj) 2026-05-22 后补; XVLA-Soft-Fold 多地副本详见 `xvla/data/xvla_soft_fold/README.md`。

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

> 提交脚本统一放在 `/vePFS/tim/workspace/deepdive_kai0/train_scripts/kai/volc/submit_yaml.py` (gf0 vePFS), 已 git 化随代码同步。

> 实现已绕过 SDK 5.0.27 `KeyError: '.models'` 反序列化 bug(直接 OpenAPI POST)。submit_yaml.py 解析 YAML → 调火山 `CreateJob`,按 `ResourceQueueName` 自动填 queue id / region / zone / vepfs / 默认镜像(下表)。

**Queue → region/zone/vepfs/默认镜像 映射** (submit_yaml.py 内置, 供查阅):

| Queue | id | region / zone | vepfs (mount, SubPath) | 默认镜像 image_cr |
|---|---|---|---|---|
| **Robot-North-H20** | `q-20260516104642-khch9` | cn-beijing / cn-beijing-e | `vepfs-cnbj875793a96d6b` (`/vePFS-North-E/vis_robot`, SubPath `/vis_robot`) | `dvs-cr-cn-beijing.cr.volces.com/vis_robot/kai:kai0-gf1` |
| **robot-task** | `q-20251204185107-fvnpx` | cn-shanghai / cn-shanghai-a | `vepfs-cnsh075262e1f815` (`/vePFS`) | `visincept-cn-shanghai.cr.volces.com/grasp/h2r:1.0` |
| **Robot-East-H20** | `q-20260516104437-2ml4v` | cn-shanghai (gf0 所在华东 H20) | — | — |

> **镜像规则 (2026-06-04 起)**: **beijing(Robot-North-H20)统一用 `dvs-cr-cn-beijing.cr.volces.com/vis_robot/kai:kai0-gf1`**(该区已缓存→秒级部署;旧 `visincept-cn-beijing.../grasp/h2r:1.0` 没缓存→冷拉 wedge staging,弃用)。**shanghai(robot-task)仍用 `visincept-cn-shanghai.cr.volces.com/grasp/h2r:1.0`**(该区已缓存)。kai0-gf1 是 cn-beijing registry 专用,勿跨区。(>1000 ep 数据集另见 chunks_size 坑 [pitfalls §3](training_pitfalls_common.md)。)

使用方式 (在 gf0 上):
```bash
ssh gf0
# AK/SK 已 export (在 ~/.bashrc 中 source ~/.volc/credentials parse)
cd /vePFS/tim/workspace/deepdive_kai0
python train_scripts/kai/volc/submit_yaml.py train_scripts/kai/volc/<your_task>.yaml
```

本地一键提交 alias (`~/.bashrc` on laptop):
```bash
alias vsubmit='ssh gf0 "cd /vePFS/tim/workspace/deepdive_kai0 && python train_scripts/kai/volc/submit_yaml.py"'
# 用法: vsubmit train_scripts/kai/volc/x1_delta_joint_16gpu.yaml
```

#### 5.6.c.5 经 gf0 状态查询 (Python, 无 SDK bug)

脚本(已抽离): **`train_scripts/kai/volc/volc_job_status.py`** — `get_svc(region)` 用 `volcengine.base.Service` 直调 `ListResourceQueues`/`ListJobs`/`GetJob`(绕开 SDK bug)。需 `VOLC_AK`/`VOLC_SK`。`python volc_job_status.py` 默认查 Robot-North-H20 running。

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
rsync -av /data/tim/data/Task_A/self_built/mix_dataset/ \
  gf0:/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/mix_dataset/

# 验证 volc 上看见
ssh gf0 "ls /vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/mix_dataset/ | head"
```

**(B) 推送数据到 Robot-North-H20 (gf3/vePFS-cnbj)**:
```bash
# 本地 (or uc02) → gf3 vePFS
rsync -av --info=progress2 /data/tim/data/Task_A/self_built/mix_dataset/ \
  gf3:/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/self_built/mix_dataset/

# 验证 volc 上看见
ssh gf3 "ls /vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/self_built/mix_dataset/ | head"
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

**job 状态 dashboard** (gf0 + cron, 可选): 脚本(已抽离): **`train_scripts/kai/volc/dashboard.sh`** — gf0 上 nohup 跑, 5min 轮询 Running/Queueing 写 `logs/volc_dashboard.txt`。本地一键看:
```bash
ssh gf0 "cat /vePFS/tim/workspace/deepdive_kai0/logs/volc_dashboard.txt"
# 或 alias vdash='ssh gf0 cat /vePFS/.../volc_dashboard.txt'
```

---

