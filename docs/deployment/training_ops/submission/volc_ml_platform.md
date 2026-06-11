# Volc ML Platform 训练任务提交

> Volc ML Platform YAML + SDK 提交模式 / 16-卡 H20 集群训练 YAML 配置 / region & queue mapping / image_cr。
>
> **同 submission 子目录**:
> - `gf0_control_plane.md` — 经 gf0 作为统一控制平面提任务
> - [`../../../backup/uc_cluster_jobs.md`](../../../backup/uc_cluster_jobs.md) — uc01-03 (已停用)
>
> **上级**: `../overview.md` (服务器全景 + 单机直接启动模板)

---

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

**封装好的 helper** (用 submit_yaml.py CLI 形式, 处理 dry-run + 错误): `train_scripts/kai/volc/submit_yaml.py`。

> **Queue ID + 容量速查见 §5.6.c.2** (附实测可用 / 已用 GPU 数)。

### 5.6.b 16-卡 H20 集群训练 YAML 配置要点

> ℹ️ **集群训练当前走 §5.6.c (vsubmit 一键提交)**。本节列出 YAML 中针对 16-GPU H20 集群的关键字段差异 (与 8 GPU 单节点对比), 用于自定义 YAML 编写。

模板: `train_scripts/kai/volc/gf3_cluster_smoke_16gpu.yaml` (2 节点 × 8 H20 = 16 GPU, FSDP=16)。

```yaml
# 实测可工作 (2026-05-21 X-VLA Stage 1 76d44):
ImageUrl: "dvs-cr-cn-beijing.cr.volces.com/vis_robot/kai:kai0-gf1"     # ⭐ kai0 标准训练镜像 (vis_robot CR)
# 备选 (smoke / grasp-h2r 任务):
# ImageUrl: "visincept-cn-beijing.cr.volces.com/grasp/h2r:1.0"
ResourceQueueName: "Robot-North-H20"                                   # auto: cn-beijing / cn-beijing-e
TaskRoleSpecs:
  - RoleName: "worker"
    RoleReplicas: 2
    Flavor: "ml.hpcpni3ln.45xlarge"                                    # 8× H20-SXM5-96GB, RDMA
Storages:
  - Type: "Vepfs"
    VepfsId: "vepfs-cnbj875793a96d6b"                                  # 华北 vePFS, 与 gf3 共享
    MountPath: "/vePFS-North-E/vis_robot"                              # ⚠️ 必须配 SubPath=/vis_robot
    SubPath: "/vis_robot"                                              # IAM 限定到 /vis_robot 子路径, 否则 AccessDenied
```

> ⚠️ **镜像 URL 易错点 (2026-05-21 踩坑)**:
> - 正确: `dvs-cr-cn-**beijing**.cr.volces.com` (beijing 拼写完整)
> - 错误: `dvs-cr-cn-**bejing**.cr.volces.com` (少一个 i) → DNS 不解析, 任务卡 Deploying 25+ 分钟无报错, 直到自动失败
> - 通过 `curl -sI https://dvs-cr-cn-beijing.cr.volces.com/v2/` 验证 — 应返回 401 Unauthorized (说明 endpoint 存在)
>
> ⚠️ **vePFS 权限 (2026-05-21 踩坑)**:
> - cn-beijing 队列 IAM 用户对 vepfs-cnbj 根目录无 RDWR 权限, 必须 `SubPath: "/vis_robot"` 限定到用户拥有的子目录
> - 不设 SubPath → 提交立刻返回 `403 AccessDenied: You are not authorized [dir: /, mode: RDWR]`
> - MountPath 也要相应改为 `/vePFS-North-E/vis_robot` (而非 `/vePFS-North-E`), 这样 entrypoint 中的路径 `/vePFS-North-E/vis_robot/workspace/...` 才能正确映射

提交:
```bash
source ~/.volc_creds
python train_scripts/kai/volc/submit_yaml.py train_scripts/kai/volc/gf3_cluster_smoke_16gpu.yaml
# 或 dry-run:
python train_scripts/kai/volc/submit_yaml.py train_scripts/kai/volc/gf3_cluster_smoke_16gpu.yaml --dry-run
```

**vePFS 与 .venv self-containment (关键)**: gf3 上的 `.venv` 是经 sed 重写并把 uv-managed Python 一并搬到 vePFS 的版本 (见 §3.1 注解), volc 集群任一新节点 mount `vepfs-cnbj875793a96d6b` 后 `source .venv/bin/activate` 直接可用, **无需在每节点重装**。

**JAX 多机协调**: entrypoint 用 volc 提供的 `MLP_WORKER_0_HOST` / `MLP_WORKER_NUM` / `MLP_ROLE_INDEX` 设 `JAX_COORDINATOR_ADDRESS=$MLP_WORKER_0_HOST:15830` (port 15830, **不要用 `MLP_WORKER_0_PORT=2222` — 那是 SSH 端口冲突**)。

**注意事项 (同 5.6 通用):**

- ckpt 写入 `/vePFS-North-E/<...>/checkpoints/<config>/<exp_name>/<step>/`, vePFS 本地立即可见
- multi-host orbax 保存可能 race, 用 `--overwrite` 或 node-0 预清理
- 日志走 vePFS 共享, `logs/cluster_smoke_*_node${MLP_ROLE_INDEX}.log`, gf3 上 tail 即可
- 任务列表 / GUI: `https://console.volcengine.com/ml-platform/region:ml-platform+cn-beijing/task`

---

## Volc 特有踩坑 (2026-06 实战归纳)

> 跨集群共性坑 (norm_stats / 绝对 repo_id / init 完整性 / eval prompt 等) 见 [`training_pitfalls_common.md`](training_pitfalls_common.md)。本段只列 **Volc ML Platform 特有**的。

### V1. 多卡任务卡 Deploying 0 进度 = 资源被占, 不是配置/镜像 ⚠️ (最常误判)

- 16 卡 (2 节点) 是 **gang-scheduling**: 必须**同时**凑齐 2 个干净的整节点。queue "free=28" 可能是**碎片化**散在多机, 没有 2 个空 8-GPU 节点 → 任务 **Deploying 卡死, `UpdateTime` 冻结在创建时刻, 0/2 实例, 无报错**。
- **先查是谁占着**: `list_jobs` 看活跃任务 (含**别的用户**的, 如 libero/GigaWorld/act-mvsw)。再查 `get_resource_queue` 的 `free=Capability-Allocated`。
- 本轮实测: cnbj 从 free=28 被别人任务占到 **free=6** → 16 卡 (需 32... 实为 2×8=16) 调度不出。**别人占满时 16 卡等不到, 单节点 8 卡 (只需 1 节点) 也可能不够 → 迁 uc**。
- ⚠️ 单节点秒起 ≠ 多节点能起: 单节点只要 1 个 8-GPU 节点; 多节点要 2 个整节点 + gang 同步。

### V2. 镜像缓存决定部署速度 (换镜像有代价) ⚠️

- **kai0-gf1 镜像** (`dvs-cr-cn-beijing.cr.volces.com/vis_robot/kai:kai0-gf1`) — ⭐ **cnbj(beijing)标准镜像,以后 beijing 任务都用它**: cnbj 节点**已缓存** → 部署秒级 (canary submit→Running **20s**)。**2026-06-04 实测 Exp-B 多机过 Step 0**——此前"distributed.initialize Logging error 崩"的结论**未复现,作废**。
  - ✅ 部署本身干净(秒级)。注意: 换了快镜像后这些任务**第一次真正跑到 data-load**, 暴露了一个**与镜像无关的数据 bug**——`chunks_size` 坑(>1000 ep 数据集单 chunk-000 布局, 见 [pitfalls §3](training_pitfalls_common.md))。别把它误判成镜像/缓存问题(我一开始按"Cloudfs 冷缓存 race"加 cache-warm 预热, **无效**, 真因是 chunks_size)。
- **h2r 镜像** (`visincept-cn-beijing.../grasp/h2r:1.0`): cnbj 节点**没缓存** → 冷拉把 **2 节点 gang staging wedge 死**(2026-06-04 卡 1h22m 不报错不前进,见 [pitfalls §10](training_pitfalls_common.md))。**cnbj 弃用**。
- → **beijing: `kai:kai0-gf1` + cache-warm 预热; shanghai(cnsh): 仍用 `visincept-cn-shanghai.../grasp/h2r:1.0`**(cn-shanghai registry 该区已缓存,单机几秒起,无需换)。kai0-gf1 是 **cn-beijing registry 专用**,别拿到 cnsh 用(跨区拉取慢)。
- 镜像 URL 拼写: `cn-beijing` 别写成 `bejing` (DNS 不解析 → 卡 Deploying 25min+ 自动失败)。`curl -sI https://<cr>/v2/` 应返回 401 = endpoint 存在。

### V3. 提交命令的坑

- **`submit_yaml.py` 用 venv python** (`kai0/.venv/bin/python`) — 系统 python 缺 `yaml` 模块。
- **凭证**: `VOLC_AK` + `VOLC_SK` env (inline 传, 别写文件/log)。
- **`VOLC_REGION` 必须设** — submit_yaml.py 默认 `cn-shanghai`。提 **cnbj 任务必须 `VOLC_REGION=cn-beijing`**, 否则提到错的区。
- cnbj vePFS 必须 `SubPath: "/vis_robot"` (IAM 限定), 否则 403 AccessDenied。

### V4. API 字段 / 状态查询

- job 状态在 **`Status.State`** (不是顶层 `State`); `Status.Message` 有人话原因 ("worker-1 实例异常结束" / "2 个实例皆在运行中" / "0/2 完成部署")。
- `list_jobs` 用 **`Items`** key (非 `List`); 有效 state: Creating/Waiting/Queueing/Deploying/Running/Stopping/Completed/Failed/Stopped。
- SDK 5.0.27 deserializer 坏 → 脚本顶部打 monkey-patch (见 §5.6.a 示例)。
- 失败但 log 截断在 "Logging error" / 看不到真错 → 查 `get_job` 的 `Status.Message` + 两个 node log (`*_node0.log` / `*_node1.log`)。

### V5. 多机 orbax + JAX 协调

- multi-host orbax 保存 race → `sync_global_devices ... mismatch` → 用 `--overwrite` + 清 stale ckpt 目录。
- JAX coordinator: `JAX_COORDINATOR_ADDRESS=$MLP_WORKER_0_HOST:15830` (port 15830, 不用 2222=SSH)。Volc 自动给 MLP_* env (需 `Framework: PyTorch`)。

