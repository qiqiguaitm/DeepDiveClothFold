# 基础设施:以 vePFS 环境为中心(而非服务器)

> **2026-07-21 定。** 单一事实源,讲清一个核心抽象转变:
> **服务器只是通道,真正的操作对象是 vePFS 环境。**
>
> 配套:环境选择见 [`../lmvla/docs/ENV_SELECTION_RULES.md`](../lmvla/docs/ENV_SELECTION_RULES.md);
> 跨集群指纹见 [`../lmvla/docs/ENVIRONMENTS_map_2026-07-20.md`](../lmvla/docs/ENVIRONMENTS_map_2026-07-20.md);
> gf0 控制平面细节见 [`deployment/training_ops/submission/gf0_control_plane.md`](deployment/training_ops/submission/gf0_control_plane.md)。

---

## 0. 核心抽象:两个 vePFS 环境,服务器是接入通道

**决定实验行为的不是"哪台机器",而是"挂了哪个 vePFS"。** 机器只是让你能接触到某个 vePFS 的通道。

| vePFS 环境 | VepfsId | 训练资源(火山队列) | 当前接入通道(机器) |
|---|---|---|---|
| **cnsh(上海)** | `vepfs-cnsh075262e1f815` | `robot-task`(A100) | **gf0**(本机 dev,2×A100) |
| **cnbj(北京 / North-E)** | `vepfs-cnbj875793a96d6b` | `Robot-North-H20`(H20) | **gsy**(提交节点,自身无 GPU) |

**关键推论:**
- gf0 挂载点 `/vePFS` = cnsh 环境;gsy/North-E 挂载点 `/vePFS-North-E/vis_robot` = cnbj 环境。
- "在 gf0 上跑" ≠ "gf0 的能力",而是"操作 cnsh 环境"。换一台挂了同一 cnsh vePFS 的机器,行为应完全一致。
- 训练**跑在火山队列分配的计算节点**上,gf0/gsy 只负责提交和监控 —— 它们是**控制通道**,不是算力。
- 因此**代码、数据、环境都存在 vePFS 上**,机器可替换。gsy 是容器(重启即重建),更印证:持久的东西必须在 vePFS。

```
        [ 通道机器 ]              [ vePFS 环境 ]           [ 算力 ]
  gf0  ──挂载 /vePFS──────────▶ cnsh 环境 ──提交──▶ robot-task (A100)
  gsy  ──挂载 /vePFS-North-E──▶ cnbj 环境 ──提交──▶ Robot-North-H20 (H20)
        (可替换的接入点)      (真正的操作对象)    (队列分配的节点)
```

---

## 1. 代码同步规范(以 vePFS 为单位,不是以机器)

**权威源 = GitHub(`qiqiguaitm/deepdive_clothfold`,原名 deepdive_kai0)。** 两个 vePFS 环境各持有一份 checkout,都从 GitHub 拉取,**不做机器间直接拷贝**(除非 GitHub 不含目标内容,见下)。

### 1.1 目录结构(两环境必须一致)

```
<vePFS>/…/deepdive_kai0/
├── lmvla/lawam    ← submodule, 纯净 RLinf/LaWAM@bd4a363(git 追踪)
├── lmvla/lmwam    ← 我们对 lawam 的修改层(patches+adapter, 71K)
├── lmvla/{crave,lmwm}  ← 当前布局(旧顶层 crave//lmwm/ 已废弃)
└── kai0/.venv     ← VLA 主力环境(见 §2)
```

### 1.2 同步流向

```
  gf0(cnsh checkout)  ──git push──▶  GitHub  ──cron 拉取──▶  North-E(cnbj checkout)
```

- **gf0 → GitHub**:人工 `git push`(含凭证扫描)。
- **GitHub → North-E**:cron 自动同步(`train_scripts/kai/volc/cron_sync.sh`),**带闸门**:
  队列有 Running 任务 / 有已跟踪本地改动 → 跳过,不在实验中途换代码。
- **`lmvla/lawam` 是 submodule**:同步必须 `--recurse-submodules`,且 `GIT_LFS_SKIP_SMUDGE=1`
  (North-E 曾有 443G LFS 遗留缓存,已清)。

### 1.3 gitignore 掉的内容不经 git 同步(必须知道)

以下**不在 GitHub 上**,换环境靠文件传输或重新生成,不是"pull 就有":
- **数据/产物**:`results/`、`dataset/`、`lmwm/outputs/`、`weights/`、`ckpts_dl/`(GB~TB 级)。
- ~~`lmvla/lawam/` 曾被 gitignore 排除~~ → 已改为 submodule,现受控。
- 每个 vePFS 环境的数据是**独立的**:cnsh 的 `results/` ≠ cnbj 的 `results/`,不自动同步。

### 1.4 跨 vePFS 传输(GitHub 到不了的大文件)

gsy **无 rsync**;TOS multipart 会 write timeout。用字节级续传:
```bash
tail -c +$((已传字节+1)) 本地文件 | ssh 远端 'cat >> 远端文件'   # 只依赖 cat/tail, ~5.8MB/s
```
传后 md5 两端校验。ckpt 跨环境必须带 `config.yaml` + `dataset_statistics.json`(`read_mode_config` 断言,**不是 config.json**)。

---

## 2. 环境搭建一致性要求

**目标:同一段代码在任一 vePFS 环境行为一致。** 环境按用途分,不按机器分(详表见 `ENV_SELECTION_RULES.md`):

| 用途 | 环境 | 两 vePFS 一致性 |
|---|---|---|
| VLA 训练/评测 | `kai0/.venv`(py3.12 / torch2.6 / tf5.13) | ✅ **逐项相同** |
| CRAVE/LMWM(抽特征/分段/训 WM) | cnsh: `conda:srpo`;cnbj: `kai0/.venv`+sklearn1.7.2 | ✅ DINOv3 走同一 HF 路径 |
| RoboTwin 仿真 | RoboTwin conda(sapien/mplib/curobo) | ✅ 逐项相同,经 `ROBOTWIN_PYTHON` |

**硬性要求:**
1. **版本钉死**:`scikit-learn==1.7.2` 等关键库两环境必须同版本(sklearn 版本会改 BGMM/PCA 输出)。
2. **产物记录来源**:数据产物旁写 `_env.json`(库版本 **+ 脚本 git hash**)。**未纳入 git 的脚本不得产出生产数据**。
3. **禁用 `py311_bak` 抽特征**:transformers 4.53<4.56,`hf_dino` 会静默回退 standalone(实测比特级一致但不可依赖)。
4. **不在 entrypoint 写机器/环境字面量** → 用 `_cluster_env.sh`(按挂载点自动判定环境)+ `mkyaml.py`(一 body 生成两环境 yaml,含硬编码则拒绝)。

---

## 3. 任务提交(操作 vePFS 环境,通道透明)

**同一份可移植 body → 两环境 yaml,只有环境相关字段不同:**

| 字段 | cnsh | cnbj(North-E) |
|---|---|---|
| ResourceQueueName | `robot-task` | `Robot-North-H20` |
| VepfsId | `vepfs-cnsh075262e1f815` | `vepfs-cnbj875793a96d6b` |
| MountPath | `/vePFS` | `/vePFS-North-E/vis_robot` |
| Flavor(8卡) | `ml.hpcpni2.28xlarge`(A100) | `ml.hpcpni3ln.45xlarge`(H20) |
| ImageUrl | `…grasp/kai:kai0-gf0` | `…vis_robot/kai:kai0-gf1` |

**这些差异全部封装在 `mkyaml.py` 的 profile 里**,body 的 entrypoint 只用 `$REPO`/`$PYTHON` 等变量。

**提交流程(两环境统一):**
```bash
# body 里 entrypoint 顶部: source <repo>/train_scripts/kai/volc/_cluster_env.sh
python mkyaml.py body.yaml --cluster both --gpus 8      # 生成 *_cnsh.yaml / *_northe.yaml
# gf0 侧持有 volcengine 凭证, 经 API 提交(gsy python 无 volcengine)
VOLC_AK=... VOLC_SK=... python submit_yaml.py <yaml>
```

**提交前 preflight(反复踩过,写进 body):**
- entrypoint 抽出跑 `bash -n`;无残留占位符 `__X__`;无集群字面量。
- 纯 LaWM 臂 `export LMWM_ 数=0`;LMWM 臂架构 env(`LMWM_DUAL`/`DUAL_2Q`/`FEAT_STRIDE=1`)齐全。
- 并行多路 eval:每路独立 `SEED=$i` + `PORT_BASE`(端口串台会让 A 臂连到 B 臂 server,产出污染数据)。
- cnbj 队列**无 4 卡规格**(`ml.pni3ln.20xlarge` 报 InvalidParameter);用 1 卡 `ml.pni3ln.5xlarge` 或 8 卡整节点。

**监控:** gf0 持凭证经 volc API 查任一环境的任务状态。`~/.volc/credentials` 若损坏会让 SDK 崩,
凭证走 `VOLC_AK/SK` 环境变量 + `HOME` 重定向避开坏文件。

---

## 4. 为什么这样设计(反面教训)

以"服务器"为中心思考,导致过多次事故:
- 把 `lmvla/lawam` 的修改直接改在 gf0 checkout 里、游离于版本控制 → North-E 拿不到。
- entrypoint 硬编码 `/vePFS/tim/...`(cnsh 路径)→ 同 yaml 在 cnbj 直接 `IndexError`。
- 以为"North-E 落后 gf0",实则**两机跟的是不同 GitHub 远端**。
- 把 cron 自恢复挂在容器可写层 → 重启即丢(gsy 是容器,持久物必须在 vePFS)。

**正确心智模型:** 我操作的是 cnsh / cnbj 两个 vePFS 环境;gf0/gsy 只是我今天恰好用来接入它们的通道,明天换一台挂了同一 vePFS 的机器,一切照旧。
