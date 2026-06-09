# Plan A — Embodiment Conditioning + Per-DS Norm (预合并单源)

> **目的**: 干净地检验 **per-dataset norm + domain-token conditioning** 能否在 kai+vis 混训下保住 vis 部署精度(并为"压双模态抖动 / kai 帮 OOD"做真机判据)。之前所有 conditioning 实验都跑在 broken 的 `datasets_yaml`/ConcatDataset 路径上、全塌成 predict-zero —— **conditioning 从没被真正检验过**。本路线改走**物理预合并单源 + 真·per-DS norm + 加权采样**。
>
> **状态**: ✅ **offline + 真机均通过 (2026-06-07)** —— `pi05_kaivis_perdsnorm_cond` 50k 训完:offline vis inline MAE@1 单调降到 **0.0086(≈ vis-only),全程没塌**;**真机实测无问题(用户 2026-06-07)**。conditioning 路线**第一次干净验证成功(offline+真机)**。
> ⚠️ **但留下一个 confound**:Exp-1 的 vis = `smooth800 + vis_dagger`,kai = `base + kai_dagger` —— **两边都含 dagger(真机纠错强成分)**,**无法分辨真机好是 kai 的贡献还是 vis_dagger 的功劳**。→ 需 **Exp-2(vis 缩到纯 smooth800,去 vis_dagger)** 隔离 kai 是否真帮(见 §7)。
>
> **最佳 ckpt**:
> ```
> /vePFS/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_kaivis_perdsnorm_cond/pi05_kaivis_perdsnorm_cond_cnsh/49999
> ```
>
> **关联**:
> - 崩溃实证: [`../../history/experiments/conditioning_vs_action_representation_ablation.md`](../../history/experiments/conditioning_vs_action_representation_ablation.md)
> - 根因核查: [`../../analysis/pi05_cross_embodiment_training_deep_dive.md`](../../analysis/pi05_cross_embodiment_training_deep_dive.md)
> - 跨本体战略 (21° 腕姿双模态 / 方案B per-DS norm): [`../../../deployment/strategy/cross_embodiment_strategy.md`](../../../deployment/strategy/cross_embodiment_strategy.md) §2.4
> - 工程踩坑: [`../../../deployment/training_ops/submission/training_pitfalls_common.md`](../../../deployment/training_ops/submission/training_pitfalls_common.md)

---

## TL;DR (结论速览, 2026-06-06)

1. **方法跑通,没塌,真机 OK**: `pi05_kaivis_perdsnorm_cond`(kai_base+dagger ⊕ vis=smooth800+dagger,per-DS norm + domain token + 帧级 1:1 加权采样,单预合并集)50k 训完。vis inline MAE@1 **0.0305(8k)→ 0.0086(49999)**,单调收敛;**真机实测无问题(用户 2026-06-07)**。对比历史三连崩(@1≈0.47),conditioning 第一次在健康路径上 offline+真机双通过。
   - ⚠️ **confound(待 Exp-2 解)**: vis 含 vis_dagger、kai 含 kai_dagger,两边都有 dagger(真机纠错强成分)→ **分不清真机好是 kai 帮的还是 vis_dagger 的功劳**。
2. **kai 没伤 vis**: 收敛 vis @1=0.0086 ≈ 纯 vis baseline(Exp-B vis 0.0080 / dagger-B 0.0085)。混入 6512 ep kai + per-DS norm + token,vis in-distribution 单步精度**持平**纯 vis。
3. **三连崩的真因不是 conditioning,是 `datasets_yaml` 代码路径**(详见 §0)。修复 = 物理预合并单源 + 真·per-DS norm(之前从没实现)+ task_index 透传 domain + 帧级加权采样。
4. **还没回答的(需真机 + control)**: Q1 减抖、Q2 超越/OOD、Q3 conditioning 必要性(要 no-cond control 对照)。offline 只证明"前置闸门通过"。

---

## 0. 为什么之前"白了" —— conditioning 从没被真正检验过

代码级核查 + E3.6 实证结论:

1. **之前 conditioning 全崩,但凶手不是 conditioning** —— E3.6(无 cond)/ Track C(cond)/ Action-delta 三连崩(val MAE ≈ 0.47 predict-zero),共同点是**都走 `datasets_yaml` → `_create_concat_torch_dataset`**(`data_loader.py:167/203`)。conditioning 接线本身正确(`pi0.py:247-254`)。
2. **per-source norm 之前在代码里根本不存在** —— `transform_dataset` 对整个 ConcatDataset 套**单一** norm。E3.6 config 名 `xvla_e3_6_per_ds_norm_no_cond` 和 yaml 注释 "per-source norm" **均为误导,机制从未实现**。(本路线才**第一次真正实现** per-DS norm,见 §2.2。)
3. **连 vis×1 也崩** —— E3.6 实际是 vis×1(无过采样)仍塌,说明崩**与过采样/平衡无关,是路径本身的 bug**。
4. **分水岭 = 预合并单源(健康)vs datasets_yaml ConcatDataset(崩)** —— Hard Prompt 混训(物理预合并单数据集)同样单一 norm 却**健康**(vis ~0.008)。

→ **两个不同的坑**(本路线绕开 A,用 conditioning 去测 B):

| 坑 | 触发 | 症状 | 性质 |
|---|---|---|---|
| **A: offline collapse** | datasets_yaml ConcatDataset | offline MAE 0.47 + 真机不动/抖 | 管线 bug(本路线绕开)|
| **B: 真机抖动** | 物理预合并(naive 混训) | offline 健康但真机抖 > 纯 vis | 真实 kai/vis 双模态冲突(21° 腕姿)|

> **"白"的教训**: 一条被底层管线 bug 污染的实验线,跑再多组、调再多超参都是白做 —— 必须先把"健康路径 vs broken 路径"这个分水岭找出来。E3.6(无 cond 也崩)是定位 bug 的关键对照。

---

## 1. 核心问题 Q1/Q2/Q3 + 当前 offline 进展

| # | 问题 | 假说 | 终判(真机) | **offline 进展 (2026-06-06)** |
|---|---|---|---|---|
| **Q1** | domain token(推理固定 vis)能否消歧、压住真机抖动? | 能 | 真机抖动 ≤ vis-only | ✅ **真机无问题(用户 2026-06-07)** —— 不抖、可用。(具体抖动/成功率数值待量化回填)|
| **Q2** | 加 kai co-train(有 cond)能否**超越** vis-only(成功率/OOD)? | 边际提升或持平 | 真机成功率 ≥ vis-only | ⚠️ 真机 work,但 **dagger confound 无法归因**:vis 含 vis_dagger → 分不清是 kai 还是 vis_dagger 的功劳 → **Exp-2 隔离(§7)** |
| **Q3** | conditioning 是否必要?(vs 预合并无 cond) | 必要(无 cond=坑 B) | 对照 no-cond control | ⏳ **control 还没跑** → Q3 未答 |

> ⚠️ **真机为终判**: offline per-source MAE 只用于 ① 训练健康闸门 ② 选 ckpt ③ 相对差。conditioning 的价值(消歧/减抖)在逐帧 MAE 上未必显著 → **必须真机对比**。

---

## 2. 实际实现 (⚠️ 偏离原计划, 以此为准)

> 原计划 vis 用 `vis_base/v3`(1940ep)+ 单一 norm。**实际执行按用户决策改为**:vis = `A_smooth800_dagger_full`(真机已验证的部署锚点),kai = base+dagger,**并真正实现了 per-DS norm + 帧级加权采样**。

### 2.1 数据 — 单预合并集 `kai_vis_merged`(`build_kai_vis_merged.py`)

| 源 | domain (=task_index) | ep | frames | 备注 |
|---|---:|---:|---:|---|
| `kai0_base` | 0 | 3055 | 3.36M | kai 官方 |
| `kai0_dagger` | 0 | 3457 | 2.42M | state/action info 声明 `[1,14]` 但实存 `(14,)` → build squeeze |
| `A_smooth800_dagger_full`(**vis**) | 1 | 1033 | 1.46M | smooth800+dagger,真机已验证;视频 symlink |
| **合计** | — | **7545** | **7.23M** | 单 chunk-000,chunks_size=7545,视频全 symlink(0 落盘膨胀) |

- **domain 逐帧带在 `task_index`**(kai=0 / vis=1),`tasks.jsonl` 2 条同 prompt。训练时 `ReadDatasetIdFromTaskIndex`(`transforms.py`)映射 task_index→`obs.dataset_id`(**custom parquet 列不保证透传,task_index 一定到 frame**;该 transform 在推理端 no-op,obs 已直接带 dataset_id)。

### 2.2 per-DS norm(本路线的真增量,`build_kai_vis_norm.py` + `transforms.DomainNormalize`)

- **2 份 norm**:kai(base+dagger 合计 5.78M 帧)/ vis(1.46M 帧),各含 q01/q99 → `kai_vis_merged/norm_domain{0_kai,1_vis}/`;单 fallback = vis(部署目标的输出 unnormalize)。
- `data_loader.transform_dataset` 在 `domain_norm_stats` 非空时改用 **`DomainNormalize`**(按 `obs.dataset_id` 选 kai/vis 的 quantile norm),替换单一 `Normalize`。**走的是健康单源路径**(单 LeRobotDataset),不碰 datasets_yaml。

### 2.3 帧级 1:1 加权采样(`_DomainWeightedJAXSampler`)

- kai:vis 按 **ep** 是 6.3:1,但训练采样的是**帧** —— kai 帧 5.78M / vis 帧 1.46M = **3.97:1**(kai ep 更长)。`domain_weights=(1.0, 3.970)` → 帧级 vis 占比 **0.507 ≈ 1:1**(smoke 实测)。**零落盘膨胀**(概率采样,非复制),JAX 多机分片。

### 2.4 config + 训练规格

`config.py` 新增 `KaiVisMergedDataConfig` + `TrainConfig pi05_kaivis_perdsnorm_cond`:
`Pi0Config(pi05=True, action_head_cond_num_domains=2)` / init `pi05_base` / 50k / bs128 / fsdp16 / `inline_eval_dataset_id=1`(vis;否则 `pi0.py:247` 跳 token)/ inline val `vis_v2_merged_val`。**cnsh 2-host 16 A100**(`pi05_kaivis_perdsnorm_cond_cnsh_16gpu.yaml`)。

---

## 3. 结果 (offline)

### 3.1 vis inline-eval MAE 曲线(`vis_v2_merged_val`, dataset_id=1)

| step | @1 | @10 | @25 | @50 |
|---|---|---|---|---|
| 8000 | 0.0305 | 0.0459 | 0.0688 | 0.0992 |
| 16000 | 0.0155 | 0.0284 | 0.0461 | 0.0679 |
| 24000 | 0.0109 | 0.0226 | 0.0372 | 0.0541 |
| 32000 | 0.0093 | 0.0200 | 0.0320 | 0.0453 |
| 40000 | 0.0088 | 0.0185 | 0.0287 | 0.0397 |
| 48000 | 0.0087 | 0.0176 | 0.0266 | 0.0363 |
| **49999** | **0.0086** | **0.0175** | **0.0262** | **0.0357** |

**单调收敛、末端 plateau(48k≈49999)、无过拟合回弹 → 最佳 ckpt = step 49999。**

### 3.2 判断

- ✅✅ **没塌**(@1=0.0086 而非 ≈0.47)→ conditioning 路线第一次干净跑通。
- ✅ **kai 没伤 vis**:vis @1=0.0086 ≈ 纯 vis baseline(Exp-B vis 0.0080 / dagger-B 0.0085)。
- ⚠️ **口径**:inline 200-frame 子集 + val=`vis_v2_merged_val`(与 vis 训练域 smooth800 不完全同分布)。严格 A/B(cond vs no-cond vs vis-only)需同 val offline 重测。

---

## 4. 经验教训(执行踩坑, 已全部修 + 落 pitfalls 文档)

1. **`datasets_yaml`/ConcatDataset 路径 broken → 一律物理预合并单源**(本路线核心前提)。
2. **per-DS norm 要自己实现**(`DomainNormalize` 按 dataset_id 切),历史命名"per-ds norm"是空壳。
3. **1:1 平衡按帧不按 ep** —— kai ep 更长,ep 6.3:1 但帧 3.97:1;采的是帧,权重要用帧比(否则 vis 过采到 0.62)。
4. **domain 用 task_index 透传**(custom parquet 列不保证进 frame dict);`ReadDatasetIdFromTaskIndex` 在推理/serve 端必须 no-op(obs 已带 dataset_id、无 task_index)。
5. **`chunks_size` 必须 ≥ N**(7545 ep 单 chunk-000;默认 1000 会让 lerobot 找 chunk-001 → 文件 assert 崩 → offline HF crash)。`info.total_episodes` 用实际写入数。
6. **多机 ckpt-init 残桩 → `sync_global_devices` name mismatch**:resume/重提前清掉上次失败留的 ckpt 目录。
7. **smoke 要覆盖到采样器实例化** —— 一个 `import math` 漏在模块级(只在 `_JAXProcessSampler` 内 import),导致首次提交在数据加载完、采样器构造时 `NameError` 崩。video/model-free smoke 没测到这行 → 教训:smoke 要实例化所有新类。
8. **inline-eval 单条 ~36min**(pi05 200 序列 denoise infer)→ 日志"冻结"是正常,别误判 hang(看两节点是否同步在同一 barrier)。

---

## 5. 评估协议 + 真机 next + 决策树

**Offline(健康闸门,已过)**:vis MAE@1 同量级 ~0.008、绝不 ≈0.47 ✅;(待补)conditioning sanity:同帧切 dataset_id=0/1 输出应明显不同。

**真机(终判,vis 机器)** — 三组对比 `pi05_kaivis_perdsnorm_cond` vs **no-cond control** vs **vis-only baseline**:

| metric | 对应 |
|---|---|
| 抖动(action diff p99 / 空桌目测) | Q1 |
| 抓衣角 / 完整折叠成功率(固定场景 30 ep) | Q2 |
| OOD 场景成功率 | Q2(kai 多样性是否帮泛化) |

```
offline vis MAE ≈ 0.47 ? — 否(0.0086)✅ → 进真机
  cond 抖动 ≤ baseline 且 成功率 ≥ baseline ? 
    是 → ✅ conditioning 路线成立
    否,但 no-cond control 抖更厉害 → cond 有效但 kai 净收益负 → 回 vis-only,cond 留给真异构
    否,且 control≈cond → 同任务近同构下 cond 无用、vis-only 最优 → 收束到 vis-only 结论
```

---

## 6. 剩余 checklist

- [x] 合并集 `kai_vis_merged` + 2 份 per-DS norm + 帧级加权采样 + config + smoke + cnsh 16卡训练 50k
- [x] offline 健康闸门(vis @1=0.0086,没塌)
- [x] **Exp-1 真机**:无问题(用户 2026-06-07)。✅ conditioning 路线 offline+真机双通过(具体抖动/成功率数值待量化回填)
- [x] **⭐ Exp-2(隔离 kai)build+提交**:vis 缩到纯 smooth800,合并集 `kai_vis_s800_merged`(7318ep)+ 权重 6.246 + per-DS norm 重算 + smoke ✅ → **8卡 cnsh 已提交 `t-20260607140155-46btn`**(2026-06-07)→ 见 §7.2
- [ ] **Exp-2 真机 vs smooth800-only baseline**(= 既有 `task_a_new_smooth_800` 模型)→ **判定 kai 是否真帮**
- [ ] (可选)**no-cond control**:同数据去掉 `action_head_cond_num_domains`(Q3,conditioning 必要性)
- [ ] 回填 `xvla_conditioning_methods_results.md` + deep-dive 结论

---

## 7. 实验进度总结 + 新实验 Exp-2(隔离 kai 贡献)

### 7.1 已完成实验

| 实验 | config | kai | vis | 采样 | offline vis MAE@1 | 真机 | 结论 |
|---|---|---|---|---|---:|---|---|
| 历史三连崩 | E3.6 / Track C / Action-delta | base+dagger | vis_v2_merged ×7 | datasets_yaml ConcatDataset(broken) | ≈0.47 塌 | 不动/抖 | ❌ 管线 bug,非 conditioning |
| **Exp-1** ✅ | `pi05_kaivis_perdsnorm_cond` | base+dagger | **smooth800+dagger** | 预合并单源 + per-DS norm + 帧级 1:1 | **0.0086** | ✅ **无问题(2026-06-07)** | conditioning 路线 offline+真机首次跑通 |

**Exp-1 已坐实**:① 物理预合并 + 真 per-DS norm + 帧级加权 → conditioning 不塌、真机可用;② kai 没伤 vis(@1 持平纯 vis)。
**Exp-1 未解(confound)**:vis 含 vis_dagger、kai 含 kai_dagger → **真机好分不清是 kai 的贡献还是 vis_dagger(真机纠错强成分)的功劳**。

### 7.2 ⭐ Exp-2 — vis 去 dagger(纯 smooth800),隔离 kai 是否真帮

> **动机**: 去掉 vis 侧 dagger 这个强 confound,让 vis 基线"干净",再看加 kai 是否真带来真机增益。

**变更(相对 Exp-1,单变量:只动 vis 源)**:
| | Exp-1 | **Exp-2** |
|---|---|---|
| vis 数据 | `A_smooth800_dagger_full`(1033ep/1.46M,含 vis_dagger) | **`A_new_smooth_800/base`(811→806ep*/0.925M,纯 smooth800,去 vis_dagger)** |
| kai 数据 | base+dagger(6512ep/5.78M) | 不变(6512ep/5,777,710 frames) |
| 合并集 | `kai_vis_merged`(7545ep) | **`kai_vis_s800_merged`(7318ep/6.70M frames)** |
| 帧级权重 (kai,vis) | (1.0, 3.970) | **(1.0, 6.246)** 重算维持帧级 1:1(kai 5,777,710 / vis 925,055,smoke vis-fraction=0.507 ✅)|
| per-DS norm | kai / vis(含dagger) | **vis 用纯 smooth800 帧重算**(已生成 norm_domain1_vis) |
| config | `pi05_kaivis_perdsnorm_cond` | **`pi05_kaivis_cond_visS800`** |
| 集群/卡 | cnsh 2-host 16 A100 | **cnsh 1-host 8 A100**(fsdp=8;bs128/50k 不变,global batch 同) |
| 其余 | per-DS norm + domain token + 预合并单源 + init pi05_base + inline_eval_dataset_id=1 | **完全相同** |

> *5 个 vis ep 因缺视频被 build 跳过(811→806);kai 全保留。

**对照 baseline(关键,干净可比)**:
| Run | kai | vis | dagger? | 用途 |
|---|---|---|---|---|
| **smooth800-only baseline** | ❌ | 纯 smooth800 | 无 | = 既有模型 `task_a_new_smooth_800(_new_norm)`(无 kai 无 dagger)→ 直接复用作干净 vis-only 参考 |
| **Exp-2(新)** | base+dagger | 纯 smooth800 | 仅 kai 侧 | 测 kai 贡献 |

**判据(真机为终判)**: Exp-2 真机 vs smooth800-only baseline →
- Exp-2 **明显优**(成功率↑/抖动↓/OOD↑)→ **kai 数据确实帮 vis**(且非 vis_dagger 功劳)。
- Exp-2 **≈ baseline** → kai 对(同任务近同构的)vis 无净增益 → 收束到 vis-only / 把 kai 留给真异构。
- ⚠️ 残留: kai 仍含 kai_dagger(属"kai 数据"一部分,合理);若要更纯,后续可再出 kai-base-only 变体(本次先一个)。

**执行链(✅ 已提交 2026-06-07)**:
1. ✅ build `kai_vis_s800_merged`(`build_kai_vis_merged.py --vis-src self_built/A_new_smooth_800/base`)→ 7318ep/6.70M。
2. ✅ `build_kai_vis_norm.py --merged kai_vis_s800_merged` → 2 份 per-DS norm(kai 5,777,710 / vis 925,055 frames)。
3. ✅ 帧级权重 = 5,777,710/925,055 = **6.246**(维持帧级 1:1)。
4. ✅ config `pi05_kaivis_cond_visS800`(克隆 Exp-1,fsdp_devices=8)。commit+push(`6099534`)。
5. ✅ smoke:per-domain norm 区分 + vis-fraction **0.507** + dataset_id 透传 ✅。
6. ✅ **提交 8卡 cnsh**:yaml `pi05_kaivis_cond_visS800_cnsh_8gpu.yaml`,**task_id `t-20260607140155-46btn`**(cn-shanghai / robot-task)。
7. ✅ **训完 50k**(2026-06-09)。inline-eval MAE(见下),收敛健康没塌。
8. ⏳ **真机 vs smooth800-only baseline(`task_a_new_smooth_800`)对比** → 判定 kai 是否真帮(终判,待做)。

**Exp-2 offline 结果(inline-eval `vis_v2_merged_val`,dataset_id=1):**

| step | MAE@1 | @10 | @25 | @50 |
|---|---|---|---|---|
| 8000 | 0.0308 | 0.0455 | 0.0672 | 0.0960 |
| 16000 | 0.0156 | 0.0276 | 0.0429 | 0.0616 |
| 24000 | 0.0111 | 0.0215 | 0.0333 | 0.0468 |
| 32000 | 0.0095 | 0.0186 | 0.0281 | 0.0387 |
| 40000 | 0.0089 | 0.0171 | 0.0251 | 0.0341 |
| **48000** | **0.0087** | **0.0163** | **0.0233** | **0.0314** |

- 单调收敛、没塌(@1 0.0308→0.0087)。inline-eval 每 8k 一次,末次 48000(49999 非 8k 倍数无 eval)。**最佳已保存 ckpt = `49999`**(48000 最低但未存;49999 收敛终点 ≈ 48000)。
- **最佳 ckpt**:`/vePFS/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_kaivis_cond_visS800/pi05_kaivis_cond_visS800_cnsh/49999`
- **vs Exp-1(vis 含 dagger)同 val**:@1 **0.0087 ≈ 0.0086**(去 vis_dagger 单步几乎不变);@50 **0.0314 < 0.0357**(去 dagger 长程反略优)。→ **offline:去 vis_dagger 不伤,甚至长 horizon 略好**;但 kai 是否真帮**仍须真机对 smooth800-only baseline**(offline 反指)。

---

## 附: 避坑铁律

1. **不走 `datasets_yaml`** —— kai+vis 混训先物理预合并单数据集。
2. **dataset_id 逐帧带**(task_index 编码 + ReadDatasetIdFromTaskIndex)—— 部署/真机 client **必须传 `dataset_id=1`**,否则 token 被跳过 → 退化为无 cond。
3. **per-DS norm 用 DomainNormalize**(按 dataset_id 切);输出端用 vis norm(部署目标)。
4. **1:1 平衡按帧**(domain_weights 用帧比 3.970,非 ep 比 6.3)。
5. **prompt 统一**,别加 `[KAI]/[VIS]` 前缀(避免与 token 双信号)。
6. **真机为终判**,offline 只做健康闸门与选 ckpt。
