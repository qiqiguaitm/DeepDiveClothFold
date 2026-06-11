# 夹爪 action 裁剪(≤5mm→0)对真机夹持稳定性的影响 — 对照实验 plan

> **建立**: 2026-06-10
> **目的(用户定档)**: 探索**夹爪数据处理**对真机效果的影响。**真实目标 = 让夹爪真机更稳,不出现过程中松手 / 衣物从手中脱落。**
> **方法**: 单变量对照 —— 只改"夹爪 action 是否裁剪",训练参数/数据 episode/init 全部锁死。
> **状态**: 📋 **规划定稿(2026-06-10)** — 5 个决策已定档(§7);**等用户发话即开 build + 训练**,本次仅文档。
> ⚠️ **铁律**: **真机为终判**;offline MAE 仅作收敛 sanity(夹爪是否更稳是真机判据)。

---

## 0. 诊断(为什么要做)— 来自夹爪分布实测

实测 Task_A(kai0 + vis)夹爪 action 分布(图:`temp/gripper_zoom_{kai0,vis}.png`):
- **action ≡ state 逐维精确相等**(|Δ|=0,连手臂都是)→ action 记录的是"**被物体限位的实际位置**",不是"**意图闭合指令**"。
- **抓取闭合的主峰在 1–3mm**(布料厚度),不是 0:vis [1,3)mm 占 32–38% / kai0 占 25–35%。
- **几乎没有"越过布、给夹持力"的负向指令**:vis `<0mm = 0.0%`(硬 clamp 在 0)、kai0 仅 1.4%/0.2%。

→ **机理**:位置控制的夹爪,命令"当前位置(布厚 ~2mm)"= 只给刚好维持位置的力;要**夹持力**必须命令一个**比物体限位更闭的目标(0)**。当前数据让策略学会"抓到布就停在 2mm"→ **给不到力 → 真机松手 / 衣物脱落**。

---

## 1. 假设

> **把"想闭合"的夹爪 action(≤5mm)硬裁到 0,策略就会在抓取时命令"全闭(满力)"而非停在布厚位置 → 夹持更紧 → 不松手、不脱落。**

5mm 阈值依据(zoom 图):≤5mm 覆盖 空夹到底(0–1mm)+ **抓布主峰(1–3mm)** + 余量(3–5mm),正好是"意图闭合"区;5–10mm 是张/闭过渡(保留),>30mm 是张开。

---

## 2. 实验设计(单变量 = 夹爪 action 裁剪)

| | **Baseline(对照)** | **Treatment(处理)** |
|---|---|---|
| 数据集 | `A_smooth800_dagger_all`(原始,1117ep/1.47M 帧)| `A_smooth800_dagger_clip_all`(夹爪 action ≤5mm→0)|
| **裁剪** | 无 | gripper action dim **[6,13]**:`value ≤ 0.005m → 0.0`(硬裁)|
| init | **二者完全相同**(克隆现有 config 的 init)| 同 |
| 训练超参 | **完全相同**(LR/warmup/steps/EMA/batch/fsdp/horizon/norm 方式)| 同 |
| norm_stats | 各自重算(裁剪后分布变了)| 同 |
| eval | val MAE(夹爪维单列)+ **真机 rollout 夹持稳定性** | 同协议 |
| exp_name | `smooth800_dagger_all_baseline` | `smooth800_dagger_clip_all` |

> **唯一变量 = 夹爪 action 是否 ≤5mm→0。** 任何真机差异都干净归因到夹爪处理。

---

## 3. ⭐ 数据处理规格 + 一个必须确认的决策

### 3.1 裁剪定义
- 源:`kai0/data/Task_A/self_built/A_smooth800_dagger_all`(1117 ep,3相机,action[14]/state[14],夹爪维 = **6(L)、13(R)**)。
- 逐帧:`action[:, [6,13]]` 中 **≤ 0.005m 的值 → 0.0**(硬裁,>5mm 不动)。
- **手臂维(0–5, 7–12)不动**;**清离群点**(>0.1m 的坏帧,如 kai0 见过的 1875,先 sanitize)。
- 视频/meta 复用(symlink),只改 parquet 的 action 列 → 重排不需要(帧数不变)。
- **重算 norm_stats**(`compute_norm_states_fast.py`)。
- 产物:`A_smooth800_dagger_clip_all`。

### 3.2 ⚠️ 关键决策:裁 **action only** 还是 **action + state**?
| 选项 | 效果 | 评估 |
|---|---|---|
| **A. 只裁 action(推荐)** | grasp 帧 **action=0 但 state=真实 2mm** → 制造 "从非零抓取态命令 0" 的**力信号**;state 保持真实测量 → **train/deploy 一致**(部署时 proprio 也是真实 2mm)| ✅ 既给力信号又不破坏 state 真实性 |
| B. action + state 都裁 | grasp 帧 state 也=0(掩盖真实位置)→ 部署时真实 proprio=2mm ≠ 训练的 0 → **train/deploy 漂移**(除非 serve 端也裁 state)| 🟡 多一处 serve 改动、信号更弱 |

→ ✅ **已定档(2026-06-10 用户确认)= 只裁 action(夹爪 6/13),state 保持真实。** 模型学"抓取态 → 命令 0(给力)",部署无 state 漂移。

### 3.3 已知副作用(文档标注,真机观察)
- **阈值不连续**:5.0mm→0 但 5.1mm 不变 → 阈值穿越处 action 有跳变(命令"突然全闭")。可接受(进入 5mm 内即命令全闭),但若真机抖动,改**平滑 remap**(如 [0,5]mm→线性压到 0,或 [0,8]→[0,0] 软过渡)。
- **过度夹持 / 不松手风险**:总命令 0 可能导致夹爪持续顶死、电机过载、或该松时松不利落(0→开要穿过 5mm 跳变)。**真机要看:① 是否更不脱落(目标)② 是否过夹/松不开/电机报警(副作用)。**
- **部署侧 gripper 映射 parity**(⚠️ 之前 XVLA 踩过 SoftFold 默认 −0.0055 坑):serve 必须把 **action=0 映射成"全闭/给力"**,且 `gripper_close_value` 对齐本体真实行程(vis ~0 / kai0 可略负)。上真机前核对。

---

## 4. 训练规格(✅ 已定档,沿用 `pi05_flatten_fold_A_smooth800_dagger_full` config.py:1798)
- **config**: 克隆该 config → 两个新 config:
  - `pi05_smooth800_dagger_all_baseline`(repo_id → `A_smooth800_dagger_all`)
  - `pi05_smooth800_dagger_clip_all`(repo_id → `A_smooth800_dagger_clip_all`)
  - **除 repo_id + 各自 norm_stats 外,所有超参/init 完全相同。**
- **init(两臂同一个)**: `CheckpointWeightLoader("shared_ckpt/Task_A/mixed_1_clean/params")` warm-start。
- **超参(两臂相同,照搬 dagger_full)**: `Pi0Config(pi05=True)` · use_delta_joint_actions=False(absolute) · cosine **warmup 1k / peak 1.5e-5 / decay 50k → 1.5e-6** · **EMA 0.9999** · **50k step** · batch **128** · **fsdp 8** · num_workers 16 · save 每 2k / keep 10k · inline-eval `vis_v2_merged_val`(n=200, every 4) · prompt "Flatten and fold the cloth."。
- **资源**: 单节点 **8 卡**(cnsh A100 / cnbj H20)。
- **范围**: 本实验**仅 vis**(`A_smooth800_dagger_all` 即 vis 单本体 smooth800+dagger);kai0 本体本轮不裁(力尾巴更小,留作后续)。

---

## 5. 评估协议(真机为终判)
**Tier 1 — offline sanity**:同 val split,逐 ckpt **val MAE**(尤其**夹爪维单列 MAE**)+ loss → 确认收敛、选 ckpt。⚠️ 夹爪 MAE 对"是否给力"不敏感,只作 sanity。

**Tier 3 — 真机 rollout(决定性)**:两个 ckpt 同硬件、同初始布、同 prompt 跑叠衣,记录**夹持稳定性指标**:
- **过程松手次数 / 衣物脱落次数**(↓ = 改善,**主指标**)
- 抓取成功率 / 折叠完成率
- 副作用:过夹/松不开/电机报警次数

**判据**:
- clip 版**脱落/松手明显↓ 且无严重副作用** → ✅ 夹爪裁剪有效,采纳为默认处理。
- ≈ baseline → 夹爪给力非脱落主因(查别处:轨迹/视觉/抓取位姿)。
- clip 版**脱落↓ 但出现过夹/松不开** → 阈值/力目标过激 → 调阈值(3mm)或改平滑 remap,或方案 B(只在"抓取意图"段裁)。

---

## 6. 落地步骤
1. **build** `A_smooth800_dagger_clip_all`:写裁剪脚本(load parquet → `action[:,[6,13]]` ≤0.005→0 → 写回;sanitize 离群;视频 symlink 复用)。
2. **验证**:对 clip 数据集**重画夹爪 action 分布**(`temp/gripper_zoom_clip_*.png`),确认 1–5mm 质量塌到 0、>5mm 不变、手臂维不变。
3. **norm_stats** 各自重算(baseline 也要,保证口径一致)。
4. **注册 2 个 config**(§4),git commit/push。
5. **训练**两臂(8卡,同步参,同 init)。
6. **eval**:val MAE 曲线 + **真机对比脱落/松手**(§5)→ 落判据 + 回填本文档。

---

## 7. 决策定档(✅ 2026-06-10 用户确认)
1. ✅ **裁 action-only**(夹爪 6/13,≤5mm→0;state 保持真实)。
2. ✅ **阈值 5mm**(0.005m)。
3. ✅ **init = warm-start `mixed_1_clean/params`**(两臂同一个)。
4. ✅ **steps/集群 = 沿用 dagger config**(50k / 8卡 / batch128 / fsdp8 / cosine 1.5e-5 / EMA0.9999)。
5. ✅ **本实验仅 vis**(kai0 本轮不裁)。

→ **5 点全定,可进入落地步骤(§6)。等用户发话即开 build + 训练。**

---

## 关联
- 诊断图: `temp/gripper_zoom_{kai0,vis}.png`(0–20mm 细看)、`temp/gripper_action_dist_{kai0,vis}.png`(全量)
- 数据: `kai0/data/Task_A/self_built/A_smooth800_dagger_all`(1117ep/1.47M)
- config 克隆源: `kai0/src/openpi/training/config.py:1798`(`pi05_flatten_fold_A_smooth800_dagger_full`)
- 部署 gripper 映射坑: `docs/training/analysis/xvla_vs_official_gap_rootcause.md §(c2)`(SoftFold −0.0055 / 须对齐本体行程)
