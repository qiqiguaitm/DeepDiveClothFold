# pi05 × 不同叠衣 SOP 范式基线小实验(Vertical Fold v1 / Horizontal Fold v1 …)

> **建立**: 2026-06-12 · **重构为多范式**: 2026-06-16(并入 Task_AH1)
> **目的**: 用**完全相同的一套训练配方**(§1),在**我们自己设计的不同叠衣 SOP 范式**数据集上各做一次 pi05 **基线小实验**,验证 ① 每种新 SOP 数据能否训出可部署策略;② **横向对比**哪种折法范式更易学/真机更稳。同时**正式记录每种新 SOP**(§2 各范式 §x.1)。
> **状态**: 📋 **配方定档,逐范式推进** —— Vertical Fold v1 (Task_AV1) 决策全定(§2.A);Horizontal Fold v1 (Task_AH1) 决策定档(§2.B,数据已落地)。**每个范式仅待 ① 用户补全该 SOP 物理流程 ② 发话"开始实施"**。本文档**只更新文档,不实施**。
> ⚠️ **铁律**: 真机为终判;VLA 训练报告先看 **val MAE**(不是 train loss)。

---

## 0. SOP 范式总览(registry)

| 范式 | 数据集 | 采集日期 | ep / 帧 | trim | prompt(规范化) | config | 状态 |
|---|---|---|---|---|---|---|---|
| **Vertical Fold v1**(竖向折)| `Task_AV1/base/v2` | 06-11-v2 / 06-12-v2 | 取前 **200** / — | v2 raw | `... Vertical Fold v1.` | `pi05_task_av1_vfold_v1_200` | 📋 定稿待实施 |
| **Horizontal Fold v1**(横向折)| `Task_AH1/base/v3` | 06-15-v3 | **200** / 272,086 | v3 前裁+尾裁 | `... Horizontal Fold v1.` | `pi05_task_ah1_hfold_v1_200` | 📋 定稿待实施 |

> ⚠️ **跨范式对比的一处不一致**: AV1 是 **v2 raw**(未前裁),AH1 是 **v3 前裁+尾裁**(前端投放裁 + 尾部 tail-cap)。各自作为独立基线没问题;但若要**严格横向对比**"哪种折法更易学",trim 版本是个混杂变量 → 解读时注意,或后续对齐到同一 trim(见 §3 caveat)。
> **共性**: 都是 vis 本体(Agilex 双臂 Piper),3 相机 `top_head/hand_left/hand_right`,14D 关节 state/action,30Hz,**action ≡ state**(含夹爪,见 §1)。

---

## 1. ⭐ 共享训练配方(所有范式严格一致 = 单变量只剩"折法 SOP")

| 项 | 配置(所有范式相同)|
|---|---|
| 模型 | **pi05**(`Pi0Config(pi05=True)`)|
| init | **warm-start `CheckpointWeightLoader("mixed_1_clean/params")`**(沿用 flatten-fold pi05 配方)|
| steps | **50,000** |
| batch / 并行 | **128** / **fsdp 8**(单节点 8 卡;cnsh A100 或 cnbj H20)|
| LR | cosine **warmup 1k / peak 1.5e-5 / decay 50k → 1.5e-6** |
| EMA | **0.9999** |
| 夹爪 | **原始 action,不裁**(action≡state、抓取停在布厚;裁剪版 ≤5mm→0 作后续独立对照,见 [`gripper_action_clip_experiment.md`](gripper_action_clip_experiment.md))|
| `use_delta_joint_actions` | **False**(absolute joint)|
| norm_stats | 每个范式数据集**各自重算**(`compute_norm_states_fast.py`)|
| save / val | save 每 2k / keep 10k;`inline_eval_val_root` → 各范式自家 held-out val |
| config 克隆源 | `kai0/src/openpi/training/config.py:1798`(`pi05_flatten_fold_A_smooth800_dagger_full`)|

> **数据用量统一规则**: 每范式取 **200 ep 量级**做基线 + **该范式自家留出 val**(in-distribution eval)。具体切分见各范式节。
> ⚠️ **prompt train==deploy 一字不差**:训练 config `default_prompt` 与真机部署 prompt 必须同一串(覆盖数据 tasks.jsonl 原值,见各范式)。

---

## 2.A 范式:Vertical Fold v1(竖向折,Task_AV1)

### 2.A.1 ⭐ SOP 记录
**已知(从 `Task_AV1` 数据提取)**:

| 项 | 值 |
|---|---|
| 任务 prompt(数据原值)| `"Flatten and fold the cloth. Vertical Flod v1."`(meta/tasks.jsonl,有 typo "Flod")|
| 单条 episode 时长 | 中位 **~47s**(1409 帧),范围 1017–1855 帧 |
| 采集日期 / 规模 | 2026-06-11-v2(133)+ 2026-06-12-v2(112+)= **245 ep**(截至 06-12,仍在增长)|

**待补充(请用户填写 — SOP 记录核心)**:
- 新流程相对旧流程改了什么 / 为什么改;竖向折分步动作(展平→第1折→…,每步双臂/夹爪做什么、抓哪个角);关键阶段 / 成功判据;采集约定(初始摆放/视角/单双布/投放等待);与 Task_A 关系。

### 2.A.2 数据准备(200 ep)
- **train 200 = 按日期顺序取前 200**:全 06-11-v2 的 133 + 06-12-v2 前 67 → `Task_AV1_200`(lerobot v2.1,episode_index 重排)。
- **val = 留出末 ~45 ep**(06-12-v2 第 68 起;同 SOP 同本体,分布一致)。
- build:`build_no_release.py --mode raw --merge-*` 或薄脚本,按日期合并 + 重排 + 视频 symlink;val 单独建;norm 重算。

### 2.A.3 决策定档(✅ 2026-06-12 用户确认)
1. ✅ 200 ep = 按日期顺序取前 200(06-11 全 133 + 06-12 前 67)。
2. ✅ init = warm-start `mixed_1_clean/params`。
3. ✅ 夹爪 = 原始 action(不裁)。
4. ✅ val = Task_AV1 留出末 ~45 ep。
5. ✅ **prompt = B 规范化** `"Flatten and fold the cloth. Vertical Fold v1."`(修 typo Flod→Fold + 去尾空格;覆盖数据原值)。

---

## 2.B 范式:Horizontal Fold v1(横向折,Task_AH1)

### 2.B.1 ⭐ SOP 记录
**已知(从 `Task_AH1` 数据提取,2026-06-16 已拉到本地)**:

| 项 | 值 |
|---|---|
| 任务 prompt(数据原值)| `"Flatten and fold the cloth. Horizontally Fold v1."`(meta/tasks.jsonl)|
| 数据集 / 路径 | `kai0/data/Task_AH1/base/v3/2026-06-15-v3/`(TOS `KAI0/Task_AH1`,**已排除 depth**,4.3G)|
| 规模 | **200 ep / 272,086 帧**(单日 2026-06-15;**尾裁后**,原前裁版 277,153,Step 3 tail-cap 删 5,067)|
| trim | **v3 前裁+尾裁**(前端投放裁 + 尾部 tail-cap;与 AV1 的 v2 raw 不同)|
| 相机 / 校验 | 3 路(无 depth),parquet 200 / 视频 600(=3×200)✅,info.json total_episodes=200 一致 |

**待补充(请用户填写 — SOP 记录核心)**:
- **横向折相对竖向折/旧折法改了什么、为什么**;横向折分步动作(展平→第1折→…,每步双臂/夹爪做什么、抓哪个角);关键阶段 / 成功判据;采集约定;与 Task_AV1(竖向)、Task_A 的关系(同本体不同折法?能否混训?)。

### 2.B.2 数据准备(200 ep,⚠️ 与 AV1 的切分差异)
- ⚠️ AH1 **正好 200 ep**(单日),无"前 200 之外的剩余"可留 val。→ **从 200 内切**:`Task_AH1` 按 episode_index **前 170 train + 末 30 held-out val**(默认,§2.B.3-Q1 可调)。
  - 备选:train 全 200 + 借 Task_AV1/其它做 val(**不推荐**,跨 SOP 分布不一致,offline 不可比)。
- build:薄脚本/`build_no_release.py --mode raw`,从 `base/v3/2026-06-15-v3` 切前 170 → `Task_AH1_170`(重排 + 视频 symlink),末 30 → `Task_AH1_val`;**norm 对 train 集重算**。
- 夹爪 = 原始 action(同共享配方)。

### 2.B.3 决策定档(✅ 2026-06-16 用户确认沿用 AV1 要求 + 本节差异项)
1. ✅ **数据 = Task_AH1 单日 200ep**;切分 **train 170 / val 30**(按 episode_index 前 170 / 末 30;in-distribution)。
2. ✅ init = warm-start `mixed_1_clean/params`(同共享配方)。
3. ✅ 夹爪 = 原始 action(不裁)。
4. ✅ steps 50k / bs128 / fsdp8 / 8 卡(同共享配方)。
5. ✅ **prompt = 规范化** `"Flatten and fold the cloth. Horizontal Fold v1."`(把数据原值 "Horizontally" → "Horizontal" 以与 "Vertical Fold v1" 平行;**train==deploy 一字不差,覆盖数据原值**)。⚠️ 若你想严格保留数据原串 "Horizontally Fold v1." 请在 Q 里说明。
- **config** 新建 `pi05_task_ah1_hfold_v1_200`(克隆 `pi05_flatten_fold_A_smooth800_dagger_full`):`repo_id`→`Task_AH1_170`、`default_prompt` 同上、`inline_eval_val_root`→`Task_AH1_val`,其余继承 §1。

### 2.B.4 ⭐ 训练结果(✅ 2026-06-17 跑完 50k,cnsh robot-task 8 A100)
> job `t-20260617140704-lfw79`;inline-eval on `Task_AH1_val`(30ep,in-distribution)。⚠️ 数据是**裁尾修复后**那版(源 06-15-v3 被 TOS 重构原地裁尾 → 重建 Task_AH1_170/val 时 video 改 copy、逐 ep parquet==video 校验过)。

| step | MAE@1 | @10 | @25 | @50 |
|---|---|---|---|---|
| 8000 | 0.0097 | 0.0212 | 0.0388 | 0.0625 |
| 16000 | 0.0085 | 0.0183 | 0.0334 | 0.0527 |
| 24000 | 0.0080 | 0.0177 | 0.0329 | 0.0524 |
| 32000 | 0.0078 | 0.0177 | 0.0328 | 0.0525 |
| 40000 | 0.0076 | 0.0178 | 0.0329 | 0.0525 |
| **49999 (best)** | **0.0075** | **0.0179** | 0.0330 | 0.0527 |

- best ckpt:`kai0/checkpoints/pi05_task_ah1_hfold_v1_200/pi05_task_ah1_hfold_v1_200_cnsh/49999/params`(已清非最佳,仅留 49999)。
- @1 单调降到 0.0075;@25/@50 在 24k 即到底(0.0329/0.0524)后平台。**真机未做**(终判)。

---

## 3. 评估(真机为终判)+ 跨范式对比
- **Tier 1 offline**(每范式):自家 held-out val 逐 ckpt **val MAE**(整体 + 夹爪维单列)+ loss → 收敛 + 选 best ckpt。
- **Tier 3 真机**(每范式,决定性):部署 best ckpt 跑**该折法**,看成功率 / 完成率 / 各 sub-phase 通过率 + 夹持稳定(松手/脱落)。
- **跨范式对比**:Vertical Fold v1 vs Horizontal Fold v1 真机表现 → 判**哪种 SOP 更易学/更稳**,反哺采集策略选型。
  - ⚠️ **caveat**:AV1=v2 raw、AH1=v3 前裁+尾裁,trim 不同是混杂变量;严格对比需对齐 trim(或都各自报告,差异归因时注明)。另两者 ep 时长分布、采集量也不同。
- **可选**:与旧 SOP(Task_A horizontal smooth800)基线对照。

---

## 4. 落地步骤(每范式独立走一遍)
1. **补全该范式 SOP 物理流程**(用户;§2.x.1 待补充)。
2. **build** 该范式 train/val 集(切分 + 重排 + 视频 symlink)+ **重算 norm**。
3. **注册 config**(§2.x;克隆 flatten-fold pi05),git commit/push。
4. **提交 8 卡训练**(50k,每 2k save)。
5. **eval**:val MAE → 选 ckpt → 真机 rollout。
6. **回填** results.md + 更新 master history;跨范式对比落 §3。

---

## 关联
- 数据:`kai0/data/Task_AV1/base/v2/`(TOS `KAI0/Task_AV1`)· `kai0/data/Task_AH1/base/v3/2026-06-15-v3/`(TOS `KAI0/Task_AH1`,已落地)
- TOS 同步:`train_scripts/kai/data/sync_task_av1_from_tos.sh` + watchdog(⚠️ **当前因 TOS 重构暂停**,见记忆 [[project_tos_sync_paused_restructure]];AH1 为本次一次性手动拉取)。
- 夹爪处理对照:[`gripper_action_clip_experiment.md`](gripper_action_clip_experiment.md)(action≡state / 夹爪给力)
- config 克隆源:`kai0/src/openpi/training/config.py:1798`(`pi05_flatten_fold_A_smooth800_dagger_full`)
- 相关:Task_A + Task_AV1 混合 co-train [`pi05_task_a_av1_mixed_1to1_plan.md`](pi05_task_a_av1_mixed_1to1_plan.md)
