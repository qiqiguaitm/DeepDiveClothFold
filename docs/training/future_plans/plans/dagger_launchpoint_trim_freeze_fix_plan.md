# dagger clip 双向"起爆点"前裁 —— 回折冻结修复实验 plan

> **建立**: 2026-07-10
> **目的**: 把 dagger clip **前后双向裁到"起爆点"**(前砍迟疑起手、后砍静止收尾,只留果断动作核心),重 build → 重训 → 真机验证:能否**修掉回折过渡冻结**,同时**保留 fresh dagger 的夹爪新语义**。
> **状态**: ✅ **实验完成 — H 成立**(2026-07-14)。真机部署 best ckpt 49999,**回折冻结已修复**。双向起爆点前裁是修法。见 §7.1 结果 + 执行记录 §8.6。
> **上游诊断**: 见 [`pi05_v4_awbc_modeB_freeze_diagnosis_plan.md`](pi05_v4_awbc_modeB_freeze_diagnosis_plan.md) §9。**最可能根因** = dagger clip 从 **06-16 起**把"操作员接管卡住态后迟疑 ~0.5s 再动"的起手段裁进 clip → 被 AWBC 标 positive → 部署永远喂 positive → 回折过渡(卡住样决策态)触发"迟疑不动" → 剂量够(任务②)则冻。
> ⚠️ **铁律**: 真机为终判;夹爪维单列看。

---

## 0. 假设(本实验要证/证伪)
**H**: 冻结由 dagger clip 的**边界段**(前:迟疑起手 v<0.02;后:静止收尾 v<0.008)注入 —— 这些低速帧被 AWBC 标 positive、被部署 always-positive 放大成"决策态迟疑不动"。**双向裁掉边界、只留果断动作核心 → 不冻**。
- **不冻** → H 成立 → 前裁起爆是修法,fresh dagger 可用(夹爪也修)。
- **仍冻** → 边界段不是根因,是碎片/剂量本质 → 转 velocity-aware discretize / 重采 / 部署兜底。

---

## 1. ⭐ 双向"起爆点"前裁算法(逐 dagger clip)
对每个 dagger clip(仅 dagger,**base demo 不裁** —— base 是完整任务、非卡住态接管):
1. 臂速 `v[t] = ‖action[t]−action[t−1]‖`(12 臂关节 idx 0-5,7-12;**排除夹爪 6/13**)。
2. 平滑 `v̄[t]` = 5 帧滑动均值(去噪)。
3. **前起爆点** `t_start` = 首个满足 `v̄>THR` 且**连续 ≥K 帧**的 t(果断起手)。
4. **后起爆点** `t_end` = 末个 `v̄>THR` 的 t(最后果断动作)。
5. 保留 `[t_start−M : t_end+M]`(留 margin M 帧,避免切太狠)→ 果断动作核心。
6. **参数**(待微调):`THR=0.02`(果断阈值,来自逐天分析:果断 dagger 起爆 >0.02)、`K=5`、`M=5`、`MIN_LEN=30`(1s)。
7. **护栏**:
   - 全程无 `v>THR`(纯 hold clip)→ **丢弃**(坏/idle clip)。
   - 裁后 `<MIN_LEN` → 丢弃(记数,别静默)。
   - 裁后仍需 3 相机 + parquet 帧一一对齐。
8. **视频同步裁 + PTS 归零**:parquet 裁 `[t_start:t_end]` 行 → 视频 mp4 裁同帧段并 **reset PTS**(⚠️ 复用 `build_no_release.py` 的裁剪+PTS归零机制,见 [[reference_v3_trim_video_pts_bug]] 别再犯 PTS 不归零 bug);frame_index/timestamp 重排。

**范围**:对**所有 dagger 日期统一裁**(05-29~最新)。06-16 前本就果断(起爆 0.03-0.17s),裁掉的前段极小、无害;统一处理保证一致、不引入日期混淆。

---

## 2. 数据构建(裁后重走 AWBC pipeline)
1. **裁**:对 `vis_dagger/v4/<所有日期>` 逐 clip 双向起爆点前裁 → 落 `vis_dagger/v4_launchtrim/<date>`(新目录,不覆盖源)。记录:各日期裁前/裁后帧数、丢弃 clip 数、平均裁掉的前/后段长度。
2. **build merged**:base(**不裁**,13 日期)+ **裁后 dagger** → `self_built/A_v4_base_dagger_launchtrim`(仿 `build_v4_awbc_merged.py`,删 intervention、symlink 裁后视频、episode 重排)。
   - **主实验(任务②-analog)**:base + 裁后(旧 dagger 05-29~06-23 **+ fresh 06-29~07-03**)→ 直接对标已冻的任务②。
3. **重算 norm_stats**(裁后集,action_dim=32)。
4. **AE 打标**:复用 `adv_est_v1`(step 100000)→ `absolute_advantage`。
5. **discretize**:binary top-30%(与所有 v4 AWBC 一致,保持可比)。

---

## 3. 训练规格(单变量 vs 任务②)
- **config** 新建 `pi05_v4_awbc_launchtrim`(克隆 `pi05_v4_awbc`,**无 DCT**):
  - `repo_id` → `A_v4_base_dagger_launchtrim`(裁后 labeled);`prompt_from_task=True`;`use_delta_joint_actions=False`。
  - init=`pi05_base`,LR warmup1k/peak1.5e-5,50k,bs128,fsdp8,EMA0.9999。
  - **唯一变量 vs 任务②(plus_freshdagger)= dagger clip 双向起爆点前裁**(其余逐字段同)。
- **8 卡**,gf3/cnbj 择空闲。

---

## 4. 评估(真机为终判)
| Tier | 做法 |
|---|---|
| offline | 裁后 val 逐 ckpt val MAE(整体+夹爪维单列)+ loss sanity |
| **真机(决定性)** | **回折过渡是否还冻 >5s**(与任务② 同协议对照)+ 夹爪微张开是否修复 + 成功率 |

**判据**:
- ✅ **H 成立** = 真机**无回折冻结** + 夹爪修复 → 前裁起爆是修法,采用;fresh dagger 可正常用。
- ⚠️ 缓解未消 = 冻结时长缩短但仍有 → 边界段是部分诱因,叠加 velocity-aware discretize。
- ❌ 仍冻 = 边界段非根因 → 转诊断分支(discretize / 重采 / 部署兜底 §diagnosis-plan)。

**对照**:任务②(裁前,已冻)↔ 本实验(裁后)= 单变量"是否裁边界"→ 直接归因。

---

## 5. 落地步骤
1. **实现前裁脚本** `launchpoint_trim_dagger.py`(算法 §1;复用 build_no_release 裁剪+PTS归零)→ 干跑打印各日期裁前后统计,人工核验 THR/K/M 合理(抽几个 clip 看裁得对不对)。
2. **裁** 全 dagger → `vis_dagger/v4_launchtrim/`。
3. **build** `A_v4_base_dagger_launchtrim`(base + 裁后 dagger + fresh)+ 重算 norm。
4. **AE 打标 + discretize** top-30% → labeled。
5. **注册 config** `pi05_v4_awbc_launchtrim`,commit/push。
6. **8 卡 50k 训练**。
7. **真机** vs 任务②,落 §4 判据。
8. 回填 diagnosis-plan §9 结果 + master history。

---

## 6. 风险 / 注意
- **THR 过大**:把慢但有效的精细操作也当"非果断"裁掉 → 丢失真实动作。先干跑抽检、`THR=0.02` 保守起,必要时降。
- **裁后 clip 太短/丢太多**:统计丢弃率;若某日期丢弃过多(说明该日期多为迟疑/idle),单独看是否该整段弃。
- **视频 PTS**:裁视频**必须 reset PTS**,否则 vision↔action 错位静默训坏(见 [[reference_v3_trim_video_pts_bug]])。裁后抽 ep 验证帧数=parquet 行数、PTS 从 0 单调。
- **base 不裁**:base 是完整任务 demo,裁边界会破坏"完整流转"覆盖(那正是不冻的保障)。只裁 dagger。
- **夹爪语义**:裁只动时间范围,不改 action 值 → fresh dagger 的 gripper-from-master 完整保留。
- **未分离"起手 vs 碎片"**:本实验裁的是**边界段**;若裁后不冻,证明是边界(迟疑起手+静止收尾);clip 仍短 → 顺带证明不是"clip 长度/碎片"本身。

---

## 7. 决策定档
- ✅ `THR / K / M / MIN_LEN` 参数:**0.02/5/2/30**(用户选 M=2 更狠,前裁 avg 0.33s/后裁 0.45s,整体保留 98%)。
- ✅ 裁范围:**全 dagger**(05-29~07-07 统一处理)。
- ✅ 集群:cnbj Robot-North-H20 8×H20。
- ✅ Option A:复用逐帧 task_index(非重打标,免 AE ckpt;与任务②严格单变量)。

## 7.1 ✅ 真机结果 (2026-07-14)

**判据对照**:

| 对照臂 | 冻结 | 说明 |
|---|---|---|
| 任务② (plus_freshdagger,裁前) | ❌ 冻 | dagger 含迟疑起手边界段 |
| **本实验 (launchtrim,裁后)** | ✅ **无冻结** | 双向起爆点前裁 |

**结论: H 成立** — dagger clip 的迟疑起手 + 静止收尾边界段是冻结根因,双向起爆点前裁是修法。fresh dagger 本身可用(夹爪语义保留),只需裁掉边界。

**根因链确认**:
1. 06-16 起 dagger clip 含 ~0.5s 迟疑低速起手(v<0.02) + 静止收尾
2. AWBC 把低速帧标 positive + 部署 always-positive → 决策态迟疑不动
3. 双向裁到起爆点只留果断核心 → 不冻
4. **与任务②唯一变量 = dagger 是否裁边界 → 单变量归因成立**

---

## 执行记录(2026-07-12)

**参数定稿**: THR=0.02 / K=5 / **M=2**(用户选"更狠一点", M从5→2 裁掉更多迟疑; 干跑核验新日期起爆点前段vbar=0.000~0.002确为迟疑, 前裁avg0.33s/后裁0.45s, 整体保留98%)。

**⭐ 关键偏离 — Option A 复用逐帧 task_index(非重打标)**: adv_est_v1 AE ckpt 已全网删除。但 `absolute_advantage[n]=V(f0,f_{n+int})−V(f0,f_n)` 是同参考帧差分, progress(f0)抵消→近似参考帧无关; 且 `task_index`(pos/neg)在labeled源里100%完整(absolute_advantage列仅~85%ep有)。故**直接复用逐帧task_index切到保留帧, 免重打标/discretize/AE ckpt**——这也正是任务②的做法(复用现成task_index未重discretize)→ **与任务②严格单变量**(仅dagger帧被裁; 裁掉的迟疑/静止边界帧整段移除, H假设直接被测)。§2.4-2.5 的"AE打标+discretize"因此跳过。见 [[reference_launchtrim_pipeline_northE]]。

**构建位置**: North-E/gf3 原生(gf3 180核+8×H20全空闲)。North-E本就是v4全镜像(vis_base/v4+vis_dagger/v4+venv+A_v4_base_dagger labeled), 只送 freshdagger_ft标签113MB。`build_launchtrim_from_labeled.py`(KAI0_ROOT切North-E)→ **A_v4_base_dagger_launchtrim=2510ep**(1200 base整段 + 1310 dagger前裁, drop2), 列与任务②一致, 抽检帧全对齐(PTS归零正确)。

**config**: `pi05_v4_awbc_launchtrim`(克隆pi05_v4_awbc无DCT, North-E路径, init pi05_base, warmup1k/peak1.5e-5/50k/bs128/fsdp8/EMA0.9999, inline_eval关)。直接插入North-E config.py(get_config验证通过)。

**提交**: `pi05_v4_awbc_launchtrim_cnbj_8gpu.yaml` → job `t-20260712080450-7f5kq`(cn-beijing/Robot-North-H20/1-host 8×H20/50k)。

**待办**: ~~训练完 → offline val MAE → 真机 vs 任务② → 回填~~ 全部完成见 §7.1。

### 8.6 训练结果 (2026-07-14)

| 项 | 值 |
|---|---|
| job | `t-20260712080450-7f5kq` (cnbj Robot-North-H20 8×H20) |
| best ckpt | `/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/checkpoints/pi05_v4_awbc_launchtrim/pi05_v4_awbc_launchtrim_cnbj/49999` |
| loss 收敛 | 0.70 → 0.003 (230×, 训练健康) |
| param_norm 增幅 | 0.16% (未过拟合) |
| val MAE | N/A (AWBC prompt_from_task=True, val 无 advantage prompt → inline-eval 失败, 预期内) |
| **真机冻结** | ✅ **不再冻结 — H 成立** |
| **夹爪** | ✅ 修复 (fresh dagger 夹爪语义保留) |

## 关联
- 上游诊断 + 根因: [`pi05_v4_awbc_modeB_freeze_diagnosis_plan.md`](pi05_v4_awbc_modeB_freeze_diagnosis_plan.md)(§9 逐天 06-16 变点 + 迟疑起手机制)
- 裁剪+PTS 机制复用: `train_scripts/kai/data/build_no_release.py`(per-date front-trim + tail-cap + PTS) · [[reference_v3_trim_video_pts_bug]]
- build 合并源: `train_scripts/kai/data/build_v4_awbc_merged.py`
- 对照 config: `pi05_v4_awbc`(不冻基线)· `pi05_v4_awbc_plus_freshdagger`(任务②,裁前,已冻)
- AE / discretize: `kai0/checkpoints/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/adv_est_v1` · `kai0/stage_advantage/annotation/`
- 数据: `kai0/data/Task_A/vis_dagger/v4/*`(源)→ `vis_dagger/v4_launchtrim/*`(裁后)
