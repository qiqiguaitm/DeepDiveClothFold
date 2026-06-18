# X-VLA proprioception 捷径 / vision-blind 开环 — 根因认证与修复训练规划

> **目的**: 认证"X-VLA 真机不抓衣服、固定动作"的根因, 给出 probe 证据 + 可执行的修复训练方案 + 离线门禁。
> **建立**: 2026-06-09 · **方法**: 离线 vision-ablation (复用 serve infer 路径重放真机 trace, 固定 seed + 关 proprio-feedback), 不训练。
> **关联**: [`xvla_track_x_curriculum.md`](xvla_track_x_curriculum.md) (Track X, p0/d5anchor 来源) · [`xvla_camera_robust_grasp_final.md`](xvla_camera_robust_grasp_final.md) (相机 gap, 假定模型读视觉的前提被本文推翻) · memory `reference_xvla_vision_blind_openloop` / `reference_vision_ablation_openloop`。
> **诊断工具**: [`train_scripts/kai/eval/eval_xvla_vision_ablation_offline.py`](../../../../train_scripts/kai/eval/eval_xvla_vision_ablation_offline.py)

---

## 0. 结论 (一句话)

**整条 X-VLA smooth800 管线 (p0 + d5anchor) 训练就是纯开环 (vision-blind): 动作是 proprioception 的纯函数, 三路相机像素对输出的影响 = 0.000。不是数据问题、不是部署问题、不是 qdur/归一化问题。在修好 `use_proprio` 捷径前, 换任何数据/qdur/norm 训出的 X-VLA ckpt 都会瞎。**

这也推翻了 Track X 之前的诊断链 ("X3.C 真机失败 = R1 缺 ImageNet 归一化 → 重训 p0 修复"): p0 已修 R1 + ImageNet 归一化, 真机**依旧失败**, 因为真根因是 proprio 捷径, R1 只是表层。

---

## 1. 认证证据 (离线 vision-ablation)

方法: 用 `eval_xvla_vision_ablation_offline.py` 复用 **serve 的 `XVLAServerPolicy.infer` 路径** (预处理与真机逐字节一致), **固定 seed + `proprio_feedback=OFF`** → 每次推理是 `(image, state)` 的独立确定函数。从真机 `--trace` dump 重放真实 obs, 做三种扰动比对 action chunk:

| 扰动 | d5anchor (trace 11:55) | p0 (trace 06-07) | 含义 |
|---|---|---|---|
| **换一张完全不同的图** (state 不变) | xyz **0.03mm** / grip 0.0000 | xyz **0.03mm** / grip 0.0000 | 图像内容对动作无影响 |
| **整张图置黑** (state 不变) | xyz **0.07mm** / grip 0.0000 | xyz **0.08mm** / grip 0.0000 | 删掉视觉对动作无影响 |
| **换 proprio state** (图不变) | xyz **311mm** / grip 0.041 | xyz **248mm** / grip 0.012 | 本体一动, 动作大变 |
| **视觉/本体影响比 (d_img/d_state)** | **0.000** | **0.000** | →0 = 纯开环 vision-blind |

> 对照健康基线: 同一批 smooth800 数据上, **pi0** 之前测出 blank/real MAE = **13.6×** (视觉健康, 见 `reference_vision_ablation_openloop`)。X-VLA 在同数据上是 **0.000**。

### 1.1 ✅ 阳性对照 (2026-06-11) — 官方原版 X-VLA-SoftFold **读视觉** (≈300–400× 我们)

> 验证"我们 ablation 测出的 `d_img≈0` 是模型属性, 不是 harness 天花板", 同时确认根因在数据链而非架构/部署: 拉官方 raw 2toINF `X-VLA-SoftFold` ckpt (与我们**同架构同任务同机器人**: 双臂 Agilex Piper, flatten+fold), 走**官方自带代码** (`models.modeling_xvla.XVLA` + `XVLAProcessor` → `generate_actions`, `domain_id=5` SoftFold, 官方 instruction), 喂**同一条 KAI0 trace** (`trace_20260601_192213`, 同机器人 → 近 in-distribution; ee6d proprio 直接取 npz `state20`, 20 维布局逐位一致), 固定 seed, n=12。脚本 [`eval_xvla_OFFICIAL_vision_ablation.py`](../../../../train_scripts/kai/eval/eval_xvla_OFFICIAL_vision_ablation.py), 结果 `logs/xvla_official_vision_ablation.json`。

| 扰动 (xyz L2, mm) | **官方 X-VLA-SoftFold** | 我们 smooth800 (§1) | 倍数 |
|---|---|---|---|
| **换一张图** (state 不变) `d_img` | **12.87mm** | 0.03mm | **≈320×** |
| **整图置黑** (state 不变) `d_blank` | **34.20mm** | 0.07–0.08mm | **≈430×** |
| 换 proprio (图不变) `d_state` | 58.47mm | 248–311mm | — |
| `d_img/d_state` | 0.220 | **0.000** | — |
| `d_blank/d_state` | 0.585 | ~0.000 | — |

- **逐帧 12/12 的 `d_img` 都在 9–21mm, 没有一帧塌到 0**; 置黑扰动 > 换帧扰动 (34 > 13mm, 符合"全黑更 OOD"); 夹爪也随图像变 (`g_img` 0.04–0.15)。→ **官方原版模型视觉通路是活的, 没有脱离视觉现象**。
- **决定性**: 官方与我们**架构/部署/归一化/动作表征/proprio 早融合逐项相同**, 唯一实质差异 = **训练数据 action 语义** (官方真实 action≠state vs 我们 relabel 成 ≡state)。同架构官方能读视觉 → **根因坐实在数据链, 非架构/部署** → 直接支撑 §4.1 (E1) 结论并把 **E0 (真实 action≠state) 锁为必需主路径**。
- **门禁可信度**: 本对照证明 ablation harness **能**检出视觉敏感 (官方 `d_img` 远 > 0)。因此我们模型的 `d_img≈0` / E1 的 `d_img=0.00` 是**真·vision-blind**, 不是测不出。
- **caveat**: `ratio 0.220` 未到 1.0 — 因 KAI0 ee6d proprio 喂官方 SoftFold 模型存在坐标系/标定偏差, proprio 偏 OOD 抬高了 `d_state` (且官方本就 `use_proprio=True`, 有 proprio 敏感属健康)。**决定性判据是绝对 `d_img`/`d_blank` ≫ 0** (与本文一贯口径一致, 不看比值伪影)。更干净版本可用真正官方 SoftFold obs, 但 300–400× 量级差已足够定论。

---

## 2. 已排除的假设 (为何不是数据/部署/时序)

| 假设 | 排除证据 |
|---|---|
| **部署 bug (图没喂进模型)** | `config.image_features` = serve 喂的三键 `observation.images.image/image2/image3`; `num_image_views=3`, `empty_cameras=0` → 三路全消费、无 zero-pad; `resize_imgs_with_padding=(224,224)` 对齐。`_prepare_images` 不报错 = 键命中。**全部三路置黑仍 0 变化** → 被消费的视图确实无影响。 |
| **数据质量 (5-19~5-27 漂移)** | p0/d5anchor 用的是 **smooth800 好数据 (04-23~05-09)**, 非漂移期。同数据 pi0 视觉健康。 |
| **qdur / publish_rate 时序** | 已修正: d5anchor `publish_rate:=15` 后速度回到 1.00× (34/35mm/s), 行为**不变**, 仍不读视觉。时序与视觉盲是两件独立事。 |
| **ImageNet 归一化 (R1)** | p0 已修 R1, sidecar `image_norm=imagenet`, ablation 用 `imagenet_norm=True`, 仍 0.000。 |
| **采样随机性** | seed 固定, 三扰动同 seed; state 扰动能产生 248~311mm 变化 = 推理管线本身正常响应输入。 |

---

## 3. 根因 (2026-06-09 深度调研定性): 数据约定 × 架构敏感性

**一句话: 我们的 KAI0 数据用了 `action[t] ≡ observation.state[t]` 的 relabel 约定 (`relabel_action_eq_state.py`), 这给了"复述本体即可低 loss"的捷径; X-VLA 的架构把 proprio 早融合进每个 action token, 对这个捷径极度敏感 → 直接塌缩成 vision-blind。pi0 同数据不塌缩 = 架构耐受性不同。不是部署问题, 不是训练 loop 问题, 是数据约定与 X-VLA 架构不兼容。**

### 3.1 数据: action ≡ state (relabel 约定)
- `relabel_action_eq_state.py` (workspace 根): "Relabel a LeRobot-v2 dataset so action[t] := observation.state[t] (bit-identical)", rationale="Match official KAI0 upstream convention (action == state)"。
- 本地实测 `A_new_pure_200_val`: `mean|action−state| = 0.00000000`, 而真实帧间运动 `|state[t+1]−state[t]| ≈ 0.005 rad` → action 列相对当前 state **零新增预测信息**。
- pi0 good ckpt `task_a_new_smooth_800` 的 norm_stats: `action.mean ≡ state.mean` (差 2.6e-7) → **pi0 也是 action≡state 训的**。X-VLA `A_new_smooth_800_xvla` 由同批 smooth800 关节经 `joint_to_ee6d` 转 EE6D (action/state 同样转换) → action_ee6d ≡ state_ee6d 传导成立。

### 3.2 架构: X-VLA 把 proprio 早融合进每个 action token
- 官方 `transformer.py:372`: `action_tokens = cat([action_with_noise, proprio_tokens, time_tokens])` — proprio 广播到**每个 action 位置**、特征级拼接、每个去噪步都在。`use_proprio` 默认 True。
- 配上 action≡state + 绝对 EE6D (首步 action[0] = 当前 state = proprio 输入本身) → 模型把"要预测的东西"(action≈state) 和"答案"(proprio=state) 摆在一起 → 平凡复制, 梯度不流向视觉。

### 3.3 决定性对照: 同数据, pi0 健康 / X-VLA 全瞎
| 模型 | 数据 (均 action≡state smooth800) | 视觉依赖 |
|---|---|---|
| **pi0 `task_a_new_smooth_800`** | action≡state ✓ | blank/real MAE **13.6×** 健康 (`reference_vision_ablation_openloop`) |
| **X-VLA p0 / d5anchor** | action≡state ✓ | 视觉/本体比 **0.000** 全瞎 |

→ **数据约定是必要条件 (使能捷径), 但是否塌缩取决于架构**。pi0 (state 走独立 token, 经 attention 弱耦合) 耐受; X-VLA (proprio 每 token 特征级早融合) 不耐受。这解释了为何只有 X-VLA 瞎。

### 3.4 官方 X-VLA 为何不瞎 (开源对照)
- 官方仓库 **github.com/2toinf/X-VLA** 的 SoftFold-Agilex (同任务) 用**真实 teleop/leader action** (action ≠ realized state), 且 `base.py:157` **丢弃 static 帧** (`|action[1]−action[0]|<1e-5 → continue`) — 专门剔除 action≈proprio 的退化样本。**我们的 `multi_domain_dataset.py` 无此 skip**。
- 官方部署/训练/归一化/动作表征与我们**逐项一致** (proprio 都喂、绝对 EE6D、ImageNet norm、domain token) — 唯一实质差异在**训练数据的 action 语义** (真实 action vs relabel 成 state) + static-skip。
- 文献佐证: causal confusion (de Haan 2019)、copycat problem、ReViP (arXiv 2601.16667, "state-dominant bias → false completion") 都记载 "绝对动作 + proprio 输入 + 程式化数据 → 忽略视觉"; 标准解 = state/proprio dropout、用真实未来动作、keyframe 重加权。

同时解释长期困惑"MAE 没问题但真机不可用": action≡state 下**开环复述本体即低 MAE**, MAE/val-loss 完全测不出视觉盲 (`feedback_offline_eval_protocol` / `feedback_real_machine_oscillation_data_tail`)。

---

## 4. 修复训练规划

修复可断**两条链**任一: ① 数据链 (让 action ≠ state, 官方做法) ② 架构链 (拿掉/遮蔽 proprio)。

### 4.0 实验 E0 — 数据正解 + 完全按官方配方 (主路径, ⭐ 进行中 2026-06-18)
- **数据 (已建好)**: 用 Task_A `v1` (`kai0/data/Task_A/vis_base/v1`) — 真实 teleop `action`, 满足 **action ≠ state** (非 relabel 自述)。转 EE6D (interleaved Rot6D + 二值夹爪, `joint_to_ee6d.py`, **observation.state / action 各自 FK, 不 relabel**) → `xvla/data/self_built/A_v1_noRelabel_ee6d/<6 日期 04-23..04-30>`, **639 ep / 603,242 chunk-sample**, domain_id=20。校验: `mean|action−state|xyz=4.37mm`(action≠state ✓)、`|action[t]−state[t+1]|=2.21mm`(setpoint ✓)、gripper∈{0,1}、20D。
  - 注: v1 是 **kai0-native schema** (`episode_id`, 非 LeRobot `episode_index`); loader 已加双 schema 兼容 + stale-manifest skip (04-24 ep53 parquet 已清但 manifest 残留 → 自动跳过并打印)。
  - **static-frame skip (官方有, E0 已实现)**: upstream `datasets/domain_handler/base.py` 丢弃"未来首步双臂 EE 位姿 `|seq[1]−seq[0]|<1e-5`"的退化帧 (action≈proprio 的 copycat 温床)。loader 加 `static_skip` (默认 off, E0 开), 镜像官方逻辑在 EE6D 动作轨迹上判 (排除夹爪维)。**实测 v1 剔除 21,529 帧 (~3.6%)**: 603,242 → **581,713** sample (机器人暂停段确实存在, 非噱头)。
- **配方 (完全按官方 `X-VLA/train.py`, config `E0_v1_official`)**:
  | 项 | 官方 / E0 |
  |---|---|
  | 精度 | bf16 mixed (autocast) |
  | weight_decay | **0.0** |
  | param groups | **4 组**: vlm & soft_prompt ×0.1 (1e-5); transformer_core & action_head ×1.0 (1e-4) |
  | freeze | 前 **1000** 步冻 **vlm + transformer_core** (仅 soft_prompt+action_head 训练) |
  | LR schedule | **constant** + warmup 2000 (官方 `use_cosine_decay` 默认 OFF) |
  | lr / iters | 1e-4 / **50k** |
  | batch | per-device **16** × 8 卡 = eff **128** |
  | action 表示 | `action_qdur=2.0` (intention abstraction, 对齐官方 real_world.py) |
  | static-skip | `static_skip=True` (官方 base.py: 丢未来首步双臂几乎不动的退化帧) |
  | 图像 | ImageNet norm + ColorJitter(0.2) |
  - **修了一个真 bug**: 旧 2-group 用 `"florence"/"vision"` name-keying 漏掉 `model.vlm.language_model.*` → VLM 文本编码器一直被 10× LR(1e-4 而非 1e-5)训练。4-group 用 `.vlm.` 匹配整个 VLM 修复。
  - trainer 改动全部 gated 在 config key 后 (`param_groups`/`bf16`/`lr_schedule` 默认=旧行为) → 历史 config (X3*/E1/A_0423…) 行为不变。
- **目的**: 复现官方 SoftFold 数据性质 — proprio 无法平凡预测真实未来 → 强制读视觉, 同时**保留 proprio 精度/平滑收益**。治本。
- **状态 (2026-06-18)**: 数据建好+校验 ✓ | 官方对齐 trainer + `E0_v1_official` config + volc yaml ✓ | **smoke (gf0 2×A100, max_steps=80) 通过** ✓ (model 载入 / 6 数据集 / static-skip 581,713 sample / 4-group [vlm601+core295+head4+soft1] / freeze 896 tensor / bf16 fwd-bwd / constant sched / ckpt 落盘 全通) | **待提交 volc 8×A100 正式 50k**。
- **判据**: `eval_xvla_vision_ablation_offline.py` 视觉/本体比 ≳0.5 + 真机会找衣服。

### 4.1 实验 E1 — 确诊性 A/B: `use_proprio=False` (已就绪, 最快)
- **做法**: 复制 p0 训练 config, 仅置 `use_proprio: false` (关 state 输入), 其余 (数据 smooth800 / 60k / lr 1e-4 / ImageNet norm) 完全不变。
- **目的**: 强制模型只能用视觉。是**确诊**而非最终模型 — 验证"只要拿掉 proprio 捷径, 模型就会读视觉"。
- **判据**: 训完跑 `eval_xvla_vision_ablation_offline.py`, **视觉/本体影响比从 0.000 抬到 ≳0.5** = 根因坐实。再上真机看是否会"找衣服"。
- **风险**: 丢掉 proprio 平滑信号, 连续控制可能更抖 (可叠 publish-time EMA 缓解)。

> #### ❌ 结果 (2026-06-10) — E1 **未能让模型读视觉, 假设被推翻**
> 训完 60k(`xvla_x3c_smooth800_noproprio/step_final`,loss 109→~5),离线 vision-ablation(trace `2026-06-01_192213`,n=12,imagenet-norm,`--no-proprio` 加载 proprio_dim=0 ckpt,load missing=0/unexpected=0):
>
> | ckpt | swap-IMAGE d_img | swap-STATE d_state | 解读 |
> |---|---|---|---|
> | **p0**(proprio ON,对照) | 0.04mm | **278.6mm** | 视觉盲 + proprio 开环(复现 §1 基线 ✅,验证门禁/trace 可信) |
> | **E1**(proprio OFF) | **0.00mm** | **0.00mm** | proprio 依赖被拿掉(d_state 279→0,符合设计),但 **视觉仍没起来(d_img 仍 ~0)** |
>
> - **比值 3605 是假阳**(d_img/d_state = 0.00/~0 的数值伪影);**真正指标 d_img 没有上升** → E1 **不通过门禁**(视觉没活)。
> - **结论**: "拿掉 proprio 捷径 → 模型就会读视觉"(§4.4 "断架构链不依赖数据")**被实测推翻**。模型在无 proprio 下**换了一种开环方式 = 复述记忆里的平均折叠轨迹**(对任何输入近乎常量输出)。
> - **含义(强化 E0)**: 任务太程式化 + 数据 `action≡state` → **视觉从来不被需要**,光断架构链不够,模型总能找到开环解。**根因主要在数据链 → E0(真实 action≠state)从"推荐"升级为"必需"**;E2(proprio-dropout)预计同样不够。
> - 结果 json:`logs/xvla_e1_vision_ablation.json`。门禁脚本已加 `--no-proprio/--imagenet-norm/--prompt` + `XVLA_BART_TOK` 支持。

### 4.2 实验 E2 — proprio-dropout (更好的最终模型, 需 patch)
- **做法**: patch lerobot XVLA 训练, 对每个样本以概率 `p_drop` (起步 0.5) 把 `observation.state` 置零/替换占位 token; 推理仍可给 proprio。
- **目的**: 保留 proprio 收益 (平滑/精度) 又强制模型读视觉。causal confusion 的标准解。
- **依赖**: E1 确诊为 proprio 捷径后再投入实现成本。
- **判据**: 同 E1 门禁 (比值 ≳0.5) + 真机抓取成功率 ≥ E1。

### 4.3 实验 E3 — 数据多样性 (治本, 最慢, 次选)
- **做法**: 采集/合成"同 proprio (同 home 起点) 下衣服位置不同 → 需不同抓取目标"的样本, 打破 proprio→action 确定性。
- **目的**: 即使有 proprio, 也强迫视觉成为消歧的唯一信息源。
- **判据**: 同门禁 + 跨衣服位置泛化。

### 4.4 对照矩阵

| 组 | use_proprio | proprio-dropout | 数据 action | static-skip | 终判 |
|---|---|---|---|---|---|
| **baseline (现状)** | True | — | ≡state | 无 | 视觉比 **0.000** ❌ |
| **E0 (数据正解)** | True | — | **真实 action (≠state)** | **加** | 视觉比 ≳0.5 + 真机找衣服 + 保 proprio 精度 |
| **E1 (确诊, 已跑)** | **False** | — | ≡state | 无 | ❌ **d_img 仍 0.00mm**(视觉没活)— 断架构链**不够**, 模型转常量开环 |
| **E2** | True | **0.5** | ≡state | 无 | 视觉比 ≳0.5 + 真机成功率(E1 已示警: 单断 proprio 恐不够) |
| **E3** | True/dropout | 0.5 | 真实+位置多样 | 加 | 跨位置泛化 |

> ~~E1 断架构链 → 最快确诊~~ → **实测推翻 (2026-06-10)**: 无 proprio 模型并不读视觉, 而是复述记忆里的平均轨迹(d_img 仍 ~0)。说明 **数据链 (action≡state + 任务程式化 → 视觉从不被需要) 才是主因**, 单断架构链不足以唤起视觉。**E0 (真实 action≠state) 从"治本可选"升为"必需主路径"**。

---

## 5. 离线门禁 (新增, 强制)

**以后任何 X-VLA ckpt 上真机前, 必须先跑离线 vision-ablation, 视觉/本体影响比 ≳ 0.5 才放行** (类比 pi0 的夹爪 SNR≳15× 门禁)。MAE / val loss **不作为视觉依赖判据** (测不出开环)。

```bash
CUDA_VISIBLE_DEVICES=<free> kai0/.venv_xvla/bin/python \
  train_scripts/kai/eval/eval_xvla_vision_ablation_offline.py \
  --trace /tmp/xvla_stack/trace_<ts> \
  --ckpt /data1/DATA_IMP/checkpoints/ckpt_xvla/<ckpt> --n 12
```
读 "视觉/本体影响比": →0 = vision-blind 禁止上机; ~1 = 健康闭环。

---

## 6. 行动顺序

1. **停止盲调 X-VLA ckpt** (换 qdur/norm/数据日期都不会改变 vision-blind, 根因在 action≡state × 架构)。
2. ✅ **E1 (`use_proprio=False`) 已跑 (2026-06-10) → 推翻"断架构链就够"**: 无 proprio 模型 d_img 仍 0.00mm(不读视觉),转成常量开环。**根因主要在数据链**。
3. ✅ **E0 数据问题已解决 (2026-06-18)**: 不需回更上游 —— Task_A `v1` 本身就是 action≠state 的真实 teleop(非 relabel)。已转 EE6D = `A_v1_noRelabel_ee6d`(639 ep/603k sample), 完全按官方配方建 `E0_v1_official` config + 官方对齐 trainer, **smoke (gf0 2×A100) 通过**。
4. ⭐ **下一步 = 提交 volc 8×A100 跑满 50k**(eff batch 128 = 官方量级)。yaml: `train_scripts/kai/volc/xvla_e0_v1_official_cnsh_8gpu.yaml`。
5. E0 出 ckpt → 过 §5 门禁(视觉比 ≳0.5)再上真机。E2 (proprio-dropout) 降级为"E0 不可行时的退路", 且 E1 已示警单断 proprio 恐不足, E2 需配合数据侧。
6. 把 §5 门禁纳入 X-VLA 上机流程。
