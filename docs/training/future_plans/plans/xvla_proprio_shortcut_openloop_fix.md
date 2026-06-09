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

### 4.0 实验 E0 — 数据正解: 用真实 action 重训 (官方等价, 推荐主路径)
- **做法**: X-VLA 训练数据**不用 action≡state relabel 版**, 改用原始 teleop/leader `action` (真实未来轨迹, action ≠ state); 并在 `LeRobotEE6DDataset`/`multi_domain_dataset.py` 加官方 static-frame skip (`|action[1]−action[0]|<1e-5 → continue`)。其余 (proprio 仍开、smooth800 同批 episode、60k) 不变。
- **目的**: 复现官方 SoftFold 的数据性质 — proprio 无法平凡预测真实未来 → 强制读视觉, 同时**保留 proprio 的精度/平滑收益**。这是治本。
- **前置**: 确认 smooth800 源是否存在未 relabel 的原始 action 版本 (vePFS `A_new_smooth_800/base` 的 action 列是否 = 当时 commanded/leader); 若已被 relabel 覆盖, 需从更上游 (采集原始 parquet) 重建。⚠️ **待查: 原始 action 是否还在**。
- **判据**: `eval_xvla_vision_ablation_offline.py` 视觉/本体比 ≳0.5 + 真机会找衣服。

### 4.1 实验 E1 — 确诊性 A/B: `use_proprio=False` (已就绪, 最快)
- **做法**: 复制 p0 训练 config, 仅置 `use_proprio: false` (关 state 输入), 其余 (数据 smooth800 / 60k / lr 1e-4 / ImageNet norm) 完全不变。
- **目的**: 强制模型只能用视觉。是**确诊**而非最终模型 — 验证"只要拿掉 proprio 捷径, 模型就会读视觉"。
- **判据**: 训完跑 `eval_xvla_vision_ablation_offline.py`, **视觉/本体影响比从 0.000 抬到 ≳0.5** = 根因坐实。再上真机看是否会"找衣服"。
- **风险**: 丢掉 proprio 平滑信号, 连续控制可能更抖 (可叠 publish-time EMA 缓解)。

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
| **E1 (确诊, 已就绪)** | **False** | — | ≡state | 无 | 视觉比 ≳0.5? (断架构链, 不依赖数据) |
| **E2** | True | **0.5** | ≡state | 无 | 视觉比 ≳0.5 + 真机成功率 |
| **E3** | True/dropout | 0.5 | 真实+位置多样 | 加 | 跨位置泛化 |

> E1 断架构链 (不管数据如何, 无 proprio 就没法复制) → 最快确诊; E0 断数据链 (官方做法) → 保 proprio 收益的治本。两者正交, 都成立则根因 = "action≡state × proprio 早融合" 双重确证。

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
2. 跑 **E1 (`use_proprio=False`, 已就绪)** 做确诊 → ablation 比值是否抬起 (断架构链, 不依赖数据, 最快验根因)。
3. **查原始 action 是否还在** (vePFS `A_new_smooth_800/base` 的 action 列是否被 relabel 覆盖) → 决定 E0 是否需从上游重建数据。
4. E1 确诊成立 → 走 **E0 (真实 action + static-skip)** 做治本的最终模型 (保 proprio 收益); 原始 action 已丢则退 **E2 (proprio-dropout)**。
5. 把 §5 门禁纳入 X-VLA 上机流程。
