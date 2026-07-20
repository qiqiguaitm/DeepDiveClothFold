# chunk-001 拼接 DAgger 数据格式 (schema)

> 生成脚本: `train_scripts/kai/data/stitch_dagger_episodes.py`
> 数据位置: `Task_A/dagger/v{3,4}/<date>-v{3,4}/data/chunk-001/episode_*.parquet`
> 最近更新: 2026-07-18

chunk-001 是把同一 `rollout_id` 的 `inference`(策略自主) + `dagger`(人类接管纠错) episode 按时间序
**拼接成的完整 on-policy episode**——保留 `INF → (卡住) → DAG纠错 → INF恢复 → …` 的真实时间线,
并逐帧打**相位标签** `dagger_frame_class`。这正是 DAgger / 部署学习 (Sirius, IWR) 需要的输入:
既有策略自主分布, 又有人类纠错信号, 且每帧可区分相位以便下游按类加权。

对照 chunk-000: chunk-000 是**纯 dagger 接管片段** (整段 `intervention=1`, 短), chunk-001 才是
**完整拼接 episode** (`intervention` 部分帧, 1400~5900 帧)。

---

## 1. 列定义 (parquet schema)

| 列 | dtype | 说明 |
|---|---|---|
| `observation.state` | list\<float\>[14] | 双臂关节角+夹爪: `[L臂6关节, L夹爪, R臂6关节, R夹爪]` (dims 6,13=夹爪) |
| `action` | list\<float\>[14] | 动作目标 (本管线 `action==state` 约定, 逐帧对齐后的下一状态) |
| `timestamp` | float32 | `frame_index / FPS`, 拼接后重新连续编号 (FPS=30) |
| `frame_index` | int64 | episode 内帧序号 0..N-1 (拼接后重编) |
| `index` | int64 | 全局帧序号 (拼接后重编) |
| `episode_index` | int64 | 拼接 episode 序号 |
| `task_index` | int64 | 任务/prompt 索引 (AWBC 阶段由 `discretize_advantage` 覆写为 pos/neg) |
| `intervention` | int8 | **粗** 二值: `0`=策略自主, `1`=人类遥操。向后兼容保留 |
| `dagger_frame_class` | int8 | **细** 相位标签 (见 §2)。**非** intervention 冗余 |

---

## 2. `dagger_frame_class` 编码

基于 **Sirius** (Liu et al., RSS 2023 / IJRR 2025) 的部署数据 4 类 `{demo, robot, intv, preintv}`,
加上 2 类**双臂遥操录制静态伪影** (Sirius 用 SpaceMouse 无此问题, 我们双臂遥操会引入):

| code | name | 中文 | intervention | 含义 | 下游建议 |
|---:|---|---|:---:|---|---|
| 0 | `robot` | 自主-正常 | 0 | 策略自主, 正常执行 | 保留, w=1 (残余) |
| 1 | `intv_core` | 人控-纠错 | 1 | 人类遥操果断纠错核心 | 保留, **上采样** (Sirius `P*(intv)=1/2`; IWR 等量采样) |
| 2 | `preintv` | 自主-临失败 | 0 | 机器人**"停下点"之前 ℓ 帧**的失败先兆 (真运动, 非静止前奏 — 见 §3.1) | 保留+标记; 正向 BC **归零** (`P*(preintv)=0`); AWBC 侧**转负样本** |
| 3 | `hesitation` | 起手迟疑 | 1 | 接管后低速遥操起手 (静态伪影) | **物理裁掉** (脚本已删, 不落盘) |
| 4 | `stationary_tail` | 静止尾 | 1 | episode 末遥操后静止 (idle 伪影) | **物理裁掉** (脚本已删, 不落盘) |
| 5 | `demo` | 纯示范 | −1/无 | base 示范 episode 全部帧 | 保留码位, 本脚本不产出 |

**落盘后 chunk-001 只会出现 `{0, 1, 2}`** (3/4 已物理裁掉, 5 不由本脚本产出)。
→ 因此 `dagger_frame_class` **携带 intervention 之外的信息**: 同为 `intervention=0`,
`class 0` (正常自主) 与 `class 2` (临失败) 语义相反, 下游必须区分。

### 3.1 ⚠️ 我们的采集流程 ≠ Sirius, preintv 必须重新定义

**Sirius**: SpaceMouse **运动中直接介入**, 机器人不停 → 接管前 ℓ 帧就是"策略搞砸的真运动"。
**我们** (`dagger_recorder_node.py`, two-step freedrive 门控): **打断 → 机器人停住 → 人准备好 → 接管**。
→ 实测 913 条带接管的 inference 段: **接管前末 15 帧 90% 是静止** (中位 100%; 静止前奏中位 0.5s,
p90 ~0.9s)。若照 Sirius 直接取"接管前 ℓ 帧"当 preintv, 拿到的是**静止前奏伪影, 不是失败动态**。

**正确切法** (`classify_segment` inf 分支):
1. 找**停下点** = 最后一个 `arm_vel > STATIONARY_THR` 的帧;
2. 停下点【之后】= 静止前奏 → 标 `stationary_tail(4)` → **物理裁掉**;
3. 停下点【之前】ℓ 帧 = **真失败先兆** → 标 `preintv(2)` (实测 arm 速度 0.0031 ≈ 正常运动);
4. 更早 → `robot(0)`。

### 3.2 ℓ 是时间尺度, 随 FPS 变 (别照抄 15)

Sirius ℓ=15 @ **20 Hz** = **0.75s** 人反应时间。本数据 **30 FPS**, 同样 0.75s 需
`round(0.75×30)=22` 帧。脚本用 `REACTION_TIME_S=0.75` → `PREINTV_MARGIN=round(0.75*FPS)=22`,
**不硬编码 15** (照抄 15 只有 0.5s, 偏短 1.5×)。

---

## 3. 打标 vs 裁剪 (设计原则)

**打标 (labeling)** 与 **裁剪 (trimming)** 解耦, 由两组常量控制:

```python
TRIM_CLASSES = {CLASS_HESITATION, CLASS_STATIONARY_TAIL}   # = {3, 4}, 物理删帧
CLASS_TRAIN_WEIGHT = {0:1.0, 1:2.0, 2:0.0, 3:None, 4:None, 5:1.0}  # 下游 loss 参考权重
```

- **静态遥操伪影 (3,4) → 物理裁掉**: 它们不是策略/纠错信号, 是录制起手/收尾的静止帧, 留着有害。
  (与之前扫描一致: chunk-001 首尾静止已极短, 见 §5。)
- **preintv (2) → 保留并标记, 不裁**: 它是"机器人快失败时的自主帧", 是宝贵的失败先兆信号。
  正向模仿要归零 (别学失败), 但作 AWBC **负样本**价值高。删掉 = 丢信息 + 让 class 退化成死列。
- **intv_core (1) → 保留 + 上采样**: 纠错是最该学的 bottleneck 动作 (Sirius/IWR 一致)。

阈值 (与历史 `find_keep_indices` 同源):
`HESITATION_THR=5e-3` rad/frame (arm), `GRIP_HESITATION_THR=0.01`; `STATIONARY_THR=3e-3`,
`GRIP_STATIONARY_THR=0.02`。夹爪在动 (抓/放) 时不算迟疑/静止 → 保留为 core。

---

## 4. 下游消费合约 (务必保持一致)

| 消费者 | 用法 | 与本 schema 的关系 |
|---|---|---|
| `kai0/stage_advantage/annotation/discretize_advantage.py` `_get_exclusion_mask` | `dagger_frame_class ∈ {2,3,4}` → 排除出 **positive** advantage 标注; 无该列则回退 velocity 检测 | **天然一致**: preintv(2) 不进 positive = Sirius `P*(preintv)=0`。落盘只剩 2 (3/4 已裁), 效果等价 |
| `build_v4_awbc_merged.py` / `build_vis_awbc_merged.py` | 建 AWBC 训练集时**丢弃 `intervention` 列**做统一 schema; frame 级标签由后续 `infer_dagger→eval→discretize_advantage` 重加 | 若也丢 `dagger_frame_class`, 则回退 velocity 检测; 要保留细类需显式带上该列 |
| dataloader 加权 (若启用 Sirius 式加权 BC) | 读 `dagger_frame_class` → 按 `CLASS_TRAIN_WEIGHT` 施 per-sample 权重 | 直接用本列; 未启用则该列仅供分析 |

> ⚠️ 改动 code 语义 (尤其 {2,3,4} 的排除约定) 前, 必须同步 `discretize_advantage._EXCLUDE_CLASSES`。

---

## 5. 已验证的数据特征 (2026-07 扫描, v4 dagger 387 chunk-001 eps)

- 首/尾静止段: 均值 ~0.5–0.7 帧 (0.02s), 无 >2s 者 → 在线录制器前裁+尾裁已生效。
- 内部最长静止段: 中位 5 帧, 全库最大 1.73s, 仅 2 ep >1.5s → 无大段停滞。
- 模型自主速度 ≈ 遥操 0.6× (手臂) / 0.3× (夹爪) — 见部署速度分析。

---

## 6. 文献

- **Sirius** — Liu, Nasiriany, Zhang, Bao, Zhu. *Robot Learning on the Job: Human-in-the-Loop
  Autonomy and Learning During Deployment.* RSS 2023 / IJRR 2025. (arXiv:2211.08416)
  4 类 `{demo,robot,intv,preintv}`, 加权 BC `w(s,a,c)=P*(c)/P(c)`, `P*(intv)=1/2`, `P*(preintv)=0`, ℓ=15。
- **IWR** — Mandlekar et al. *Human-in-the-Loop Imitation Learning using Remote Teleoperation.*
  arXiv:2012.06733. intervention 与非 intervention 等量采样 → 等效上采样纠错帧。
