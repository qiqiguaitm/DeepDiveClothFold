# CRAVE-RPO 最小验证实验（一页 plan）

> **目的**：验证「CRAVE dense 进度 + SRPO 的 outcome 信号 + 自参考 rollout」三者拼成的 **CRAVE-RPO**，能否在折叠任务上**超过纯离线 CRAVE-AWBC（AB_plan A 臂）**——即补上 CRAVE 公认的两个洞：**无结果信号**、**挖矿域脆弱**（[positioning §0/B2](CRAVE_positioning_and_roadmap.md)）。
> **定位**：这是 [AB_plan](awbc_milestone_value_AB_plan.md) 的**上层闭环**。AB_plan 答「CRAVE 当离线 value 源行不行」；本实验答「再加 RL 结果信号+自参考，能不能从经验改进/逼近超示教」。对标 SRPO（[2511.15605](https://arxiv.org/abs/2511.15605)）真机 offline-AWR 路线。
> **一句话**：A 臂 = CRAVE 离线打标 BC；**CRAVE-RPO = A 臂 + 二值成功 outcome + 用策略自己成功 rollout 重挖 milestone**。

---

## 1. CRAVE-RPO 相对 A 臂只加两味料（单变量隔离）

| 成分 | A 臂（已有） | **CRAVE-RPO（本实验）** |
|---|---|---|
| 进度信号 | CRAVE per-frame value（dagger demo 挖） | CRAVE per-frame value（**自参考挖**，见 §2） |
| 参考集 | 固定 dagger 挖矿集 | **策略自己的成功 rollout**（治挖矿域 gotcha / D1） |
| 结果信号 | 无 | **二值成功 R∈{0,1}**（SRPO outcome，补 B2 洞） |
| 数据来源 | 1124ep 遥操 demo | demo + **当前策略 rollout（含失败）** |
| 优化 | offline AWBC（=AWR） | **同 offline AWBC（=AWR）**，不变 |

> 下游 Stage 3/4（三档离散化 + `pi05_flatten_fold_awbc` warm-start）与 A/B/C **逐字段一致**，唯一变量 = advantage 的来源是否含 outcome+自参考。

---

## 2. 四个关键决策（用户问的四点）

**① 成功集怎么定** — primary = **真 outcome**，不用 CRAVE 自检（B2 已证终点可达性抓不到细微失败）：
- **数据源**：先用**已有的 autonomy 真机 rollout**（3 轮叠衣，含 2 次真退步/重试，CRAVE §3.4 已挖过）+ sim01 部署当前 SFT/A 策略采的新 rollout。**零新采集即可起步**。
- **打成功标**：① sim01 若有任务成功检测器 → 直接用；② 否则**人标**每条 rollout 二值成功（叠好=1/否=0），~100 条几十分钟，这是注入 CRAVE 缺的真 outcome 的**最小成本**；③ CRAVE「到终止 milestone 簇」仅作**自动预筛**，不当 ground-truth。

**② GRPO 还是先走 AWR** — **先 offline AWR，不上 GRPO**：
- pi05 是 **flow-matching** 策略，GRPO 需在去噪链上算高斯 logprob（RISE 式），重且有坑；SRPO 真机本身也是走 AWR。
- **AWR = 你现成的 AWBC**（advantage-conditioned BC，三档 prompt）→ **零新 RL 机器**，复用 Stage 3/4。
- GRPO 留作 **Phase 2**：只有当 AWR 证明「outcome 信号确实加分」后，才值得为 online GRPO 解决 flow-policy logprob。

**③ CRAVE value 怎么接 advantage**（薄 adapter，扩 `milestone_value_to_advantage.py`）：
```
# 每条 rollout i：
v[t]   = CRAVE per-frame value（自参考 milestone 挖出，[0,1]）
prog[t]= clip(v[t+50] − v[t], −1, 1)            # 进度差，Δ=50，与 pi0-AE/A 臂同窗
adv[t] = (1−α)·R_i + α·prog[t]   ，  α=0.8       # SRPO blend：成功轨迹整体抬，失败按进度塑形
# 失败轨迹 R_i=0 → adv 纯由进度差驱动（退步段自然 neg）；成功 R_i=1 → 全程 outcome 抬升
```
→ 写回 `absolute_advantage` 列（与 pi0-AE **同列名，下游零改动**）→ **三档离散化**（`adv<−ε neg / >ε pos / else normal`，ε=0.02，CRAVE 天然形态，见 [AB_plan §5b](awbc_milestone_value_AB_plan.md)）。**关键**：失败 rollout 现在能贡献真 **neg 档**（治 A 臂 neg 仅 5.1% 的洞）。

**④ 在哪台机器跑**：
- **特征提取**（DINOv2 三路，GPU）：gf0 无 GPU → **Volc Robot-GPU 开发机队列**（离线 V-JEPA/DINOv2 设 `TORCH_HOME=.../openpi_cache/torch_hub`）。
- **CRAVE 自参考挖掘 + adapter**（CPU）：gsy-cpu-dev / gf0 本地，几分钟。
- **AWBC 收敛训练**（8×A100）：**cnsh robot-task 闲时 Preemptible**（同 A-3lvl job `awbc_mv_A_3lvl_cnsh_8gpu.yaml`，self-heal resume）；本地 2 卡仅 sanity。
- **Tier3 rollout 评估**：sim01。

---

## 3. 对照 baseline（决定性 = CRAVE-RPO vs A-3lvl vs C）

| 臂 | = | 隔离的问题 | 状态 |
|---|---|---|---|
| **SFT** | warm-start `task_a_new_smooth_800_step49999` | 地板（MAE@1 0.0089） | 已有 |
| **C** | pi0-AE 人标 GT AWBC | 完整监督 RECAP | 已跑（Tier3 待补） |
| **A-3lvl** | CRAVE 离线直打 AWBC（无 outcome/无自参考） | CRAVE-RPO 去掉两味料的消融 | 已落地（[AB_plan §10](awbc_milestone_value_AB_plan.md)，集群训练中） |
| **CRAVE-RPO** | A-3lvl + 二值成功 + 自参考 rollout（offline AWR） | **加 outcome+自参考是否真改进** | 本实验 |

**判据矩阵**（Tier3 sim01 成功率，≥20 trials/臂）：

| 结果 | 结论 |
|---|---|
| **CRAVE-RPO > A-3lvl** | outcome+自参考确有增益 → CRAVE 该闭环化，验证 C2 路径 ✅✅ |
| CRAVE-RPO ≈ A-3lvl | 离线 demo 已够，rollout/outcome 在此任务不加分 → 省掉闭环 |
| **CRAVE-RPO ≥ C** | 零人标 + 自参考能逼近/超监督 → 主线 |
| CRAVE-RPO < A-3lvl | rollout 标签噪声/分布外害事 → 检查成功标质量 + α |

> 决定性对照 = **CRAVE-RPO vs A-3lvl**（单变量隔离「加 SRPO 两味料」），其余对 C/SFT 给绝对水平。

---

## 4. Phase 拆分（最小路径，~5-7 天）

| Phase | 任务 | 工作量 | 机器 |
|---|---|---|---|
| **P0** | 收 rollout（autonomy 现成 + sim01 采 ~100 条）+ 二值成功标 | 0.5-1d | sim01 + 人标 |
| **P1** | rollout 三路特征提取 | 0.5d | Volc GPU 队列 |
| **P2** | **自参考挖 milestone**（在成功 rollout 上，非固定 dagger）+ 全 rollout 打 `mv_value` | 0.5d | CPU |
| **P3** | `milestone_value_to_advantage.py` 加 `(1−α)R+α·prog` blend + 三档离散 | 0.5d | CPU |
| **P4** | AWBC 训练 CRAVE-RPO（demo+rollout 混合集，warm-start，~15-20k） | 1-2d | cnsh 8×A100 |
| **EVAL** | Tier1 MAE sanity + **Tier3 sim01（CRAVE-RPO / A-3lvl / C 三臂同口径）** | 1-2d | local+sim01 |

**先跑最小版**：只用 autonomy 现成 rollout + ~100 条 sim01 rollout 起步，跑通 CRAVE-RPO vs A-3lvl 一对，再决定是否扩量 / 上 GRPO。

---

## 5. 风险与兜底

| 风险 | 兜底 |
|---|---|
| rollout 太少/成功率≈0 → 自参考无成功可挖 | **CRAVE demo 挖矿冷启**（A 臂 milestone）兜底，待策略有成功后切自参考——正是 CRAVE↔SRPO 互补点 |
| 人标成功成本 | 先 100 条；CRAVE 终点簇自动预筛降人工量 |
| flow-policy 无法 GRPO | 本实验**只用 AWR**，规避；GRPO 留 Phase 2 |
| rollout 分布外污染 demo 集 | demo:rollout 混合比例做消融；advantage 三档已对尺度鲁棒 |
| α 敏感 | 扫 α∈{0.5,0.8,1.0}，对照 SRPO 最优 0.8 |

---

## 6. 链接
- 方法 → [METHOD](cross_episode_recurrence_value_METHOD.md) · 定位 → [positioning §0/C2](CRAVE_positioning_and_roadmap.md) · 下层 → [AB_plan](awbc_milestone_value_AB_plan.md)
- SRPO 原文 → [arXiv 2511.15605](https://arxiv.org/abs/2511.15605)（self-referential + V-JEPA 进度 + GRPO/AWR）
