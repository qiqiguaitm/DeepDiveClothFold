# Realtime-VLA FLASH — 深度研究方向 (deepdive_kai0 视角)

> **目的**: 在通读 [Realtime-VLA FLASH](https://dexmal.github.io/realtime-vla-flash/) (arXiv:2605.13778, Niu et al. 2026) + 源码 (`/data1/tim/workspace/realtime-vla-flash`, openpi/pi0 衍生) 之后, 结合 deepdive_kai0 已有的 realtime-VLA 工作 (`roadmap.md` / `strategy.md` / `v1_triton_log.md` / `layer_b_plan.md` / `ee_stability_layer1.md`) 与本仓库近期诊断结论 (开环/因果混淆、夹爪校准漂移、视觉消融), 提出**可深度研究的后续课题**。
>
> **建立**: 2026-06-07
> **代码**: `realtime-vla-flash/` (root) · 模型 `huggingface.co/Dexmal/RealtimeVLA-Flash`
> **关联**: [`roadmap.md`](roadmap.md) §3.1 阶段5 "#3 Flash 推测推理" (本文把它从"研究性、边际有限"升级为有明确抓手的课题) · [[reference_vision_ablation_openloop]] · `../../strategy/rlt_implementation_plan.md` · `../../strategy/dagger_implementation_plan.md`
> **执行日志**: [`flash_impl_log.md`](flash_impl_log.md) — 逐步实施记录 (改动/决策/验证), **铁律: 只新增文件, 不影响旧推理路径**。

---

## 执行进度 (live)

> 起步顺序见 §3。每个增量在 [`flash_impl_log.md`](flash_impl_log.md) 有详细记录。

| 子任务 | 状态 | 产出 / 备注 |
|---|---|---|
| **R1-a** draft head + 双夹爪接受逻辑 (CPU) | ✅ 2026-06-07 | `draft.py` + `spec_pi0_pytorch.py`(纯函数) + CPU 单测 16/16; 双夹爪(6,13)泛化完成 |
| **R1-b(seam)** draft 挂真模型 + 延迟实测 | ✅ 2026-06-08 | `spec_draft_attach_probe.py`; 真 pi05 prefix(968×2048)挂载成功, VLM-layer0 warm-start OK, **draft 2.5ms vs eager-full 263ms**; 确认投机在 32-D padded 空间 |
| **R1-b(full)** `SpeculativeSampler` 完整状态机 | ✅ 2026-06-08 | wrapper(不改 PI0Pytorch); draft→K-way verify→accept→双臂夹爪门控→stitch→fallback; 机制验证: 垃圾draft全拒+回退(radius3.0), oracle draft全接受50/50(radius0.012) |
| **R1-c** draft 蒸馏 (smoke) | ✅ 2026-06-08 | `spec_draft_distill.py`; 96 帧蒸馏 → holdout 平均接受 **27.9/50** (huber 5e-4); 坑: 训练发散需 grad-clip+cosine+best-ckpt |
| **R1-d** 扩数据重蒸 + 真实接受率基线 | ✅ 2026-06-09 | `spec_draft_r1d.py`(零噪声 teacher + 磁盘分片 cache + 真实 verify-from-draft eval) + `spec_draft_r1d_control.py`(证伪); 1600 帧蒸 → holdout **真实接受 50.0/50, 0 回退, radius 0.018** (逼近 oracle); 证伪: 同路径未训 draft 0/50+全回退 → 50/50 是真信号。⚠️ 离线天花板≠上机, 接受率在此 ckpt **饱和** → 坐实 R5 需"接受率×SNR"联合 |
| **R1-深(部署)** FLASH server + v2 一键启动 | ✅ 2026-06-09 | `kai0/scripts/serve_policy_flash.py` (server-only, 替换 `Policy._sample_actions` 一个 seam) + `start_scripts/kai/start_autonomy_from_ckpt_v2.sh` (拉 server → `start_autonomy.sh --mode websocket`); emit 标准 joint 14D, **现有 ROS2 客户端零改动**。冒烟: `policy.infer`→(50,14), accept 50/50 radius 0.017 draft16+verify102ms vs full≈510ms; fallback 强制抛错走 eager `full_denoise_from_observation` 不崩 (5090 compiled 路径会崩, 见 impl_log §7.3)。**未上真机/未提交** (部署红线: 真机 A/B 通过再 commit) |
| **R5** 接受率↔开环在线探针 | ✅ 2026-06-09 (部分证伪) | `spec_r5_probe.py`; pure200 real vs 全置黑: 模型 vision-SNR **7.36×** (确在用视觉) 但 FLASH 接受率 **50/50→50/50 Δ=0.00**、radius 也不动 (corr≈0) → **接受率/radius 不是开环探针 (证伪)**。机理: 接受率是**自洽**量 (draft 与模型在*同一输入*上一致), 非*信息*量, 对输入退化**结构性失明**。R5 论点锐化: `接受率×SNR` 须**真乘积**、SNR 外接独立。详见 `flash_impl_log.md` §8 |
| **R5-followup①** server 内"接受率×SNR"健康探针 | ✅ 2026-06-09 | `serve_policy_flash.py` 加 `_FlashHealthPolicy` (opt-in `--health-probe-every N`, 默认 0=关=裸 v2): 每 N 帧对同帧 raw-blacked 重前向算 vision-SNR (外接独立, §8.3 铁律), 与接受率联合日志+挂 `flash_health`。冒烟 (合成图) probe 触发、不崩、动作仍 (50,14)。详见 `flash_impl_log.md` §8.5 |
| **R5-followup②** 第二个开环 ckpt 直验 | ⏸ 阻塞 | 需第二个**开环 PyTorch ckpt**+其 draft; 现仅 pure200 一个 PyTorch ckpt, 已知开环 ckpt 均 JAX → 需 JAX→PyTorch 转换 (独立工程, 本轮不做)。§8.2 机理已预判两者都饱和; 有 ckpt 时复用 `spec_r5_probe.py` |
| **R1-真机①** 真机 spec 动得慢/欠完成诊断 | ✅ 2026-06-18 | `spec_draft_magnitude_probe.py`; `--no-spec` 正常但带 spec 慢 → 确诊 **draft v1 幅值欠驱动** (spec/full均值 **0.61**, cosine 0.52)。根因: 1 层+Huber 蒸馏退化为条件均值, 方向对幅值压到 61%; 接受率 50/50 查不出 (verify 锚 draft 自洽盲点)。详见 `flash_impl_log.md` §9 |
| **R1-真机②** draft v2 重训 (加层+MSE+幅值/速度损失) | ✅ 2026-06-18 | `spec_draft_r1d.py` 加 `--num-layers/--loss mse/--vel-weight/--mag-weight`; `draft.py` 加多层支持 (向后兼容, 默认 1 层)。v2 (2 层+MSE+vel+mag, 300ep): spec/full均值 **0.61→1.16**, cosine **0.52→0.71**, 逐维都回到位。产物 `/tmp/draft_r1d_pure200_v2.pt`。待真机 A/B; 未提交 |
| **R3** draft↔full 散度作不确定度信号 | ✅ 2026-06 离线证伪 | `spec_r3_probe.py`; in-distribution holdout 上 radius 近恒定 (R1-d 饱和), corr(radius,floor)≈0 → 离线测不出, 需 off-manifold/闭环帧 (docstring 为权威记录, 原 §9 已随 rebase 丢失) |
| **R6** FLASH Triton kernel 移到 5090 | ✅ 2026-06 量测 | `optimize/v1_triton/bench_v1_per_step_5090.py`; v1 5090: prefill 25ms (vision 4+VLM 21), per-step 0.9ms, full 34ms。FLASH prefill-skip (复用 VLM KV) 下 FLASH+v1 ≈ **3.83×** 但代价是视觉陈旧/开环 (docstring 为权威记录) |
| **R2/R3** 相位强验证 / 散度触发 | 📋 研究主线 | 见各课题 |

---

## 0. FLASH 是什么 (源码级精读)

FLASH = **首个面向 diffusion/flow-matching VLA 的"投机推理"(speculative inference) 框架**, 思想类比 LLM 的 speculative decoding: 用一个极轻 draft 先猜整条 action chunk, 再用原模型 Action Expert 验证, 只执行"被接受的前缀", 失败则回退全量推理。

| 组件 | 源码 | 机制 |
|---|---|---|
| **Draft 模型** | `src/openpi/models_pytorch/draft.py: DraftChunkHead` | **单层 Gemma decoder** over prefix embedding(VLM 已算好的 img+lang+state token)。`_action_queries` (Embedding[chunk_m]) + `_state_token` (Linear 32→h) + 1×`GemmaDecoderLayer` + `_action_head` (Linear→out_dim=7=6pose+1grip)。**一次前向出整条 chunk, 无 flow-matching 去噪迭代** → 这是提速来源。 |
| **接受准则** | `spec_pi0_pytorch._compute_radius_prefix_acceptance` | **半径前缀接受**: `dist=‖draft−verified‖`, 接受满足 `dist≤τ_radius` 的最长前缀。 |
| **相位感知回退** | `_detect_verify_gripper_switch_any_k` + `_truncate_accepted_prefix_on_gripper_switch` | **夹爪开合事件处截断**: chunk 内夹爪穿越开/合阈值 → 在该处截断接受前缀、强制全量验证。即**绝不在夹爪状态切换(精度关键事件)上做投机**。 |
| **拼接** | `_stitch_radius_prefix_output` | 接受的 draft 前缀 + 全量输出尾段拼成最终 chunk。 |
| **Triton serving** | `scripts/spec/triton/{pi0_spec_infer,triton_pi0_runtime}.py` | 定制 kernel, draft 路径 **7.8ms(2 视角)**, 任务级 **3.04× 加速**。 |
| **Draft 训练** | `enc_cache.py` → `spec_draft_train.py` | 冻结 backbone, 先 dump prefix-embedding cache(sliding-chunk-shift 采样), 再蒸馏 DraftChunkHead 回归 pi0 的 action chunk。成本极低。 |

**结果**: 全量 58.0ms → flash 7.8ms / 任务均 19.1ms; LIBERO 仅掉 0.3pt 成功率; 66.8% 重规划轮由 flash 路径完成; 唯一在 15 m/min 传送带动态任务上拿到非零成功率。

**为何与 deepdive_kai0 高度契合**: FLASH 基座是 **openpi/pi0**, 与 kai0(pi0/pi05)同源; kai0 已有 V1 Triton 路径(P50≈32ms/8×, `v1_triton_log.md`)与 5090 wgmma 调优(`layer_b_plan.md`)。FLASH 的 draft 几乎可直接挂到 kai0 现有栈上。

---

## 1. 课题总览 (按优先级)

| # | 课题 | 类型 | 价值 | 与本仓库结论的关联 |
|---|---|---|---|---|
| **R1** | FLASH 移植到 kai0 pi05 + 接 V1 Triton | 工程+研究 | <10ms 闭环 → 高 Hz 重规划 | roadmap 阶段5 落地 |
| **R2** | **夹爪/相位门控的视觉强验证** | 研究 ⭐ | 治"抓空开环" | [[reference_vision_ablation_openloop]] |
| **R3** | draft↔full 散度作在线不确定性/OOD 信号 | 研究 ⭐ | 主动学习 + RLT 触发 | dagger/rlt plan |
| **R4** | 125Hz 投机重规划 取代 RTC blend | 研究 | 去抖、去 chunk 陈旧 | `ee_stability_layer1.md` RTC |
| **R5** | 投机接受率作开环/因果混淆在线探针 | 研究 | 免费部署门禁 | 视觉消融 SNR |
| **R6** | 5090(Blackwell)kernel 移植 + roofline | 工程 | 复用已有 wgmma 工作 | `layer_b_plan.md` |
| **R7** | draft 作"轻量校正头"绕开重训 | 研究 | 救夹爪漂移数据 | 夹爪校准漂移结论 |
| **R8** | 把投机推理推广到 X-VLA(Florence2) | 研究 | 更慢的模型更受益 | `reference_xvla_inference` |
| **R9** | **MTP 式联合训练原生 draft 头**(取代 R1-c 事后蒸馏) | 研究 ⭐ | 接受率上限更高 → 真正高 Hz | LLM 投机解码演进 (DeepSeek MTP / EAGLE-3) |

---

## 2. 课题详述

### R1 — FLASH 移植到 kai0 pi05, 接入 V1 Triton 路径 (落地基座)
- **做什么**: 用 `enc_cache.py`+`spec_draft_train.py` 从 **pi05 flatten-fold ckpt** 蒸馏 `DraftChunkHead`(注意 kai0 是 14D joint / EE6D 输出, `out_dim` 需从 LIBERO 的 7 改成 kai0 的 per-step 维度); 把 `pi0_spec_infer.py` 的投机循环接到 kai0 `serve_policy_v1.py` 的 Triton runtime。
- **为何**: kai0 现 P50≈32ms(3Hz 推理 + RTC 插值)。FLASH 把单轮压到 ~10ms → 可把推理频率拉到几十 Hz, 这是 R2/R4/R5 的前置。
- **实验**: LIBERO 先复现 3.04×; 再在 kai0 vis val 上测 draft 接受率 + 半径分布; 真机 A/B(FLASH vs 现 V1)看任务完成 + EE 抖动。
- **风险/坑**: pi05 vs pi0 的 Action Expert 结构差异; kai0 sm_120 与 FLASH kernel(4090/4090D 调过)的 autotune 不兼容(复用 `layer_b_plan.md` 经验)。

### R2 — 夹爪/相位门控的"视觉强验证" ⭐ (把 FLASH 相位逻辑接到本仓库核心病)
- **观察**: FLASH 已经**在夹爪开合处禁止投机、强制全量验证**(`_truncate_accepted_prefix_on_gripper_switch`)。而本仓库视觉消融实测: **坏模型在夹爪通道几乎不看画面**(置黑 SNR 5.7× vs 好模型 26.6×), 真机"抓空也往下做"。两件事指向同一处: **抓取瞬间是精度+视觉关键事件**。
- **研究问题**: 把"夹爪切换处强制全量"升级为"**强制全量 + 强制重新编码视觉**"(smooth 段可用缓存/降采样视觉省时, 抓取段必须吃最新全分辨率帧)。**能否用这种"相位自适应视觉预算"在不重训主模型的前提下缓解开环抓取失败?**
- **实验**: 在 trace 里按相位统计 draft 接受/回退; 真机对比"全程等额视觉" vs "抓取段视觉强验证"的抓取成功率。
- **价值**: 把推理框架变成**闭环安全机制**, 而非单纯提速。

### R3 — draft↔full 散度作在线不确定性 / OOD / "需要人介入"信号 ⭐
- **机制**: 接受准则里的 `dist=‖draft−full‖`(半径)天然是**每步免费的置信度**: draft 与全量分歧大 = 模型不确定 / 当前帧 OOD。
- **研究问题**: (a) 半径散度是否与抓取失败 / 新场景 / 夹爪漂移帧相关? (b) 能否用"高散度段"**自动触发 DAgger 采集**(active learning, 只在模型没把握处要人接管)或 **RLT critical-phase 介入**(见 `rlt_implementation_plan.md`)?
- **实验**: 离线在 vis val + 真机 trace 上算 per-step 半径, 与失败标注做相关; 设阈值跑一次"散度触发 DAgger"闭环。
- **价值**: 把 FLASH 副产物变成 dagger/rlt 两条 plan 的**触发器**, 三者打通。

### R4 — 125Hz 投机重规划是否可**取代 RTC chunk-blend**
- **背景**: kai0 现在 3Hz 推理 + `StreamActionBuffer` min-jerk 重叠混合(`ee_stability_layer1.md`), 混合带来过 euler-wrap、attractor freeze、走3退1 等一系列 deploy 层补丁。
- **研究问题**: FLASH 若能 100+Hz 每个控制步重规划, **是否可直接丢弃 chunk 重叠混合**(每步都是新鲜 chunk 的第 0 帧)→ 从根上消除陈旧/混合伪影?
- **实验**: 仿真/真机对比 "3Hz+RTC blend" vs "≥30Hz FLASH no-blend" 的 EE 抖动 FFT 主频(沿用 [[feedback_real_machine_oscillation_data_tail]] 的判别)、折返比、任务速度。
- **风险**: 高 Hz 下 proprio 反馈时序与固件 t_motion 必须重新标定(roadmap 真机测试2)。

### R5 — 投机接受率作"开环/因果混淆"在线探针
- **联系**: 本仓库已有**离线**视觉消融门禁(置黑 MAE 比值)。draft 蒸馏自主模型, 若主模型开环(无视视觉), draft 也会无视视觉 → 其**接受率/半径分布可能成为开环的在线指纹**。
- **研究问题**: FLASH 部署统计(接受率、夹爪强制率 `gripper_force_rate`、半径分布)能否在线区分"闭环好模型"与"开环坏模型", 作为**比 MAE 更敏感的实时健康度**?
- **实验**: 对好/坏两个已知 ckpt 跑 FLASH, 比较接受统计与离线视觉消融 SNR 的一致性。

### R6 — FLASH Triton kernel 移植到 5090 (Blackwell) + roofline
- **做什么**: FLASH kernel 在 4090/4090D 调优(`exp/plot_pi0_roofline_4090d.py`)。kai0 已在 5090 上做过 V1 重 autotune + wgmma 升级(`layer_b_plan.md`)。把两边 kernel 互相借鉴, 出 5090 roofline。
- **价值**: 纯工程但能把 7.8ms 进一步压到 Blackwell 上限, 且沉淀可复用 kernel 资产。

### R7 — draft 作"轻量校正头", 绕开主模型重训
- **联系**: 本仓库刚定位 5-18~5-21 **夹爪校准漂移**导致主模型不可用(action↔物理映射错)。重训全模型成本高。
- **研究问题**: draft 头(~百 K 参数, 蒸馏即得)能否被训成一个**校准/校正头**——在不动 pi05 主体的前提下, 用少量正确标定数据把夹爪输出映射拉回真机 firmware? (与 RLT residual 思路互补, 但走推理框架而非 RL)。
- **风险**: draft 设计初衷是"快而粗", 当校正头需验证其精度足够; 可能更适合作 RLT 的 state encoder。

### R8 — 投机推理推广到 X-VLA / 更慢模型
- **联系**: kai0 另有 X-VLA(Florence2 + flow-matching, `reference_xvla_inference`), 比 pi05 更慢更重。FLASH 思路 model-agnostic(只要是 diffusion/flow VLA)。
- **研究问题**: 给 X-VLA 蒸一个 DraftChunkHead 是否能拿到比 pi05 更大的相对加速(基数更慢)? Florence2 prefix embedding 与 Gemma draft 的对接如何设计?

### R9 — MTP 式联合训练原生 draft 头 (取代 R1-c 事后蒸馏) ⭐
- **动机 (LLM 演进映射)**: LLM 投机解码已从"外挂/事后蒸 draft"(DistillSpec) 演进到"**把多 token 预测头作为一等训练目标从头训进模型**"(DeepSeek-V3 **MTP** / **EAGLE-3**)——模型出生即自带 draft, 草稿与目标**共享表示、同分布** → 接受率显著更高。kai0 现在的 R1-c 走的正是被 LLM 那边超越的"事后蒸馏"路线。
- **做什么**: 训 pi05 时**联合挂一个 DraftChunkHead**, 在主 flow-matching 损失之外加一项 draft 回归损失 (draft 预测整条 chunk → 监督到主模型自身的 teacher chunk 或 GT, step-weighted)。主模型与 draft **同时优化**, draft 吃的是**正在被训练**的 prefix 表示而非冻结快照。
- **研究问题**: (a) 联合训练能把 kai0 接受率拉到多高? —— 注意 R1-d 事后蒸馏在**离线同任务 holdout 上已达 50/50 天花板**, 故 R9 的真正增益不在离线分内, 而在 **off-manifold / 闭环漂移帧**: 那里 R1-c/R1-d 的冻结-prefix 事后对齐会掉接受率+起回退, R9 的同分布共享表示应更抗掉。衡量 R9 要用闭环或扰动帧的接受率, 不是离线分。(b) 加 draft 损失是否轻微正则/损伤主策略 MAE? (c) draft 损失权重、是否 detach prefix 梯度 (EAGLE 不 detach, Medusa 常 detach) 的取舍。
- **价值**: 接受率是投机加速的天花板; R1-c 蒸馏受限于"冻结 prefix"与"事后对齐", R9 直接抬高这个天花板 → 才可能支撑 R4 的真·高 Hz 重规划。

#### R9 vs R1-c 取舍 (关键决策点)
| 维度 | **R1-c 事后蒸馏** (已 smoke ✅) | **R9 联合训练** (本条) |
|---|---|---|
| 碰不碰主模型训练 | **不碰** (冻结 pi05, 只训 head) — 合"只动部署"红线 | **碰** (改 TrainConfig/训练 loop, 重训 pi05) |
| 成本 | 极低 (几分钟~小时, 单卡, 复用 ckpt) | 高 (一次完整 pi05 微调, 8×A100 量级) |
| 接受率上限 | 受限 (冻结 prefix + 事后对齐) | 更高 (同分布共享表示, LLM 已验证) |
| 风险 | 零 (主模型不变, 失败只是 draft 不好用) | 可能轻微动主策略 MAE; 需 A/B 守门 |
| 落地时机 | **现在** (R1-d 扩数据即可) | **后置** (等某次 pi05 重训窗口顺带挂上) |
| 互斥? | **否, 互补**: 先用 R1-c 在现有 ckpt 上拿基线/验证 R5; 下次重训 pi05 时再上 R9 抬上限 |

- **结论性建议**: **R1-c 不是被 R9 否定, 而是 R9 的前置探针**——R1-c 几乎零成本地回答"draft 头在 kai0 上能不能用 / 接受率↔开环 (R5) 假说成不成立", 这些结论再决定值不值得为 R9 花一次 pi05 重训。**先 R1-c/R1-d 拿证据, 再在下一个 pi05 训练窗口顺带做 R9。** 切忌为 R9 单独起一次重训。
- **风险/坑**: 联合训练需在带补丁 venv 的 PyTorch 训练栈 (非冻结推理); draft 损失权重过大会抢主模型容量; 同样要处理 kai0 双臂夹爪 (6,13) 的 step-weighted + 相位加权 (FLASH 训练里已埋 hook, 见 `flash_impl_log.md` §阶段2)。

---

## 3. 建议起步顺序

1. **R1**(基座, 2–3 周): 先在 LIBERO 复现, 再蒸 kai0 pi05 draft、接 V1 Triton、出 kai0 接受率基线。
2. 并行 **R5**(几乎零成本, 复用 R1 的统计)验证"接受率↔开环"假说 → 立刻反哺部署门禁。
3. **R2 + R3**(研究主线): 把 FLASH 从"提速"升级为"闭环安全 + 主动学习触发", 直接打本仓库核心病(开环抓取)。
4. R4 / R6 / R7 / R8 视 R1 结果再排。
5. **R9**(联合训练 draft 头)**后置**: 必须先用 R1-c/R1-d 的零成本蒸馏拿到接受率基线 + 验证 R5 假说, **再决定**是否在下一个 pi05 重训窗口顺带挂 MTP 头。R1-c 是 R9 的前置探针, 二者互补非互斥 (详见 §2 R9 取舍表)。

> **一句话立项理由**: FLASH 与 kai0 同源(pi0), 落地成本低; 而它的两个机制——**半径接受**(免费不确定性)与**夹爪相位强验证**(精度事件保护)——恰好正面命中本仓库这轮诊断出的**开环抓取/视觉脱钩**根问题。提速是表, 闭环安全与在线健康度才是对 deepdive_kai0 更大的价值。
