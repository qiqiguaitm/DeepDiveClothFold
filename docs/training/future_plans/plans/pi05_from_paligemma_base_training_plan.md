# 从 PaliGemma base 自训 pi0.5(不用 PI 机器人预训练)— 训练 Plan

> **建立**: 2026-06-10
> **目标(用户定档)**: 用**自己的数据**(kai0 base+dagger + vis base+dagger)、从 **PaliGemma VLM base** 起(**不 warm-start PI 的 pi05_base 机器人预训练 ckpt**)、单节点 **8×A100、几天**,训出**可部署的叠衣策略**(kai0 + vis 两本体,**非**跨本体泛化研究)。
> **依据**: 深度调研(101 agents / 19 源 / 25 claim 3-票核验,primary = pi0 `2410.24164` / pi0.5 `2504.16054` / Knowledge-Insulation `2505.23705` / openpi repo)+ kai0 代码事实(本仓 openpi fork)。
> ⚠️ **方法学铁律**(本项目): **真机为终判,offline MAE 系统性反指**。VLA 训练报告先看 **val MAE**(不是 train loss)。

---

## 0. ⭐ 裁决:可收敛吗?

**可以收敛到可部署单任务策略 —— 但有两个硬条件,且真实算力 ≫ "几天"。**

| 维度 | 结论 |
|---|---|
| **一般性命题**(从 PaliGemma base、跳过机器人预训练能否收敛) | 🟢 **YES(高置信)**。PI 自己的消融:from-scratch(PaliGemma init + 随机 action expert)在 LIBERO 拿 96.6/97.2/94.6/84.8/92.7 ≈ from-generalist 98.0/...;pi0 论文:from-scratch 在所有测试任务都 work,预训练"有时快 2×"但非必需;open-pi-zero 从 PaliGemma 训到 SIMPLER 87.9%/97.8%,单机 8 卡 1-2 天 |
| **你的精确场景**(真机/双臂/可变形布/长程/百万帧/2本体) | 🟡 **marginal-to-YES**。最强 from-scratch 证据全来自**仿真 + 单臂/单本体 + 短程 + ~50 demos**;你是**真机双臂可变形长程**。**大 in-domain 数据(7.7M帧)是有利因素**,但没有 primary 源直接证过"从 base 训真机双臂叠衣" → 谨慎乐观 |
| **必须用对配方** | 🔴 **naive flow-matching from base 是错的**(见 §2):慢 7.5× + 毁 VLM。需用 **Knowledge Insulation (KI)** 或接受 naive 的慢/风险 |
| **算力现实** | 🔴 **从 base 约 240k step(planning anchor)**,8×A100 实测 ≈ **1-2 周**,**远超"几天"**(见 §5)。这是最大冲突点 |

> **一句话**: 技术上能收敛、有 primary 证据支撑;但 (a) 必须 KI-style 不能 naive,(b) 8×A100 跑满配方要 ~1-2 周不是几天,(c) 真机双臂叠衣这个具体域没人直接证过。**如果"可部署"是硬目标 + 时间紧,warm-start 仍是最稳的(但你已排除)。**

---

## 1. pi0 / pi0.5 训练逻辑 + 阶段顺序(调研核验)

**pi0**(`2410.24164`,3-0):两阶段
1. **机器人预训练** >10k 小时(7 robot configs / 68 tasks + 全 OXE)→ 建广域知识 base
2. **后训练** 在窄高质量 per-task 数据(~5–100+ 小时)上 finetune

**pi0.5**(`2504.16054`,3-0):两阶段,stage-1 目标不同
1. **stage-1 = 280k step**,广域机器人+非机器人数据,**离散 FAST-token 的 next-token-prediction**(预测文本/物体位置/动作 token)。⚠️ **97.6% 数据来自其它机器人/跨本体/web**,只 ~400h(2.4%)是目标任务
2. **stage-2 = 80k step**,特化到目标域 + **新增随机初始化的 flow-matching action expert**(NTP + flow-matching 联合);action expert "post-training 开始时随机初始化"

**关键**: 两者都**从 web 预训练 VLM(PaliGemma)初始化**,机器人预训练是"广域知识"层,不是收敛前提。**web/跨本体数据驱动的是泛化(尤其 OOD)**,对单一 in-distribution 目标任务**不是收敛必需**(`2504.16054`:no-WD 在 in-distribution mock-home 差异"不显著")。

---

## 2. 三个必须避开的失败模式(决定配方)

| # | 失败模式 | 证据 | 对策 |
|---|---|---|---|
| **F1** | **naive flow-matching from base**(pi0 配方:随机 action expert 直接 + VLM 联合训)| 随机 action expert 梯度**污染预训练 VLM backbone** → **训练慢 7.5×** + **语言跟随退化**(`2505.23705`,3-0;Fig6b) | 用 KI(F2 对策),或接受慢 + 单 prompt 场景下语言退化影响小(见 §4 路径 B1) |
| **F2** | **冻结 VLM**(以为冻住就不污染) | 冻结的 web-VLM 表示**对机器人不够** → **0%**(叠衫 ~10% vs pi0.5+KI ~55%,KI 论文 Fig4a/8) | **绝不冻 VLM**,backbone 必须 finetune |
| **F3** | **VLM grounding 灾难性遗忘** | action expert 梯度降 backbone 表示 → 模型忽略指令(去抓垃圾而非勺子)| **web/VL co-train**(泛化提升最大) **或** KI stop-gradient(无 web 数据时尤其有用) |

→ **正解 = Knowledge Insulation (KI)**(PI 当前 SOTA 配方,`2505.23705`):
1. **stop-gradient** 在 action-expert↔VLM backbone 的 attention KV 路径 → 随机 action expert 权重**永不改 VLM**
2. backbone 上加 **FAST 离散动作的 NTP loss**(仅训练期当表示学习信号)+ 轻量 VL/web 数据
3. action expert 用 flow-matching 训连续动作(**α=1**,因为现在作用在独立权重上)
→ 训得和 pi0-FAST 一样快、稳定收敛。

---

## 3. kai0 代码现状(本仓事实)

| 项 | 现状 |
|---|---|
| **你要的 init 路径** | `weight_loaders.PaliGemmaWeightLoader`(`kai0/src/openpi/training/weight_loaders.py:64`)**已实现**:加载官方 PaliGemma 权重 + action expert 保持随机。**但全仓 25+ config 没一个用它**(全用 `CheckpointWeightLoader` 从 `pi05_base` warm-start)→ **你需要手接** |
| 三个 loader | `NoOpWeightLoader`(全随机,不要)/ `CheckpointWeightLoader`(pi05_base warm-start,你排除)/ **`PaliGemmaWeightLoader`(你的路径)** |
| 模型 | `pi05=True`:PaliGemma **gemma_2b** + action expert **gemma_300m** + SigLIP So400m/14 ≈ **3.3B**;discrete_state_input + adaRMSNorm 注入 flow 时间步 + max_token 200;action_dim 32 / **horizon 50**(~1.67s@30Hz) |
| 现有(warm-start)超参 | AdamW · cosine warmup1k / peak **1.5e-5** / decay 50k→1.5e-6 · EMA **0.9999** · batch **128** · fsdp **8** · **50k step** · resize224 · 无 quantile norm |
| **双本体机制(可复用)** | `pi05_kaivis_perdsnorm_cond`:**domain conditioning(2 域 kai/vis)+ domain_weights (1.0, 3.97)** 做帧级 1:1 平衡(kai 5.78M / vis 1.46M 帧)→ 你的 co-train 直接套 |
| **KI 训练路径** | 🔴 **kai0/openpi 都没有**。stop-grad + FAST-NTP co-train 头**需自己实现**(openpi issue #365/#814 也是用户在问)→ **非平凡工程** |
| 数据 | kai0_base 3055ep/3.36M + kai0_dagger 3457ep/2.42M + vis(vis_v2_full)1406ep/1.93M + vis_dagger(待补)≈ **8.5k ep / 7.7M+ 帧** |

---

## 4. 推荐配方(两条路径,按工程量/风险)

> 都满足:`PaliGemmaWeightLoader` init、**不冻 VLM**、双本体 domain-cond co-train、per-embodiment norm。

### 路径 B1 — naive-from-base(工程最小,慢但可行)⭐ 给你这个目标的务实首选
**理由**: 你是**单任务、单一固定 prompt("Flatten and fold the cloth")、纯部署**——F3(语言遗忘)对你**影响小**(不需要开放世界语言跟随)。所以可以接受 naive flow-matching 的"语言退化",只需吞下"慢 7.5×"= 多跑 step。**几乎零额外工程**(只换 weight_loader + 加 step + 调 LR)。
- **init**: `weight_loader = PaliGemmaWeightLoader()`(action expert 随机)
- **不冻 VLM**;optimizer AdamW
- **LR**: 比 finetune 高一档 + 更长 warmup —— **peak 3e-5~5e-5 / warmup 2k-5k / cosine decay → 1/10**(随机 action expert 需更激进起步;open-pi-zero 用 5e-5/global-batch-1024 作参考)
- **steps**: **planning anchor ~240k**(openpi DROID:from-PaliGemma 240k vs from-robot-ckpt 100k);可先订 **150k 看曲线再续**
- EMA 0.9999 · batch 128(显存够可 192-256)· fsdp 8 · horizon 50
- **双本体**: domain conditioning(套 `kaivis_perdsnorm_cond`),domain_weights 按帧级平衡 + **可上调部署目标本体权重**
- **per-embodiment norm**: kai0 / vis **各自算 norm_stats**(关节 scale/DoF 不同,绝不共用)
- **数据配比**: 4 个数据集(kai base/dagger + vis base/dagger)co-train,大致按帧数比 + 部署目标本体适度上权

### 路径 B2 — KI-from-base(正解,工程重)
若要**又快又保 VLM grounding**(或将来要做语言条件/多任务),实现 KI:
- stop-gradient(action-expert→backbone attention KV)+ **FAST 离散动作 NTP 头**(训练期)+ flow-matching action expert(α=1)+ 轻量 VL co-train
- 同样 PaliGemmaWeightLoader init、不冻 VLM、双本体 co-train
- 收敛快(≈ pi0-FAST)、稳;但 **kai0/openpi 无现成实现 → 需 ~1-2 周工程**(改 model forward 加 stop-grad + 加 FAST tokenizer/NTP loss + co-train 数据管线)

### 收敛判据(两路径通用)
- **val action MAE**(held-out,**先看这个**,不是 train loss)逐 ckpt 曲线 → 单调降 + plateau
- 周期性 **sim/真机 rollout 成功率**(终判)
- **语言跟随 spot-check**(B1 尤其:换个无关 prompt 看动作是否变,检测 grounding 是否还在/已塌)

---

## 5. ⚠️ 算力现实(必须正视的冲突)

| 配置 | step | 8×H100 wall-clock | **8×A100 估算** |
|---|---|---|---|
| from **robot ckpt**(warm-start,你排除) | ~100k | ~2 天 | ~3-4 天 |
| from **PaliGemma base**(你的路径) | **~240k** | ~5 天 | **~1-2 周** |

→ **"几天"装不下 from-base 满配方(~240k step)。** 选项:
1. **接受 ~1-2 周**(订满 240k);
2. **砍 step 到 ~120-150k**(欠收敛风险,先出 ckpt 看 val MAE 曲线决定续不续);
3. **只训部署目标单本体**(数据少一半,可能更快收敛,但丢另一本体——与"2 robots 部署"目标冲突);
4. **(诚实)若可部署是硬约束 + 时间紧 → warm-start 仍最稳**(你已排除,但保留在桌面)。

---

## 6. 诚实权衡 + 最终建议

| | from-base(你的选择) | warm-start pi05_base(对照) |
|---|---|---|
| 收敛到可部署 | 🟡 能,但慢(240k)+ 真机双臂域未被直接证过 | 🟢 最稳(你既有 work 锚点 smooth_800 MAE@1=0.0089) |
| 算力 | 🔴 ~1-2 周 8×A100 | 🟢 ~10-20k step 续训即可 |
| 工程 | B1 小 / B2 重(KI 需自实现) | 零(现成 config) |
| 价值 | 自主可控、不依赖 PI 权重;科研意义 | 快、稳、省 |

**建议(按你"可部署 + 几天 + 从 base"的约束排序)**:
1. **先跑 B1(naive-from-base)的小规模 sanity**:`PaliGemmaWeightLoader` + 150k step + LR 3e-5/warmup3k,双本体 co-train。**~3-5 天出第一个 ckpt 扫 val MAE 曲线** → 判断是否在收敛轨道上。单任务单 prompt 下 naive 的语言退化不致命,这是性价比最高的验证。
2. **若 val MAE 明显下降**(进入收敛轨道)→ 续到 240k + 真机 rollout 终判。
3. **若想要快/保 grounding 且愿投工程** → 上 B2(KI)。
4. **全程保留 warm-start 作对照基线**(同数据、同 eval),量化"自训 vs PI 预训练"的真实差距 —— 这也回答了你科研层面的问题。

---

## 7. 落地步骤

1. **建 config** `pi05_kaivis_from_paligemma`(克隆 `pi05_kaivis_perdsnorm_cond` → 改 `weight_loader=PaliGemmaWeightLoader()`、LR/warmup/steps 按 §4 B1)。注册到 `config.py`。
2. **数据**: 合并 kai base/dagger + vis base/dagger(复用 `kai_vis_merged` build),**per-embodiment norm_stats 各自算**(`compute_norm_states_fast.py`)。
3. **PaliGemma 权重就位**: `PaliGemmaWeightLoader` 默认拉 `gs://vertex-model-garden-paligemma/.../pt_224.npz` → 离线环境需先下到本地 cache,改 loader 路径。
4. **提交 8 卡训练**(cnsh/cnbj 8×A100/H20,fsdp8,batch128,150k step,每 5-10k save)。
5. **监控**: val MAE 曲线(`vis_v2_merged_val` 同协议)+ loss;~50k 起看是否在降。
6. **判定**: 150k 出 ckpt → val MAE vs warm-start 基线 → 决定续 240k / 真机 rollout / 转 B2 / 回 warm-start。

---

## 7.5 ⭐ 实施结果(B1 naive-from-base,config `pi05_kaivis_from_paligemma`)

> **job**: 闲时多次重提 → **转独占** `t-20260619152024-pthmf`(cnbj Robot-North-H20 8×H20)。停在 **148000/150000(~99%)**。⚠️ offline MAE 系统性反指,**真机为终判**(未做)。

### 7.5.1 初始化(冷启动 from PaliGemma base)
**weight_loader = `PaliGemmaLocalWeightLoader(npz_path=…/paligemma_weights/pt_224.npz)`** — npz 里有的载入,没有的走模型 Flax 随机 init:

| 模块 | 初始化 |
|---|---|
| **SigLIP-So400m 视觉编码器**(`img/*`)| ✅ PaliGemma 预训练 |
| **Gemma-2B LLM / VLM**(`llm/*`)| ✅ PaliGemma 预训练 |
| **Action Expert(Gemma-300m)** | 🎲 随机 init |
| **动作 in/out 投影、state 投影、time MLP(adaRMS)** | 🎲 随机 init |
| **域条件 hub `action_head_cond_hub`**(num_domains=2,kai=0/vis=1)| 🎲 随机 init |

→ 与 warm-start(`CheckpointWeightLoader(pi05_base)`)本质不同:**只继承 PaliGemma VLM,action expert 从零学** → 早期 MAE 高(6k @1=0.47)、需 150k 步。

### 7.5.2 模型 + 训练参数

| 项 | 值 | 项 | 值 |
|---|---|---|---|
| model | `Pi0Config(pi05=True, action_head_cond_num_domains=2)` | dtype | bf16 |
| VLM / action expert | gemma_2b / gemma_300m | action_dim / horizon | 32 / 50 |
| max_token_len | 200 | num_train_steps | **150,000** |
| batch / 并行 | **128** / fsdp_devices=**8** | LR | CosineDecay warmup **3,000** / peak **3e-5** / decay 150k / end **3e-6** |
| EMA | **0.9999** | optimizer | AdamW(默认)|
| save / keep | 2,000 / 10,000 | num_workers | 16 |
| data | `kai_vis_merged`(kai+vis 双本体合并) | default_prompt | `"Flatten and fold the cloth."` |
| use_delta_joint_actions | False(absolute joint) | domain_weights | **(1.0, 4.2925)**(kai:vis 帧级配平≈1:1)|
| inline-eval | `vis_v2_merged_val` 200 帧,每 4 save | dataset_id | 1 |

### 7.5.3 inline-eval MAE 历史(val=`vis_v2_merged_val`,单调深度收敛、无回弹)

| step | MAE@1 | @10 | @25 | @50 |
|---|---|---|---|---|
| 6000 | 0.4685 | 0.4682 | 0.4691 | 0.4712 |
| 20000 | 0.0462 | 0.0417 | 0.0467 | 0.0602 |
| 40000 | 0.0128 | 0.0197 | 0.0284 | 0.0387 |
| 64000 | 0.0079 | 0.0148 | 0.0212 | 0.0286 |
| 80000 | 0.0072 | 0.0133 | 0.0187 | 0.0249 |
| 96000 | 0.0070 | 0.0123 | 0.0168 | 0.0223 |
| 112000 | 0.0068 | 0.0115 | 0.0155 | 0.0203 |
| 128000 | 0.0067 | 0.0111 | 0.0146 | 0.0190 |
| 136000 | 0.0066 | 0.0109 | 0.0142 | 0.0185 |
| **144000(末次 eval)** | **0.0066** | **0.0107** | **0.0139** | **0.0181** |

- **best ckpt**(落盘最新,> 末次 eval):`…/kai0/checkpoints/pi05_kaivis_from_paligemma/from_paligemma_cnbj/`**`148000`**`/params`(gf3 cnbj)。
- **best MAE**(@144000,148k 无 eval 但 ≈ 或微优):**@1=0.0066 / @10=0.0107 / @25=0.0139 / @50=0.0181**。
- @1 早 plateau(136k 起 0.0066 不动),@50 仍在微降 → 取最新 148000。

### 7.5.4 结论
- **裁决坐实**:从 PaliGemma base 冷启动(跳过 PI 机器人预训练)**确实收敛**(§0 YES 兑现),曲线干净无塌缩。
- **反超 warm-start**:对照 Exp-C `v3early_dagger_ptsfix`(warm-start best 49999)@1=0.0081/@50=0.0274 → **from-PaliGemma 全 horizon 更优(@1 −19%、@50 −34%)**(⚠️ 同 val 可比,但 offline 非终判)。
- **下一步**:真机 rollout 终判;若真机 work → 该路线(自有数据 + 不依赖 PI 预训练)成立。

---

## 8. 开放问题(调研未能直接 de-risk)
1. 有没有人发表过**真机双臂可变形/叠衣、百万帧、from-PaliGemma-base(无机器人预训练)**的收敛结果?(目前证据全是仿真单臂短程)
2. 双本体 co-train 的最优配比 + per-embodiment norm:部署目标本体该上权多少?混训 2 本体 vs 每本体单独训一个模型,哪个更好?
3. PI "from scratch" LIBERO run 的完整超参表(LR/warmup/wd/EMA/horizon/FAST vocab/web 配比)论文未给全 → §4 数字是起点需调。
4. kai0/openpi 里 KI(stop-grad + FAST-NTP + flow expert)的最小正确实现?有无社区 fork 现成?

---

## 9. ⭐ 变体实验:用 LeWM image encoder 替换 SigLIP(2026-06-17)

> **目标(用户)**: 把 pi0.5 的视觉编码器从 **SigLIP** 换成 **kai0 数据训出的 LeWM image encoder**,**不改官方架构**(新建一个变体模型,SigLIP 路径保持不动),仍按本文档从 PaliGemma base 起训;分 **freeze** 与 **no-freeze** 两个 run 提交。
> **状态**: ✅ 调研完成 + 3 决策定档(§9.4,B1/忠实分辨率/DINOv3-L16);**下一步:实现 LeWMVisionEncoder → gf0 2卡 smoke → 提交 freeze/no-freeze 两 run**。

### 9.1 ⚠️ 先厘清:LeWM 的"image encoder"到底是什么(已查 ckpt + 源码)

ckpt `gf3:.../lewm-kai0-3view-V1-aa27438-TS0616.1510/lewm-kai0-3view_epoch_10.pt`(182M):

| 组件 | 在 ckpt? | 是什么 |
|---|---|---|
| **DINOv3-ViT-L/16**(冻结)✅已查实 | ❌ 不在(HF 现成,gf3 缓存 `.CACHE/hf_cache/hub/dinov3-vitl16-pretrain-lvd1689m`)| 像素→patch 特征,1024-d,patch=16,CLS+**4 register token**。提取脚本 `scripts/cache_dinov3L_kai0.py`。⚠️**不是** DINOv2/14 |
| **OctCompactor**(per-view th/hl/hr,**学习的**)| ✅ 在(`compactor.{th,hl,hr}`)| 每视角:可学习 1 CLS + `n_obj=4` queries 对 DINO patch 做 cross-attn(depth2/heads4,1024→256)→ **每视角 5 token(256-d)** |
| predictor(AC-WM)/ sigreg / distill | ✅ 在 | 世界模型预测头 / 高斯正则 / 蒸馏 adapter —— **本任务不用**(只取视觉编码) |

→ **"LeWM image encoder" = DINOv2-L(冻结,现成)+ 学习的 OctCompactor(kai0-3view 特定)**。`VIEWS=(th,hl,hr)` 正好对齐 pi0.5 三相机(top_head/hand_left/hand_right)。`action_dim=14`(kai0)。
- ✅ **per-view 分辨率(忠实复刻,Q2 用户定)**:kai0 原生 640×480 → **top_head 纯 resize 384×288(W×H)→24×18=432 patch**;**hand_left/right 256×192→16×12=192 patch**(patch16);INTER_AREA;ImageNet mean/std;取 CLS+4reg 之后的 last-P patch token 喂 compactor。⚠️ 与 pi0.5 的 224 resize_with_pad **完全不同**,需为 LeWM 分支单独走这套 per-view 预处理。

### 9.2 与 pi0.5 现状的对比(token 经济学差异巨大)

| | SigLIP(现状)| LeWM encoder(本变体)|
|---|---|---|
| backbone | SigLIP So400m/14(~400M)| DINOv2-L/14(~300M,冻结)|
| 每相机 token | 256 × 1152-d | **5 × 256-d**(1 CLS + 4 obj)|
| 3 相机总 token | **768** | **15**(object-centric,极压缩)|
| 接入 LLM | proj 1152→gemma_width | **新建 proj 256→gemma_width**(随机初始化,训练)|

→ 把 768 个 dense 视觉 token 换成 15 个 object-centric token。**研究问题**:这种 kai0 预训练的紧凑 object 表示能否驱动 PaliGemma LLM 训出叠衣策略(信息够不够 / 是否更好的归纳偏置)。

### 9.3 集成设计(不动官方架构,新建变体)

- ⚠️⚠️ **必须走 PyTorch 路径**(`models_pytorch/pi0_pytorch.py` + `scripts/train_pytorch.py`):LeWM compactor 是 PyTorch 权重,移植到官方 JAX/Flax 路径工程量极大。**本变体 = PyTorch**(本文档 B1/B2 是 JAX 描述,但 init 思想[PaliGemma base + 随机 action expert]在 PyTorch 同样成立)。
- **新模块** `LeWMVisionEncoder`(新文件,不改 SigLIP):`DinoBackbone(dinov2_vitl14, frozen)` → per-view `OctCompactor`(载入 ckpt 权重)→ concat 3 view = 15 token(256-d)→ `nn.Linear(256, gemma_width)` 投影 → 当作 image tokens 喂给 PaliGemma LLM(替换 `embed_image` 的 SigLIP 输出)。
- **接入点**:`pi0_pytorch.py:embed_image`(:200)加分支 `if config.vision_encoder=="lewm": return lewm_encoder(imgs)` else 原 SigLIP。**官方 SigLIP 分支零改动**。
- **image mask / ar_mask**:15 个 LeWM token 仍按 image-token 处理(双向 attention),与 SigLIP 逻辑一致,只是数量 768→15。
- **LLM init**:PaliGemma VLM base(本文档主旨,`PaliGemmaWeightLoader` 思想的 PyTorch 等价);action expert 随机。**SigLIP 不再加载**(被 LeWM 替代)。
- **DINO 输入分辨率**:✅ 忠实复刻(top 384×288 / wrist 256×192,patch16;§9.4-Q2)。
- **依赖**:DINOv2-L hub 权重(gf3 已缓存,需同步到训练集群 TORCH_HOME 离线);LeWM ckpt(取 `compactor.*` 子树)。

### 9.4 决策定档(✅ 2026-06-17 用户确认 / 数据锁定)

1. ✅ **Q1 配方 = B1 naive-from-PaliGemma-base**(随机 action expert,~150–240k step,符合"从 base 自训"主旨)。
2. ✅ **Q2 = 忠实复刻 per-view 分辨率**(top 384×288 / wrist 256×192,patch16;frozen compactor 在训练分布内)。
3. ✅ **Q3 = DINOv3-L/16**(数据锁定:`cache_dinov3L_kai0.py` + patch16 + 1024d + 432/192 patch 证实;**非自由选项**,frozen compactor 只认 DINOv3 特征,用 DINOv2 即废)。权重 gf3 HF 缓存 `dinov3-vitl16-pretrain-lvd1689m` → 同步训练集群 `HF_HUB_OFFLINE=1`。

### 9.5 两个提交 run(freeze / no-freeze,单变量 = compactor 是否训练)

| run | DINOv2-L | LeWM OctCompactor | proj 256→width | PaliGemma LLM + action expert | 目的 |
|---|---|---|---|---|---|
| **L-freeze** | 冻 | **冻**(用 LeWM 学到的表示)| 训 | 训(LLM=base init,不冻)| 测 LeWM 表示**直接可用性** |
| **L-nofreeze** | 冻 | **训**(随策略微调)| 训 | 训 | 测 compactor **适配叠衣**后上限 |

> DINOv2-L 两 run 都冻(标准做法,300M 现成 backbone);差异只在 compactor freeze 与否。⚠️ 注意 KI 教训(§2 F2):**不冻 PaliGemma LLM**——这里冻的是视觉前端(DINO/compactor),不是 LLM。
- **数据**:沿用本文档(kai0 base+dagger + vis base+dagger,domain-cond)或先单本体 smoke;**norm 用 LeWM stats 还是 openpi 各自重算待定**(LeWM ckpt 带 state/action mean-std,但 pi05 action_dim=32 padding,需对齐)。
- **集群**:单节点 8×A100(cnsh)或 gf0 本地 2×A100 先 smoke;`scripts/train_pytorch.py <config> --exp_name=...`。

### 9.6 落地步骤(待 §9.4 确认后)
1. 把 LeWM ckpt(`compactor.*`)+ DINOv2-L hub 权重同步到 gf0/集群,提取 compactor 子 state_dict。
2. 实现 `LeWMVisionEncoder`(DinoBackbone + 3×OctCompactor + proj),单测 forward(3×[B,3,H,W]→[B,15,width])+ 载权重 strict 校验。
3. `pi0_pytorch.py` 加 `vision_encoder` 分支(SigLIP 不动)+ config 字段。
4. 注册 2 个 config `pi05_lewm_{freeze,nofreeze}`(克隆 pytorch flatten-fold + PaliGemma-base init)。
5. **gf0 2 卡 smoke**(forward/backward/loss 下降)→ 通过再 8 卡提交两 run。
6. eval:val MAE + 真机;对照 SigLIP 基线。

> **诚实提示**:这是**研究级架构手术**(PyTorch 前端替换 + DINO 依赖 + token 768→15 + 分辨率对齐),非配置级改动。3 问已定(§9.4);我**不会在 gf0 2卡 smoke(forward/backward/loss)通过前直接提交 8 卡训练**。实现→smoke→提交两 run。

---

## 关联
- LeWM 源: `gf3:/vePFS-North-E/shared_data/shock/distill-wm/`(`kai0_lewm.py` Kai0LeWM/OctCompactor · `dino_backbone.py` DinoBackbone · ckpt `data/exps/lewm-kai0-3view-V1-*/lewm-kai0-3view_epoch_10.pt`)
- pi05 PyTorch 接入: `kai0/src/openpi/models_pytorch/pi0_pytorch.py`(embed_image:200)· `preprocessing_pytorch.py`(resize 224)
- 数据/已有 work 锚点: `task_a_new_smooth_800_new_norm_results.md`(warm-start MAE@1=0.0089 基线)
- 双本体已有: `pi05_kaivis_perdsnorm_cond`(config.py:1036)domain conditioning + domain_weights
- 代码: `kai0/src/openpi/training/weight_loaders.py:64`(PaliGemmaWeightLoader)· `config.py`(新建 config)
- 调研 primary: pi0 `2410.24164` · pi0.5 `2504.16054` · Knowledge Insulation `2505.23705` / `pi05_KI.pdf` · open-pi-zero(github.com/allenzren/open-pi-zero)· openpi DROID `README_train.md`
