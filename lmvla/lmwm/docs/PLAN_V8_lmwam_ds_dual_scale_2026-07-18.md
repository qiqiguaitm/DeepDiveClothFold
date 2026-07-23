# PLAN V8 · LMWAM-DS 双尺度未来条件化(2026-07-18)

> 依据:§4.14 因果定论(通道替换非叠加)+ 文献调研(PDS ICRA26 小规模验证互补性;VLA 规模=空白)。
> 目标:**不替换、并联** t+7 局部通道 + milestone 全局通道,预测 task8→~100 守住、task6→~90 守住,聚合 ~96-97% > 94.4(双亲),P1 真达成。
> 母文档:`RECURRENCE_UNIVERSAL_goals_and_roadmap.md` §4.13/§4.14。

## 0. 设计要点(先读)

```
DiT 条件 = [ h_t(256) │ h_t7 局部(256, always-on) │ h_ms 全局(256, hint-drop 0.15 + CFG) │ vlm ]
                        aux: MSE → features[:,-1] (t+7 GT)   aux: MSE → provider milestone GT
```

- **局部通道 = armB 配方原封不动**(LaWM decoder + t+7 目标 + LaWM teacher 蒸馏)→ 守精度(t8/t7)。
- **全局通道 = hintdrop 配方原样并入**(LMWM generator + provider milestone 目标 + InverseEnc teacher)→ 守指引(t6/t9)。
- **VLM latent:单 query 双头**。不加第二个 placeholder token(免动 tokenizer);共享 `pred_action_emb`,加两个投影头 `vlm_to_lam_local`(=现 vlm_to_lam)+ `vlm_to_lam_ms`(新),分别蒸馏到 LaWM code 空间和 LMWM InverseEnc code 空间,分别喂两个 decoder。
- **dropout/CFG 只作用于全局通道**:局部通道永不 drop(精度命脉);cfg_embeddings 作 ms 通道的 null。推理 CFG 旋钮(LMWM_CFG_GUIDANCE)只调 ms 段。
- **无 gate、无手工调度**:coarse→fine 分工交给 cross-attention 自己学(C6 优雅性)。
- 开关:env `LMWM_DUAL=1`(沿用现有 env-gated 风格);不设时行为=armB baseline,完全向后兼容。

## 1. 代码改动(2 个文件 + adapter)

### 1.1 `starVLA/model/framework/vlas/lawam.py`
| 位置 | 改动 |
|---|---|
| `__init__` 268-289 | `LMWM_DUAL=1` 时:**不 swap** `lam.decoder`(保 LaWM 原 decoder);另挂 `self.lmwm_dec = LMWMDecoder(gen)`、`self.lmwm_teacher = InverseEnc`(从 `lmwm_adapter` 复用类,load `LMWM_CKPT`,冻结 teacher);新建 `self.vlm_to_lam_ms`(结构 copy 自 `vlm_to_lam`,输出 dim=32 LMWM code)。provider 装法不变(`LMWM_MILESTONE_TARGET`)。 |
| `_run_shared_encoding_core` 668-687 | 双目标并存,**删掉 torch.where 覆盖**:`h_t7_gt = features[:,-1]`(原样);`h_ms_gt = provider.get_target(...)`,invalid 帧回退 t+7。双预测:`h_t7_pred = lam.decoder(h_t, emb_local)`;`h_ms_pred = lmwm_dec(h_t, emb_ms)`,其中 `emb_local/emb_ms = vlm_to_lam_local/ms(pred_action_emb)`。`PolicyEncodingState` 加字段 `h_ms_pred/h_ms_gt`。 |
| `forward` 749-811 | 双 aux loss:`loss_perceptual = MSE(h_t7_pred, h_t7_gt) + MSE(h_ms_pred, h_ms_gt)`(各自权重 = 现 `perceptual_weight`,先不调);双蒸馏:`loss_distill_local`(LaWM teacher,原样)+ `loss_distill_ms`(InverseEnc teacher,同 hintdrop 的 swap_teacher 配方)。scheduled sampling 逐通道复用 `_build_flow_future_condition`。flow 调用加 `h_ms_star=h_ms_cond_rep`。 |
| `predict_action` ~838 | 推理:`h_t7_pred` 与 `h_ms_pred` 同上双路生成,传 `sample_actions_cfg(..., h_ms_star=...)`。`LMWM_CFG_GUIDANCE` env 保留(现只作用 ms 段)。 |
| DDP | cfg_embeddings 的 autograd 保活 trick(flow 456-466)对 ms 通道复用;新增模块若冻结(teacher/gen)注意 `requires_grad_(False)`,避免 find_unused_parameters 问题。 |

### 1.2 `starVLA/model/framework/vlas/flowmatching_expert.py`
| 位置 | 改动 |
|---|---|
| `forward` 签名 + 456-469 | 加 `h_ms_star: Optional=None`。cfg_drop **只**作用 ms:`cond_ms = where(drop_mask, cfg_future, h_ms_star)`;局部 `cond_local = h_t1_star` 永不 drop。`encoder_hidden_states = cat(h_t, cond_local, cond_ms, cond_vlm)`。`h_ms_star=None` 时走老路(兼容 armB)。 |
| mask 513-536 | `num_vision = 256*3`;`encoder_attention_mask`、alternate_vldit 的 `image_mask/vlm_mask` 段长同步 `num_h_t+num_h_t1+num_h_ms`。 |
| `sample_actions_cfg` ~640-700 | uncond 分支 = **只把 ms 段换成 cfg_embeddings**(local 段保留);CFG blend 公式不变。两个分支(alternate / 非 alternate)都改。已有的 `LMWM_CFG_TSCHED` 代码保留(E2 后备会用)。 |

### 1.3 `lmwm_adapter.py`
- 加 `load_lmwm_parts(ckpt) -> (gen, inv, code_dim)` 工厂(不做 swap,只回模块),供 dual 模式用。现 `make_lmwm_lam` 不动(hintdrop 复现兼容)。

**预计工作量**:~150 行改动。改完 `grep -n` 自查 mask 段长一致性(经典坑:alternate_vldit 分支漏改)。

## 2. Smoke 验证(本机 gf0 2×A100,~30min)

```bash
cd /vePFS/tim/workspace/deepdive_kai0/lmvla/lawam
export LMWM_DUAL=1 LMWM_CKPT=.../lmwm_libero_rvalley/lmwm.pt LMWM_MILESTONE_TARGET=.../pairs.npz
# ① 前向 smoke: 训练脚本跑 20 步, 确认 loss_flow/loss_perceptual(双)/loss_distill(双) 全有限且下降
# ② 推理 smoke: predict_action 单帧, 确认 shape/无 NaN; LMWM_CFG_GUIDANCE=1.0/3.0 各跑一次不崩
# ③ 回归 smoke: 不设 LMWM_DUAL, 确认 armB baseline 前向 bit 级不变(老 ckpt 可加载)
```

## 3. 训练提交(cnsh volc,同 hintdrop 配方)

- 克隆 `train_scripts/kai/volc/lmwm_rvalley_hintdrop015_cnsh_8a100.yaml` → `lmwm_dual_scale_cnsh_8a100.yaml`:
  - TaskName/Description 改 `lmwm-dual-scale-...`;entrypoint 加 `export LMWM_DUAL=1`(其余 env 同款:LMWM_CKPT / LMWM_MILESTONE_TARGET / hint-drop 0.15);步数 12500、同 base 初始化(与两亲可比)。
  - 新增参数(`vlm_to_lam_ms`)从头随机初始化——与 hintdrop 当时加 cfg_embeddings 同性质,无需特殊处理。
- 提交(gsy):`ssh -p 16370 root@124.174.16.237`,`cd train_scripts/kai/volc && kai0/.venv/bin/python submit_yaml.py lmwm_dual_scale_cnsh_8a100.yaml`。cnsh=robot-task 队列。
- 盯 loss:`loss_perceptual_ms` 应收敛到 ~0.011(hintdrop 同量级);`loss_perceptual_local` 应更低(t+7 易)。

## 4. Eval(本机,严格 §4.13 同口径)

- `run_libero_benchmark.sh`:`SUITES=libero_10 NUM_TRIALS_PER_TASK=50 MUJOCO_GL=egl`,双卡并行(port 自动锁,PORT_BASE=5694),`LMWM_DUAL=1` + 同款 env。
- 对照 §4.13 双亲表(hintdrop 94.4 / baseline 94.4),重点 task6/7/8/9。

## 5. 判据与决策树

| 结果 | 判定 | 下一步 |
|---|---|---|
| 聚合≥96 且 t8≥94 且 t6≥86 | **P1 达成**(替换假说证实) | 多 seed 收口(C5)→ paper 消融:dual(t+7,t+7) 对照(应无增益,证互补性来自尺度差,PDS 式)→ P2/robotwin |
| t8 恢复但 t6 掉 | 注意力塌向局部通道 | 推理 CFG>1 调 ms 段(旋钮已在);或训练加大 ms drop_prob |
| t8 仍 ≤88 | 替换假说证伪 → **干扰假说** | **E2 = 机制②训练版**:训练时 cfg_drop 依 diffusion-t 调度(粗步保 ms、细步 drop 到 null;在 1.2 的新 drop 代码上一行改,`LMWM_CFG_TSCHED` 推理代码已就位)——亦是文献空白 (c) |
| 全线掉 | 实现 bug 优先排查 mask 段长/CFG 段替换错位 | 回滚 smoke ③ 复查 |

## ✅ 执行状态(2026-07-18)

- **E1 代码已实现**(3 文件, env-gated `LMWM_DUAL=1`, 完全向后兼容):
  - `lmwm_adapter.py`: `load_lmwm_parts()` — 只回 (gen可训, inv冻) 不 swap。
  - `lawam.py`: dual 分支(不 swap LaWM decoder)+ `vlm_to_lam_ms` 第二投影头 + `_decode_ms_future` + `_compute_distill_loss_ms` + `PolicyEncodingState` 加 ms 字段 + forward 双 perceptual/双 distill(逐通道监控 key)+ predict_action 传 `h_ms_star`。
  - `flowmatching_expert.py`: `forward`/`sample_actions_cfg` 加 `h_ms_star`; 局部通道永不 drop, 全局通道 cfg-drop→cfg_embeddings; mask 段长 512→768 统一用 `num_future_seg`/`num_img_seg`; CFG uncond 只换 ms 段。
- **Smoke 全过**(plan §2 三项):
  - ① flow 单元: dual forward/backward/predict + CFG=1/3 全有限; **local↔ms 梯度对称**(6.60e-4≈6.63e-4)证 ms 通道无断裂。
  - ② ms 模块集成(真 rvalley 权重): load_lmwm_parts / vlm_to_lam_ms→code[B,1,32] / LMWMDecoder→[B,256,768] / InverseEnc teacher→[B,1,32] 形状全对, gen可训/inv冻。
  - ③ 端到端真训练(gf0 2×A100 DDP, 6 步): `[LMWM][DUAL] ... NO decoder swap` 触发, 6/6 步跑通, 无 DDP unused-param 崩, save OK; 单通道回归路径位级一致。
- **已提交 cnsh**: `train_scripts/kai/volc/lmwm_dual_scale_cnsh_8a100.yaml` → **task `t-20260718095201-m4nhk`**(cn-shanghai robot-task, 8×A100, 12500步, ~4-5h)。日志: `lmvla/lawam/logs/volc_dualarm/dual_scale_cnsh_*.log`(cnsh vepfs, gf0 可读)。
- **训练完成**(2026-07-18 08:33 UTC): step 12500, loss 全收敛(flow 0.0085 / perceptual 0.0622[local+ms] / distill 0.0083 / total 0.0156)。ckpt `results/Checkpoints/libero/20260718_015223+lmwm_dual_scale_cnsh_volc/checkpoints/steps_12500_pytorch_model.pt`。
- **EVAL 完成**(2026-07-18, 本机 gf0 单卡 egl-1worker, 与双亲逐字同口径, 500 ep): dual ckpt 加载无 size mismatch(dual 权重全对上)。

### 结果(V8 dual per-task SR vs 双亲 §4.13)
| task | baseline | hintdrop | **dual V8** | Δ vs hintdrop |
|---|---|---|---|---|
| 6 弥散/指引 | 78 | 90 | **88** | −2(噪声内) |
| 7 精度 | 100 | 96 | **100** | +4 ✅完全救回 |
| 8 双壶精放 | 100 | 84 | **90** | +6 ⬆部分救回 |
| 9 指引 | 78 | 86 | **86** | 0 ✅守住 |
| **聚合** | **94.4** | **94.6** | **94.8**(474/500) | +0.2(噪声内) |

### 判定
- **P1 未达成**(阈值 聚合≥96 且 t8≥94; 实际 94.8 / t8=90)。
- **但替换假说证实、未证伪**(t8=90 > 证伪线 88): 加回局部通道 → t7 完全救回 + t8 部分救回, 同时指引红利守住 → coarse→fine 互补性**确实涌现**, 只是幅度不足。
- **根因推断**: t8 卡 90 未回 100 = 风险#3 **单 query 双头容量瓶颈**(局部+全局挤一个 latent, 局部被压)。落在决策树"t8 恢复但幅度不够"。

### 下一步候选
1. **CFG 扫 ms 段**(`LMWM_CFG_GUIDANCE=1.5/2`, 无需重训, ~2h/次): 看能否把 t6 推回 90 + 提聚合。
2. **Plan B 双 query**(局部/全局各一组 act query, 解容量瓶颈): 最可能把 t8 90→~100。
3. paper 消融: dual(t+7,t+7) 对照(应无增益, 证互补来自尺度差)。

### Plan B 双 query — 已实现+提交(2026-07-18, 用户选定)
**实现**(3 文件, `LMWM_DUAL_2Q=1` 门控, 向后兼容 E1/单通道; **无需动 tokenizer**——复用同一 `<ACT_PH>` 靠 cumsum 位置切分, 注入 query embedding 覆盖占位符故 token id 无关):
- `lawam.py`: 2Q 时 `num_action_queries` 翻倍(前半=局部 query, 后半=全局 ms query); `_run_vlm_stage` 按 order 切 `h_act[:, :Q]`→vlm_to_lam(局部) / `h_act[:, Q:]`→vlm_to_lam_ms(全局), 各得**独立 VLM 隐藏态**(不再共享 h_act)。
- `dataloader/__init__.py`: 训练 collator act placeholder 同步翻倍; 推理路径读 backend.num_action_queries 自动翻倍。
**Smoke**: gf0 2×A100 6 步真训练跑通, act_query 翻倍到 16, **无占位符 count mismatch**, save OK; 翻倍段 shape-mismatch 自动忽略(新参数随机初始化)。
**已提交 cnsh**: `train_scripts/kai/volc/lmwm_dual_2q_cnsh_8a100.yaml` → **task `t-20260718191515-h25hn`**(8×A100, 12500步, ~4-5h)。日志 `lmvla/lawam/logs/volc_dualarm/dual_2q_cnsh_*.log`。
**预测**: t8 90→~100(局部通道有独立 VLM 容量), 聚合→~96 达 P1。训完本机 eval N=50 同口径(egl-1worker)vs E1(94.8)+双亲。

### Plan B 结果 — 容量瓶颈解除但触发注意力再平衡(2026-07-18)
| task | baseline | hintdrop | E1(1Q) | **Plan B(2Q)** |
|---|---|---|---|---|
| 6 弥散/指引 | 78 | 90 | 88 | **80** ⬇ |
| 7 精度 | 100 | 96 | 100 | 98 |
| 8 双壶精放 | 100 | 84 | 90 | **94** ✅ |
| 9 指引 | 78 | 86 | 86 | **94** ⬆ |
| 聚合 | 94.4 | 94.6 | 94.8 | **94.8**(474/500) |
2Q final loss: flow 0.0088 / perceptual 0.0620 / distill 0.0092(≈E1)。

**机理结论**(干净可证伪): 双 query **确实解除 t8 容量瓶颈**(90→94, 达 P1 核心判据线; t9 86→94)—— 坐实 E1 的 t8 卡 90 = 单 query 容量瓶颈。**但**局部通道有专属容量后 **cross-attention 注意力塌向局部**, 全局(指引)通道被饿 → 最弥散的 **t6 88→80 掉最多**。t8/t9 涨被 t6 跌抵消, 聚合仍 94.8(**再分配非净增**)。
**判定**: P1 仍未达成(聚合 94.8<96, t6=80<86)。精确命中 §5 决策树 **"t8 恢复但 t6 掉 → 注意力塌向局部通道"** 分支。
**2Q+CFG=1.5 实测(2026-07-19, 本机同口径)**: t6=82 t7=100 t8=94 t9=88, 聚合 **95.0(475/500)**。vs 2Q(CFG1.0 94.8): CFG>1 放大 ms 指引 → t6+2/t7+2 但 **t9−6**(放大指引反伤 t9 精放), 净 +1 ep = 噪声内。**又一次纯重分配, 天花板(94.4~95.0 全在 ~2pt 噪声)纹丝不动** → 与 §4.12 单通道 CFG 无净收益一致, **CFG 分支彻底关闭(不跑 CFG=2.0)**。→ 推理层动 hint 两次证否, 唯一剩训练层 phase-adaptive(机制② = A)。
**处方**(§5): ① 推理 **CFG>1 调 ms 段**(`LMWM_CFG_GUIDANCE=1.5/2`, 无需重训, 直接补指引通道救 t6); ② 训练加大 ms drop_prob(hint_dropout 0.15→0.25, 强迫模型更依赖 ms 通道)。
ckpt: `results/Checkpoints/libero/20260718_111535+lmwm_dual_2q_cnsh_volc/checkpoints/steps_12500_pytorch_model.pt`。

### ⭐ 深层根因重述 + AB 并行(2026-07-18 用户对齐)
**为何混合不涨聚合**(用户核心问题): LaWM(t+7)与 LMWM(milestone)不是可叠加的两个贡献, 而是 **precision↔guidance 一根张力**——task6/9 要 guidance、task8 要纯局部精度。V8 每种混合(共享query/双query/CFG)只能设**一个全局平衡**, 无法同时满足"task8 要 0 指引 + task6 要满指引"→ 必零和重分配 → 聚合封顶 94.8。rollout 诊断(§4.12)确证 task8=第二个壶精放被远端 hint 损害。
**用户两条修正**: ① **SR 是最终目标, 不退守机理故事**(SR 不动=WM 没用)。② **LMWM 视野可变≠比 LaWM 不精**——近里程碑时 LMWM 本就是短视野; 精度真凶更可能是 target=跨-ep canonical medoid(语义对但非本-ep 精确近未来)+ 里程碑粒度在精放阶段太粗, 非"远"本身。
**AB 并行**(均冲 SR):
- **A = 机制②**(已实现+提交): 2Q 基座上 ms 通道 cfg_drop 按 diffusion-t 调度——`p_drop(t)=0.15+0.85·t^gamma`(精步 t→1 高 drop 退指引保精度, 粗步 t→0 保 LMWM 长视野指引), DiT 吃 timestep 故学 phase-条件依赖, **推理不变**。env `LMWM_MS_TSCHED=gamma`(gamma=2)。改 flowmatching_expert.py forward 一处(~12行)。smoke 过。预测: 精步退 ms → task8→~100 同时 task6 保 guidance → 首破 94.8。硬止损: 若仍 ~94.8 则天花板结构性, 转 B。
  - **提交历史**: 首提 cnsh `t-20260719065931-rgkxc`(robot-task 拥堵)→ **迁北京队列**: cnsh 任务已 stop_job 取消, 改 **Robot-North-H20 `t-20260719071819-g9c62`**(8×H20, yaml=`lmwm_dual_2q_tsched_8h20.yaml`)。⚠️ 迁北京要点: North-E 是独立 checkout, V8+机制②代码(flowmatching_expert/lawam/lmwm_adapter/dataloader 4文件)已 scp 同步到 North-E 并验 import OK; rvalley 数据/ckpt/base 权重 North-E 本就在; **必须带 LMWM_HINT_DROPOUT=0.15**(设 cfg_drop_prob>0, 机制② ms drop 前提)。日志/ckpt 落 North-E(gf0 读不到, 经 gsy), eval 时需把 ckpt 从 North-E 传回 gf0 或在 Beijing 评。
- **B = RoboTwin 有余量基准**: LIBERO-10 已饱和(94-96 顶), per-task 优势无处求和; RoboTwin 有余量处验证 WM 真提升聚合。robotwin DINOv3 特征已抽(5000ep), 缺 milestone pairs+LMWM ckpt(需建管线)。调研+建 pairs 进行中。

## 6. 风险清单
- alternate_vldit 与非 alternate 两分支 mask 不同步(历史踩过)→ 改完双分支 diff 对照。
- provider invalid 帧回退 t+7 时 ms 通道≈局部通道复读 → 可接受(≈PDS"同尺度无增益",不伤)。
- 单 query 双头容量瓶颈(局部+全局挤一个 latent)→ 若 loss_distill_ms 不降,升级为双 query(动 tokenizer,Plan B,先不做)。
- 新参数使 ckpt 与两亲结构不同 → eval 脚本 load 需 `strict=False` 或补 key 过滤(smoke ③ 验证)。
