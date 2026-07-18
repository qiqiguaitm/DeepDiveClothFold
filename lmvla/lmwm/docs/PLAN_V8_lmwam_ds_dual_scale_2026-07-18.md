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

## 6. 风险清单
- alternate_vldit 与非 alternate 两分支 mask 不同步(历史踩过)→ 改完双分支 diff 对照。
- provider invalid 帧回退 t+7 时 ms 通道≈局部通道复读 → 可接受(≈PDS"同尺度无增益",不伤)。
- 单 query 双头容量瓶颈(局部+全局挤一个 latent)→ 若 loss_distill_ms 不降,升级为双 query(动 tokenizer,Plan B,先不做)。
- 新参数使 ckpt 与两亲结构不同 → eval 脚本 load 需 `strict=False` 或补 key 过滤(smoke ③ 验证)。
