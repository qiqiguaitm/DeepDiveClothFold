# Action 表示(delta / abs)兼容设计 + 当前版本基线快照

本文档两部分:
- **Part A — 当前版本基线快照(v1, delta)**:冻结记录现行生产模型、norm、mask、配置、指标,作为回滚/复现基准。
- **Part B — delta/abs 兼容设计规范**:让数据处理 / norm 计算 / 模型架构 / 离线推理 / 在线推理对 delta 与 abs(以及 per-embodiment 混合)统一兼容,abs = mask 全 False 的退化情形,模型架构零改动。

> 关键判断(背景):delta/abs 的取舍由 **base checkpoint 的预训练分布**决定,不是第一性原理。pi0.5 base 训于 absolute → kai0 fold 用 abs;GWP/Policy-DROID base 训于 delta → 用 delta。两者机制等价(per-embodiment mask),abs 只是 mask 全 False。详见 `docs/gigaworld_policy_recipe_vs_experiment.md` 与项目记忆 `wam-delta-action-fix`。

---

## Part A — 当前版本基线快照(v1)

**冻结时间**:2026-06-09 · **repo git**:`b6fc8d7` (main)

### A.1 模型

| 项 | 值 |
|---|---|
| 架构 | `CasualWorldActionTransformer`(`world_action_model/models/transformer_wa_casual.py`),causal-masked |
| backbone | Wan2.2-TI2V-5B-Diffusers(`../checkpoints/Wan2.2-TI2V-5B-Diffusers`) |
| action_dim | 14(左臂6 + 左夹爪 + 右臂6 + 右夹爪) |
| num_frames | 48 · dst_size (256,192) · 3 views(cam_high / cam_left_wrist / cam_right_wrist) |
| flow_shift | 5.0 · expand_timesteps=True · state_repeats=1 |
| loss 权重 | `lambda_action=5.0, lambda_video=1.0`(官方 5:1;日志里 action_loss 是加权后 ≈5× 真值,看收敛需 ÷5) |
| EMA | **关闭**(`with_ema=False`;eval 用 raw transformer,EMA 滞后会造成假平台,见记忆 `wam-eval-ema-pitfall`) |

**生产 checkpoint(当前线上)**:
```
runs/visrobot01_fold_aihc_latent_5x/models/checkpoint_epoch_1_step_50000/transformer
```
DeepSpeed zero2 分片;`zero_to_fp32.py` 可合并;`pytorch_model/` 为权重目录。

**训练配置**:`world_action_model/configs/visrobot01_fold_aihc_latent_5x.py`
- 数据:`../kai0/data/wam_fold_v1`,`visrobot01_train ×3` 上采样 + `kairobot01 ×1`(≈3:1 跨 rig 平衡)
- 优化:`CAME8Bit` lr=2^-13.5(≈8.6e-5),`WarmupCosineScheduler` warmup 2000 / decay 50000
- 训练:`max_steps=50000`,bs/gpu=2 × 8 GPU,bf16,activation_checkpointing=True,latent 预缓存(`skip_video_decoding=True`)

### A.2 Action 表示与 mask(v1 现状)

- **表示**:**DELTA**(12 个臂关节)+ **ABSOLUTE**(2 个夹爪 idx 6/13)。
- **mask `_piper14`**:`[T,T,T,T,T,T, F, T,T,T,T,T,T, F]` — `True`=delta(减 state),`False`=abs。
- **现状缺陷(待 Part B 修)**:mask 在 **5 处硬编码且各自重复**,无单一真值源,易漂移:
  - `world_action_model/transformers/wa_transforms_lerobot.py:248`(`_piper14` + `delta_mask_templates={0,1}`)
  - `scripts/inference_server.py`(`--delta_mask` 默认 `"1,1,1,1,1,1,0,1,1,1,1,1,1,0"`)
  - `scripts/wam_pipeline/{eval_watch,episode_report,viz_traj}.py`(各自同款默认字符串)
  - `scripts/wam_pipeline/compute_norm_stats_fast.py`(`PIPER_MASK`)

### A.3 Norm 统计(v1, zscore)

- **norm_mode**:`zscore`(mean/std);json 同时含 q01/q99,但训练/eval/serving 默认走 zscore。
- **文件**(按 embed_id 索引):
  - embed_id 0 = visrobot01 → `assets_visrobot01/norm_stats_vis.json`
  - embed_id 1 = kairobot01 → `assets_visrobot01/norm_stats_kai.json`
- **action 统计已是 delta 空间**(关节 mean≈0、范围窄;夹爪 mean≈0.02、范围 [0,0.08]→证实 joints=delta / grippers=abs):

`norm_stats_vis.json` · `action`:
```
mean = [-.001,-.002,.001,.001,.0,-.001, .026,  .0,-.002,.001,-.001,.0,.001, .019]
std  = [.113,.356,.318,.232,.206,.194, .028, .091,.326,.330,.156,.168,.114, .027]
q01  = [-.537,-1.223,-.936,-.689,-.698,-.735, .0, -.311,-.922,-1.010,-.587,-.591,-.374, .0]
q99  = [.349,1.077,1.090,.906,.636,.624, .08,  .326,1.130,1.115,.472,.479,.371, .079]
```
`norm_stats_vis.json` · `observation.state`(绝对关节,q01/q99 ±2.3):
```
q01 = [-.779,.0,-2.322,-1.359,.092,-.599, .0, -.476,-.001,-2.527,-.421,.338,-.900, .0]
q99 = [.423,2.406,-.442,.741,1.221,1.305, .08, .699,2.395,-.403,1.106,1.222,.352, .079]
```
`norm_stats_kai.json` · `action`:
```
mean = [-.007,.016,-.019,.0,.001,-.002, .027, .001,.019,-.028,-.002,.001,.001, .034]
q01  = [-.570,-1.022,-1.039,-.602,-.629,-.495, -.001, -.349,-1.056,-1.163,-.582,-.565,-.459, .0]
q99  = [.365,1.351,1.003,.630,.530,.495, .099, .417,1.408,1.006,.534,.524,.423, .099]
```
> 对照参考:abs-action 的 joints q01/q99 会是 ±2.3(同 state),与上面 delta 的 ±0.5~1.2 区分明显——这是判断一份 stats 属 delta 还是 abs 的最快依据。

### A.4 离线指标(step 50000, 200 eps)

| 指标 | GWP v1 (delta) | pi0.5 baseline |
|---|---|---|
| mae@1 | **0.0028** | 0.0219 |
| mae@10 | 0.0347 | 0.0425 |
| mae@24 | 0.0720 | 0.0743 |
| mae@48 | 0.1128 | 0.1155 |
| latency | action 636 ms / video 1126 ms | — |

> mae@1=0.0028 远低于 pi0.5——其中含 delta 锚定效应(t=1 锚点精确),非纯策略质量;判 abs/delta 优劣需看闭环 SR,勿只看 mae@1。

### A.5 端到端不变量(v1 已验证 round-trip ~0)
```
训练:  norm_action = ( (action − state·m) − μ ) / σ
上线:  action_hat  = denorm(model_out)·σ + μ + state·m        # m = _piper14
```
反算公式 `add_state_to_action` 仅存在于后处理(`pipeline/utils.py:200`),被 server + eval 脚本调用;**模型/pipeline 对 mask 无感**。

---

## Part B — delta/abs 兼容设计规范

### B.1 三条不变量
1. **mask 是唯一开关**;`abs = 全 False`,无独立代码路径(per-dim / per-embodiment 一律用 mask 表达)。
2. **mask 与 stats 物理绑定**:把 `delta_mask` 写进 `norm_stats` json(stats 本就在"减完 state 后"统计,天然属于某 mask)。消除 A.2 的五处漂移。
3. **一个 checkpoint 绑定一个 (mask, stats) 家族**。模型权重 mask-specific:delta 训的 ckpt 不能用 abs mask 上线。"同时支持" = ①代码 config 可选 + ②两个 ckpt(或一次 cross-rig 跑里 rig0=delta、rig1=abs)。与 kai0(`pi05_flatten_fold` abs / `..._task_a_base_delta` delta 两套 config)一致。

### B.2 唯一真值源:`norm_stats` 内嵌 mask
每 embodiment 一份 stats(`norm_path` list 按 embed_id 索引,`_get_stats_dict` 已实现)。schema 增顶层字段:
```json
{
  "norm_stats": { "observation.state": {...}, "action": {...} },
  "delta_mask": [true,true,true,true,true,true,false, true,true,true,true,true,true,false],
  "action_repr": "delta"          // "delta" | "abs" | "mixed";仅供人读/校验
}
```
abs 版 = 同 schema、`delta_mask` 全 false、且 `action` 统计在绝对动作上重算。**切 abs/delta = 换 `norm_path` 指向另一份 stats,别的不动。**

共享 helper(置 `world_action_model/pipeline/utils.py`,五处共用):
```python
def resolve_delta_mask(stats_dict, action_dim, fallback=None):
    m = stats_dict.get("delta_mask")
    if m is None:                       # 老 stats 无字段 → 回退,保持 v1 行为
        m = fallback if fallback is not None else [True]*6+[False]+[True]*6+[False]
    m = np.asarray(m, dtype=bool)
    if len(m) < action_dim:
        m = np.pad(m, (0, action_dim-len(m)), constant_values=False)
    return m[:action_dim]
```

### B.3 逐阶段改动

| 阶段 | 文件 | 改动 | 兼容策略 |
|---|---|---|---|
| **1. norm 计算** | `scripts/compute_norm_stats.py`,`scripts/wam_pipeline/compute_norm_stats_fast.py` | 已有 `--delta-mask`;算完把 `delta_mask`+`action_repr` **写进输出 json** | abs 跑法=`--delta-mask` 全 false → 产 `*_abs.json`;delta → `*_delta.json`。**mask 诞生地,后续全继承** |
| **2. 数据处理** | `wa_transforms_lerobot.py:246-258` | 删 `_piper14`/`delta_mask_templates` 硬编码,改 `base = resolve_delta_mask(stats_dict, d, fallback=_piper14)`(`stats_dict` 已按 embed_id 选好) | mask 自动 per-embodiment;老 stats 回退 `_piper14`,v1 行为不变 |
| **3. 模型架构** | transformer / `wa_pipeline.py` | **零改动** | 只认 `action_dim`;abs/delta 对模型不可见 |
| **4. 离线推理/eval** | `eval_watch.py` `episode_report.py` `viz_traj.py` | `--delta_mask` 默认改 `None` → 从 stats `resolve_delta_mask`;`--delta_mask` 仅作 override | 不传即自动跟 stats,永不错配;eval 的 sub-chunk 重锚用同一 mask |
| **5. 在线推理** | `inference_server.py:56,110` | 同上:从 `--stats_path` 的 json 读 mask;`--delta_mask` 可 override | 上线只需指对 `--stats_path`;abs/delta 由 stats 文件决定 |

### B.4 不变量与正交项
- **round-trip 自检(上线前必跑)**:`abs == add_state_to_action(delta_normspace(abs−state·m)) `,误差应 ~0。m 全 False 时退化为纯 abs(`action_hat = denorm + 0`)。
- **norm_mode 与 mask 正交**:`--norm_mode {zscore,minmax}` 是 z-score vs quantile([-1,1]),与 delta/abs 无关;stats json 四个量(mean/std/q01/q99)都在,两 mode 都能跑。本设计只动 mask。

### B.5 实现清单(改动局部)
1. `pipeline/utils.py`:+ `resolve_delta_mask()`(唯一真值源)。
2. `compute_norm_stats.py` + `compute_norm_stats_fast.py`:输出写入 `delta_mask`/`action_repr`。
3. `wa_transforms_lerobot.py:246-258`:删硬编码,改调 helper。
4. `inference_server.py` / `eval_watch.py` / `episode_report.py` / `viz_traj.py`:`--delta_mask` 默认 `None` → 从 stats 读,保留 override。
5. 生成 `*_abs.json`(全 false mask 重算 stats),与现有 vis/kai delta stats 并列;config 经 `norm_path` 二选一。
6. **回填**:给现有 `norm_stats_vis.json` / `norm_stats_kai.json` 补 `"delta_mask": _piper14` + `"action_repr":"delta"`,使 v1 也纳入新机制(无字段也能回退,但显式写入更稳)。

**净效果**:abs/delta(及 per-rig 混合)= 选一份 stats 文件的事;模型/架构/训练循环代码完全共用;mask 永远跟着配套 stats,五处不再失配。等价 kai0,且因 mask 内嵌 stats,比 kai0 "config flag + 手保持一致" 更难出错。

---

## Part D — 小规模实测对比(delta vs abs, 2026-06-09)

用**完全相同**的配方在 b0/b1 各起一个 1000-step 训练验证重构正确性 + 直接对比两种表示:
- 配置:`visrobot01_fold_cmp_{delta,abs}.py`(继承 5x latent 配方;唯一差别=`norm_path` 指 delta 或 `*_abs.json`),同一初始化(action head from scratch + Wan backbone)、5:1 loss、warmup100/cosine、cross-rig vis×3+kai、EMA off。
- 训练:**两个 1000 步全程 0 error**,各落 step500/1000 ckpt。→ 实跑验证了重构(mask 内嵌 stats + 关键 `action_dim_mask` line-308 NameError 修复)在真实 DeepSpeed 循环里端到端不回退,delta 与 abs 路径都跑通。
- 评测:`eval_watch` 在 visrobot01_val 上 coverage=exec、**同一套 17,993 windows**,`--delta_mask` 留空→**从各自 stats 内嵌 mask 自动解析**(delta→piper14 重建 `abs=denorm+state·m`;abs→全 False 即纯 `denorm`)。这同时实测验证了推理/eval 侧的 mask 解析。

| 指标(↓ 越小越好) | **delta @1k** | **abs @1k** | abs/delta 倍数 |
|---|---|---|---|
| mae@1  | **0.130** | 0.256 | 1.97× |
| mae@10 | **0.141** | 0.261 | 1.85× |
| mae@24 | **0.180** | 0.274 | 1.52× |
| mae@48 | **0.235** | 0.303 | 1.29× |
| action_mae | **0.181** | 0.276 | 1.53× |
| action_mse | **0.074** | 0.165 | 2.22× |
| mae_move | **0.223** | 0.330 | 1.48× |
| psnr / ssim(视频) | 18.82 / 0.702 | 18.98 / 0.708 | ≈持平 |

**读数**:
1. **同等 1k 预算下 delta 全面优于 abs**(所有动作 horizon),action_mse ~2.2×。
2. **gap 随 horizon 单调收窄**(mae@1 1.97× → mae@48 1.29×):正是 delta 的**锚定效应**——目标锚在当前 state,短步预测几乎白送;长 horizon 两者都得预测远期运动,优势缩小。
3. **视频 PSNR/SSIM 几乎相同**:视频生成与动作表示无关,符合预期(交叉验证 eval 管线正常)。
4. **但这是 undertrained(1k)对比**:mae@1 的 delta 优势含表示/度量红利(delta 拿到 state 当锚)。不能据此断言 abs "更差"——pi0.5 用 **absolute** 在充分训练后 mae@1=0.0022(全场最佳),说明 abs 并非不可行,只是收敛慢、且需 base 分布对齐。**1k 步内 delta 的先发优势主导;abs 是否在 50k 追平,本实验不回答。**

**结论(对"abs 是否更合理")**:对当前 WAM(Policy-DROID/GWP 的 delta-预训练 base),**delta 是正确选择**,实测也更快收敛——切 abs 会重新引入分布失配(正是上次修掉的 55× 退化)。abs 的合理性只在"base 本身 absolute(如 pi0.5)+ 足够训练 + 关心真机对 proprioception 噪声的鲁棒性"时成立。本兼容设计的价值=让这条对照实验**只改一行 `norm_path`** 即可复现。

复现:`world_action_model/configs/visrobot01_fold_cmp_{delta,abs}.py` + `scripts/wam_pipeline/_orchestrate_cmp_eval.sh`;结果 `runs/cmp_{delta,abs}_1k/eval_log.jsonl`。

---

## Part C — 实施状态 / 用法(2026-06-09 已落地)

### C.1 已完成
- **helper**:`world_action_model/pipeline/utils.py` 新增 `resolve_delta_mask(stats, action_dim, fallback)` + `DEFAULT_PIPER14_DELTA_MASK`(唯一真值源)。
- **norm 计算**:`scripts/compute_norm_stats.py` 新增 `serialize_json_with_mask()`,输出写入 `delta_mask`+`action_repr`;`compute_norm_stats_fast.py` 同步改用。
- **数据处理**:`wa_transforms_lerobot.py` 删硬编码 `_piper14`/`delta_mask_templates`,改读 `resolve_delta_mask(stats_dict, d, fallback=piper14)`。
- **推理/在线**:`inference_server.py` + `wam_pipeline/{eval_watch,episode_report,viz_traj}.py` 的 `--delta_mask` 默认改 `""`(空=从 stats 取),非空作显式覆盖。
- **回填**:`norm_stats_{vis,kai}.json` 补 `delta_mask=piper14`/`action_repr="mixed"`(关节 delta+夹爪 abs → 严格说是 mixed;mask 数组才是权威,`action_repr` 仅人读)。
- **abs stats 生成**:`norm_stats_{vis,kai}_abs.json`(全 False mask 全量重算;joints q01/q99≈±2.3、mean=绝对关节位、与 state 同分布;grippers 不变)。
- **自检**:`scripts/wam_pipeline/check_delta_abs_roundtrip.py`,vis/kai 的 delta 与 abs、zscore 与 minmax 全部 round-trip max_err ~1e-7,fallback(旧 stats→piper14)通过。

### C.2 切换 delta ↔ abs(用法)
**只改 config 的 `norm_path`,其余不动**:
```python
# delta(现行 v1):
norm_path = ["./assets_visrobot01/norm_stats_vis.json",     "./assets_visrobot01/norm_stats_kai.json"]
# abs:
norm_path = ["./assets_visrobot01/norm_stats_vis_abs.json", "./assets_visrobot01/norm_stats_kai_abs.json"]
# per-rig 混合(如 vis=delta、kai=abs)亦可,按 embed_id 各指一份。
```
- 训练:transform 自动从所选 stats 的内嵌 mask 走 delta/abs;**需重训得到对应 ckpt**(权重 mask-specific,delta ckpt 不能配 abs stats 上线)。
- eval/serve:`--stats_path` 指向同一份 stats 即可,`--delta_mask` 留空自动跟随;无需再手传 mask 字符串。
- 自检:上线/重训前跑 `python -m scripts.wam_pipeline.check_delta_abs_roundtrip --stats <那份 stats>`,确认 max_err~0。

### C.3 注意
- `--delta_mask` 仍保留为显式覆盖入口(debug / 老脚本兼容);正常流程留空。
- 新出的 stats 一律带 `delta_mask`;**旧 stats 无字段时回退 piper14**——若某旧 abs 数据误用回退会被当成 delta,故新数据务必用本套 `compute_norm_stats*` 重算以写入正确 mask。
