# FLASH 移植实施日志 (kai0 视角)

> **配套**: [`flash_future_research.md`](flash_future_research.md) (课题与计划) — 本文是**执行日志**, 记录每一步的处理、决策、验证结果。
> **铁律 (用户要求)**: **绝不影响旧版本代码运行**。所有改动**只新增文件**, 不修改 `pi0_pytorch.py` / `serve_policy*.py` / ROS2 节点等现有推理路径; 一切新功能默认关闭、可独立 import、独立测试。
>
> **开始**: 2026-06-07
> **执行模型**: Claude Opus 4.8 (1M)

---

## 0. 基线侦察 (2026-06-07)

### 0.1 两套代码栈的对接面 (已读源码确认)

| 维度 | FLASH (`realtime-vla-flash/`) | kai0 (`deepdive_kai0/kai0/`) | 结论 |
|---|---|---|---|
| 基座 | openpi/**pi0** (LIBERO) | openpi/**pi0 + pi05** | 同源, 对接面一致 |
| `embed_prefix()` | `(embs,pad,att)` shape (B,S,H) | **签名完全相同** (`pi0_pytorch.py:186`) | draft 可直接挂 |
| KV cache | prefill 后复用 | `sample_actions` prefill 后复用 (`pi0_pytorch.py:393`) | spec verify 可复用 |
| action 维度 | 7 (xyz+rpy+1grip, **单夹爪 idx6**) | **14** (双臂 joint, **双夹爪 idx 6,13**) | ⚠️ 主要适配点 |
| action_horizon | 50 | 50 | 一致 |
| 状态注入 | pi0: state_proj 单独 token (32D) | pi05: 状态进 language prefix (离散化); pi0: state_proj | draft state token 需参数化 |
| GemmaDecoderLayer | 装 transformers 自带 | `.venv` 自带 4.53.2 (无 adarms, 但有 `**kwargs`) | draft 用 stock 层即可 (adarms_cond=None 被 kwargs 吞) |

### 0.2 环境确认 (`kai0/.venv`, Python 3.12.3)
- `torch 2.7.1+cu126`, `transformers 4.53.2`, CUDA 可用 ✅
- `from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer, GemmaRotaryEmbedding` ✅
- stock `GemmaDecoderLayer.forward` 参数: `hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position, position_embeddings, **kwargs` — **无 `adarms_cond`** → draft 永远传 `adarms_cond=None`, 被 `**kwargs` 安全吞掉。
- ⚠️ `transformers_replace` (adarms 补丁) **未 copy 进 `.venv`** (`grep -c adarms` = 0)。即 `.venv` **当前不能跑完整 pi05-pytorch 主模型** (`pi0_pytorch.py:121` 的 check 会 raise); 生产 V1 pytorch 推理走独立 venv (`.venv_5090_trt`)。
  - **对本增量无影响**: draft head 只依赖 stock GemmaDecoderLayer, CPU 形状测试在 `.venv` 即可跑通。
  - 完整 `SpecPI0Pytorch`(加载真 ckpt)需在带补丁的 venv — 留待 R1.4。

### 0.3 关键 FLASH 源码定位 (移植蓝本)
- `src/openpi/models_pytorch/draft.py:8-160` — `DraftChunkHead` (单层 Gemma query decoder)
- `src/openpi/models_pytorch/spec_pi0_pytorch.py:50-178` — 4 个纯函数: `_truncate_accepted_prefix_on_gripper_switch` / `_detect_verify_gripper_switch_any_k` / `_compute_radius_prefix_acceptance` / `_stitch_radius_prefix_output`
- 同文件 `:279-309` — `SpecArgs` dataclass
- 同文件 `:799-997` — `_sample_actions_impl` 投机主循环 (R1.4 移植)
- 夹爪 idx **硬编码为 6** (LIBERO 单臂): `:75,79,114,118` + radius 里 `if d>=7: eval_d=min(eval_d,6)` (`:144`) — **kai0 双臂需泛化**。

### 0.4 本轮决策
1. **文件命名镜像 FLASH** (`draft.py` / `spec_pi0_pytorch.py`), 便于未来与上游 diff/merge; kai0 无同名文件, 纯新增。
2. **拆分增量**: 先交付可在 CPU 独立测试的 (a) draft head + (b) 纯接受函数, 再(R1.4)交付需 GPU+ckpt 的 `SpecPI0Pytorch` 子类。每个增量都跑通验证再往下。
3. **双夹爪泛化**: 把 FLASH 写死的 `idx=6` 改为 `gripper_dims: tuple[int,...]`(kai0 传 `(6,13)`); radius 距离改为"排除夹爪维"而非 LIBERO 的"取前 6 维"。这同时是 R2(夹爪相位强验证)的底座。
4. **不碰 `.venv` 的 transformers**: 不 copy 补丁, 不动现有环境。

---

## 1. 增量 R1-a: draft head + 双夹爪接受逻辑 (2026-06-07) ✅

### 1.1 新增文件 (纯新增, 0 改动旧代码)
| 文件 | 作用 | LoC |
|---|---|---|
| `kai0/src/openpi/models_pytorch/draft.py` | `DraftChunkHead` (kai0 适配版) | ~215 |
| `kai0/src/openpi/models_pytorch/spec_pi0_pytorch.py` | 4 个接受/夹爪/拼接纯函数 + `SpecArgs` (双夹爪泛化) | ~280 |
| `train_scripts/kai/eval/spec_draft_offline_test.py` | CPU 无 ckpt 单测 | ~190 |

### 1.2 关键适配 (vs 上游 FLASH)
- **`DraftChunkHead`**: `out_dim` 7→14 (双臂 joint); `state_dim` 从硬编码 32 改为参数 (kai0 raw state 14D); 新增 `use_state_token` 开关 (pi05 状态进 language prefix, 可关掉显式 state token)。其余逐行镜像 FLASH, 便于上游 merge。
- **双夹爪泛化** (核心): FLASH 夹爪 idx 写死 6 (LIBERO 单臂)。改为 `gripper_dims: tuple = (6,13)`:
  - `_detect_verify_gripper_switch_any_k` / `_truncate_accepted_prefix_on_gripper_switch`: 对每个夹爪维分别算开/合穿越, **OR 合并**, 截断取**最早**切换点。
  - `_compute_radius_prefix_acceptance`: radius 距离改为"**排除夹爪维**"(对 12 个臂关节算), 取代 LIBERO "取前 6 维" 的 hack。`gripper_prev` 支持 `(B,G)` 或 `(B,)` 广播。
  - 这套双臂相位门控同时是 **R2 (夹爪相位强验证)** 的底座。

### 1.3 验证 (CPU, `kai0/.venv`, `CUDA_VISIBLE_DEVICES=""`)
`kai0/.venv/bin/python train_scripts/kai/eval/spec_draft_offline_test.py` → **16 passed, 0 failed**:
- draft forward 输出 `(B,50,14)` (带/不带 state token); 错误 state 维度正确 raise。
- radius: draft==verify 全接受; 臂维 step6 扰动→接受 6; **仅夹爪维扰动被 radius 忽略** (符合设计); min-over-K 取更严成员。
- 双夹爪: L 在 step5 开 + R 在 step8 开 → 检测到 + 截断到**最早 step5**; 全程开/无切换→不截断; 单臂 R step3 切换→截断到 3。
- stitch: 前缀取 draft / 尾段取 verified; 0 接受→全 verified。
- `ruff check` 两个 src 文件全通过。

### 1.4 旧代码无影响核验
- 两个 src 文件**无人 import** (新文件, 旧路径不引用); `spec_pi0_pytorch.py` 暂未 import `PI0Pytorch` (纯函数模块), 故不触发 `.venv` 缺 transformers 补丁的 check。
- 测试在 stock `.venv` 跑通, 未 copy 任何补丁、未改环境。

### 1.5 下一步 (R1.4, 待 GPU+带补丁 venv)
`SpecPI0Pytorch(PI0Pytorch)` 子类: 复用 `embed_prefix`+KV cache 做 draft, 复用 Action Expert `denoise_step` 做 K-way verify, 串起 draft→verify→accept→截断→stitch→full fallback 状态机, 适配 pi05 adaRMS。需在带 `transformers_replace` 补丁的 venv + 真 pi05 ckpt 上验证 draft forward 与全量输出的半径分布。

---

## 2. 增量 R1-b(seam): draft 挂载真模型 + 延迟实测 (2026-06-08) ✅

在写完整投机状态机前, 先做**接缝去风险**: 验证 draft head 能挂到真 pi05 ckpt 的真实 prefix 上、能从 VLM layer0 warm-start、并实测 draft 延迟。

### 2.1 新增文件
| 文件 | 作用 |
|---|---|
| `train_scripts/kai/eval/spec_draft_attach_probe.py` | 加载真 pi05 ckpt → hook `embed_prefix` 抓真实 prefix → 建 draft(VLM config)→ warm-start → 实测 draft vs full 延迟 |

### 2.2 环境/坑
- venv: `kai0/.venv_5090` (py3.12, **带 adarms 补丁**, count=18); GPU3 空闲; ckpt `pytorch_pure200_step50000` (config `pi05_pytorch_a_new_pure_200`, asset `a_new_pure_200`)。
- **坑1**: `policy.infer()` 内部 `sample_actions` 是 `torch.compile(max-autotune)`+cudagraph, 在该 venv/5090 上崩 `RuntimeError: invalid dtype for bias`(SDPA bias dtype)。**这是模型自身编译路径的问题, 与 draft 无关**。
- **坑2**: 即便不崩, monkeypatch `model.embed_prefix` 在 compiled graph 里也不会触发。
- **解法**: 绕开 compiled wrapper, 直接调**eager 类方法** `type(model).sample_actions(model, ...)`; observation 走正常 transform 构造 (复刻 `policy.infer` 前半段)。eager 路径下 hook 正常触发、不崩。

### 2.3 实测结果 (GPU3, bf16, 10 iters)
```
[model]  pi05=True  action_horizon=50  action_dim=32  device=cuda
[prefix] embs=(1, 968, 2048) bf16   (768 图像 patch + 200 lang token)
[full]   actions=(50,32)  P50=263.2ms   (eager 10-step 去噪)
[draft]  hidden=2048 heads=8 kv=1 head_dim=256  warm_start=OK
[draft]  out=(1,50,32) finite=True  P50=2.51ms
[speed]  draft 比 eager-full 快 105x (单次, 未训练)
```
- **接缝成立**: draft head 吃真实 prefix(968×2048)产出整条 chunk, 数值 finite。
- **warm-start 成立**: `init_from_vlm_layer` 的 `load_state_dict(strict=True)` 通过 → `gemma_config` 复用 VLM 配置使形状对齐 (验证了 draft.py 新增的 config override)。
- **延迟**: draft 2.51ms。注: 对照应是**compiled** full(V1 P50≈32ms), 故真实加速 ≈13x; eager-to-eager 105x 是上界参考。draft 输出未训练故数值无意义(`|mean|`30 vs full 0.9), 训练在 R1-c。

### 2.4 关键结论: 投机在 **32-D padded 空间** 进行
pi05 主模型原生 `action_dim=32`(前 14 = 真双臂动作, 后 18 = padding)。draft 与 verify 都在 32-D 出。**夹爪仍在 (6,13)**, 臂关节 (0-5,7-12) 不变; `SpecArgs.dist_dims=12` 的 cap 会自动取前 12 个非夹爪维 = 12 个臂关节, **天然排除夹爪 + padding**。⇒ R1-a 的 spec 模块在 32-D 空间**无需改动**, 只要 `out_dim=32`、`dist_dims=12`。部署 14-D 是 `action_out_proj`/output transform 之后的投影, 与投机层解耦。

### 2.5 旧代码无影响核验
- 仅新增 1 个 probe 脚本; 未改任何模型/服务/ROS2 文件; 未改 venv。
- 探针只读模型 (eval, no_grad), 跑完即恢复 `embed_prefix`。
- 顺带发现的 compiled-path SDPA 崩溃是**既有现象** (非本次引入), 已记录, 留作 R1-b 完整实现时的注意项 (投机 verify 也要走 eager 或修复 compile)。

---

## 3. 增量 R1-b(full): `SpeculativeSampler` 完整状态机 + 机制验证 (2026-06-08) ✅

### 3.1 设计决策: wrapper 而非 subclass
不做 `SpecPI0Pytorch(PI0Pytorch)` 子类(会重建 7GB 模型 + 重新触发 `torch.compile`)。改为**附加 wrapper** `SpeculativeSampler(model, draft, spec_args)`(在 `spec_pi0_pytorch.py` 末尾), 持已加载模型 + draft head 引用, 通过调用模型既有方法 (`embed_prefix`/`denoise_step`/`paligemma_with_expert.forward`) 驱动投机。**零改动 `PI0Pytorch`**。

### 3.2 一轮投机流程 (`sample()`, 全 eager)
1. **prefill**: 复刻 `sample_actions` 前缀缓存阶段, 出 `past_key_values` (一次)。
2. **draft**: draft head 出整条 chunk → 铺进模型 32-D 空间 `x0_draft`。
3. **verify-from-draft** (K 路, **顺序**): 对每个 `t∈t_list`, 建 `x_t=t·noise+(1-t)·x0_draft`, 一次 `denoise_step` → `v_t`, **`x0_hat=x_t-t·v_t`**。顺序 K(=2)而非 batch B*K, 避开 HF KV-cache batch 扩展的脆弱性。
4. **accept**: `_compute_radius_prefix_acceptance` (dist_dims=12, 排除夹爪+padding); `x0_tail=mean_K(x0_hat)`; `_stitch_radius_prefix_output`。
5. **夹爪相位门控** (双臂): verify 阶段任一 verify member 预测夹爪切换 → 该样本 accepted=0 (`_detect_verify_gripper_switch_any_k`); post-verify 在拼接结果上按最早切换截断 (`_truncate_...`)。
6. **full fallback**: accepted≤0 或夹爪切割 → 跑 `_full_denoise` (parent 等价 10-step eager 去噪) 出干净 chunk。
7. 返回 dict: `actions` + **投机信号** `accepted_prefix_len / radius_dist / gripper_verify_stop / gripper_switch_cut / used_full_fallback / draft_ms / verify_ms` (R3/R5 直接消费)。
- 新增 `x0_draft_override` 参数: 供"oracle draft"机制验证(未蒸馏前)。

### 3.3 机制验证 (GPU3, `spec_sampler_mechanics_probe.py`, τ=0.3)
```
[model] pi05=True H=50 action_dim=32
A. 未训练 draft : radius_dist=3.039  accepted=0/50   used_full_fallback=True   actions=(1,50,32) finite ✅
B. oracle draft : radius_dist=0.0121 accepted=50/50  used_full_fallback=False  gripper_stop/cut=False ✅
==== A(reject+fallback)=OK  B(oracle-accept)=OK ====
```
- **拒收+回退路径**: 垃圾 draft (radius 3.0 ≫ τ0.3) → 全拒 → fallback 出 finite chunk。✅
- **接受+拼接路径**: draft==全量去噪结果 (radius 0.012 ≪ τ0.3) → 全接受 50/50 → 不回退。✅ 同时证明 `x0_hat=x_t-t·v_t` 重构忠实于模型真 x0。
- **判别力**: 好/坏 draft 的 radius 差 250× (0.012 vs 3.039), 接受准则干净二分。
- ⇒ **整条投机状态机在真 pi05 上端到端跑通**; 唯一未验证的是"真实 draft 的接受率", 需 R1-c 蒸馏。

### 3.4 旧代码无影响核验
- `spec_pi0_pytorch.py` 新增 `SpeculativeSampler` 用**惰性 import** (`pi0_pytorch`/`time` 在方法内), 纯函数 CPU 单测 (stock `.venv`) 仍 16/16 通过, 不触发缺补丁的 check。
- `draft.py` / `spec_pi0_pytorch.py` `ruff check` 全通过; 无任何现有文件改动。

### 3.5 下一步 R1-c (蒸馏真 draft, 需 GPU + 数据 + 训练时长)
`enc_cache.py`(冻结 backbone, 滑窗采样 dump prefix-embedding cache)→ `spec_draft_train.py`(回归蒸 DraftChunkHead 到 pi05 全量 chunk)。产出真 draft 后即可: (a) 测 kai0 val 真实接受率/半径分布 (R1-d), (b) 跑 R5 好/坏 ckpt 接受率↔开环 SNR 对照。

---

## 4. 增量 R1-c: 蒸馏真 draft (smoke) (2026-06-08) ✅

### 4.1 决策: kai0-native 单脚本蒸馏 (不照搬 FLASH 分片 cache 机制)
FLASH 的 `enc_cache.py`+`spec_draft_train.py` 有 sharded safetensors cache + manifest, 为规模化设计。对 kai0 首个 draft (百~千帧) 过重。改为单文件 `train_scripts/kai/eval/spec_draft_distill.py`: 同样的数学 (冻结 backbone, prefix→teacher chunk 回归), 但 prefix cache 直接放 CPU RAM。**ADDITIVE: 只训新 draft head, 不动主模型, draft 存独立 .pt。**

### 4.2 流程
1. **CACHE**: 采样 val 帧 → 真实 obs → `embed_prefix` (CPU 存 prefix) + **teacher target = 模型自身全量去噪 chunk x0** (固定 noise; 让 draft 学 Action Expert 输出 → 利于接受)。
2. **TRAIN**: `DraftChunkHead(prefix)→teacher` step-weighted Huber; 冻结 backbone; VLM-layer0 warm-start; Adam + **grad-clip(1.0) + cosine LR + 存 best-not-last**。
3. **EVAL**: holdout 帧上算真实 `accepted_prefix_len` + radius(draft vs teacher)。

### 4.3 坑: 训练发散 → 存到坏权重
首跑 (lr1e-3, 无 clip/decay/best): huber 到 epoch150 = **0.0064** 收敛漂亮, 但 epoch175 **突刺到 1.08** (Adam 失稳), 又存了**最后一轮(发散)权重** → holdout radius 9.5, 接受 0。**修复**: grad-clip + cosine LR + 存 best。

### 4.4 结果 (GPU3, pure200 pi05, 96 训练帧 / 32 holdout 帧, 300 epoch)
```
huber: 0.519(e0) → 0.0044(e50) → 0.00088(e175) → 0.00051(best, 平滑无发散)
holdout: mean accepted_prefix_len = 27.9/50  (median 39, max 50)
         mean radius(draft vs teacher) = 0.285  (tau=0.3)
         zero-accept frames = 11/32
=> draft USEFUL: 仅 96 帧蒸馏, holdout 平均已接受 ~28/50 步
```
- **R1-c 跑通**: 蒸出的 draft 在 holdout 上有**真实、可观的投机接受** (28/50)。
- **数据受限信号**: radius 均值 0.285 紧贴 τ0.3, 11/32 帧仍 0 接受 → 仅 96 训练帧, 扩到全 base 数据集 (千帧) 接受率会显著上升 (R1-d)。
- **诚实 caveat**: 此处接受率以 **draft-vs-teacher(缓存全量去噪)** 的 radius 度量 (忠实代理); 部署真实接受用 verify-from-draft 的 `x0_hat` (近终去噪), 二者应接近但未经完整 `SpeculativeSampler` holdout 实测 — 留 R1-d。
- 产物: `/tmp/draft_pure200.pt` (smoke artifact, 临时)。

### 4.5 旧代码无影响核验
- 仅新增 `spec_draft_distill.py`; 训练只动新 draft head; 主模型 eval/no_grad 只读; draft 存独立文件。CPU 单测仍 16/16。

---

## 5. R1 阶段小结 (2026-06-08)
**FLASH 投机推理已在 kai0 pi05 上端到端打通** (移植 → 接缝 → 状态机 → 蒸馏), **全程只新增文件, 0 改动旧推理路径**:
| 文件 (全新增) | 验证 |
|---|---|
| `kai0/src/openpi/models_pytorch/draft.py` | CPU 形状 + GPU 真 prefix 挂载 (2.5ms) + warm-start |
| `kai0/src/openpi/models_pytorch/spec_pi0_pytorch.py` (helpers + `SpeculativeSampler`) | CPU 16/16; GPU 机制 (拒收+回退 / 接受+拼接) |
| `train_scripts/kai/eval/spec_draft_offline_test.py` | 16/16 |
| `train_scripts/kai/eval/spec_draft_attach_probe.py` | draft 2.5ms vs full 263ms |
| `train_scripts/kai/eval/spec_sampler_mechanics_probe.py` | A/B 机制 OK |
| `train_scripts/kai/eval/spec_draft_distill.py` | holdout 接受 27.9/50 |

**下一阶段**: R1-d (扩 base 数据集重蒸 → 真实接受率基线 + 用 `SpeculativeSampler` 完整 holdout 实测) → R5 (好/坏 ckpt 接受率 ↔ 离线视觉消融 SNR 对照, 把接受率做成开环在线指纹) → R2/R3。

> 未提交 (git): 全部为新增研究文件 + 文档, 按既有"部署改动经真机验证再提交"的稳健姿态, 留待用户审阅后决定是否 commit。

---

## 6. 增量 R1-d: 扩数据重蒸 + 真实 verify-from-draft 接受率基线 (2026-06-09) ✅

补齐 R1-c 留下的三个口子, 仍**全程新增文件 + 0 改动旧推理路径** (仅重构我自己的新文件 `spec_pi0_pytorch.py`):

### 6.1 三个修正 (vs R1-c)
| 口子 (R1-c) | R1-d 修正 |
|---|---|
| teacher 用**固定随机噪声** full-denoise | teacher = **零噪声** full-denoise (照 FLASH `enc_cache.py` 的 `zero_noise` 约定; 确定性目标, 正是 verify 检的不动点) |
| prefix cache 全塞 **CPU RAM** (~百帧封顶) | **磁盘分片** safetensors + `manifest.json` (FLASH 风格); 训练时逐 shard 流式加载, RAM 只占 1 shard |
| eval 用 **draft-vs-teacher 代理** radius | eval 走**完整 `sample_from_prefix`**: draft→K-way denoise verify→`x0_hat = x_t − t·v_t`→radius→夹爪门控→fallback, 即**生产路径真实信号** |

### 6.2 sampler 重构 (additive, 仅本人新文件)
`_prefill` 拆 → `_embed_prefix`(obs→prefix 张量) + `_prefill_kv`(prefix→VLM KV); 抽出 `_spec_core`; 新增 **`sample_from_prefix()`** —— 从磁盘缓存的 prefix 张量重建 KV 跑投机, 无需重跑视觉编码 / 重解码视频。`sample()` 行为零变化 (CPU 离线 16/16 仍过)。

### 6.3 数据现实
Task_A 下**只有 `A_new_pure_200_val` (20 eps × 1k–3k 帧) 本地有可解码视频** (advantage/base 目录无本地 mp4)。故 R1-d 靠**密采样**这 20 eps 扩规模: 16 train eps × 100 帧 = **1600 训练帧** (R1-c 的 ~17×), 4 holdout eps × 60 帧 = **240 帧**, **episode-disjoint 无帧泄漏**, 300 epoch。

### 6.4 结果 (GPU3, pure200 pi05 step50000, τ=0.3, verify 随机噪声)
```
train_huber best=0.00250 (1600 帧, 干净收敛无发散)
holdout 真实 accepted_prefix_len: mean=50.0/50  median=50  p25=p75=50  max=50
zero-accept=0/240   full-fallback=0/240   gripper-verify-stop=0/240
mean min-over-K radius (eval window) = 0.0181  (τ=0.3, 余量 ~16×)
```
**240/240 帧整段 50/50 接受, 0 回退** —— 命中接受率天花板, radius 0.018 已逼近 oracle (机制探针 B 的 0.012)。

### 6.5 证伪对照 (关键: 排除"路径恒接受"bug)
50/50 太完美, 故跑 `spec_draft_r1d_control.py`: **同一 cache、同一 `sample_from_prefix`** 喂**未训练** draft:
| draft | mean accept | radius | fallback |
|---|---|---|---|
| 未训练 (仅 VLM-layer0 warm-start) | **0.0/50** | 1.96 | 40/40 |
| R1-d 训练后 | **50.0/50** | 0.018 | 0/40 |
→ verify 路径**确实判别** (垃圾 draft 全拒+全回退), 50/50 是真信号非 bug。余量极大 (0.018 vs 1.96), 对 τ 不敏感 (τ=0.05 仍同结论)。

### 6.6 为何 R1-d (50/50) >> R1-c (27.9/50)
(a) 17× 数据; (b) 零噪声 teacher = verify 检的确定性不动点, draft 学到该流形 → x0_hat≈x0_draft → radius→0; (c) R1-c 的代理 radius 用随机噪声 teacher, 本身更糙, 高估了差距。

### 6.7 ⚠️ 诚实边界 (写进 R5 前必读)
- **离线 / 单帧 / 同任务同机位** holdout, **非闭环 rollout**。闭环时场景漂出训练流形, 接受率必降、fallback 必起 —— 50/50 是**离线天花板, 不等于上机 50/50**。
- **0 回退 + 全接受 ⇒ 接受率在此 ckpt 上"饱和", 单看接受率分不出好 ckpt vs 开环 ckpt** —— 这恰好**坐实 R5 论点**: 必须看 **接受率 × 视觉消融 SNR 联合分布**, 接受率单独不构成开环门禁。pure200 是否偏开环 (参 0.99Hz 振荡 / SNR 门禁历史) 直接影响这个 50/50 的解读。

### 6.8 旧代码无影响核验
`spec_pi0_pytorch.py` 重构后 CPU 离线 16/16 仍过; `sample()` 签名/行为不变; 新增 `sample_from_prefix` 只被 R1-d eval 调用。`draft.py` 未动。新增文件: `spec_draft_r1d.py` (cache+train+eval) + `spec_draft_r1d_control.py` (证伪)。磁盘 cache 落 `/data1/tmp/spec_cache_r1d_pure200` (~6GB, 可 `--reuse-cache` 复用给 R5)。

**下一步**: R5 —— 取一个已知偏开环的 ckpt (或对 pure200 做视觉消融) 重跑 R1-d eval, 验证"接受率 × SNR"是否能把开环 ckpt 从好 ckpt 里分出来。

---

## 7. R1 部署落地: FLASH server + v2 一键启动 (2026-06-09) ✅

把离线验证过的 FLASH 投机推理接到真机部署栈。**严格 server-only / 不改任何旧代码**, 与 XVLA (`serve_policy_xvla.py`) / V1 (`start_serve_v1.sh`) 同款思路。

### 7.1 集成 seam (为何零侵入)
`Policy.infer` (`policy.py:119`) 的唯一推理调用是 `self._sample_actions(device, observation, **kw) -> [B,H,action_dim]` (normalized model 空间)。而 `SpeculativeSampler.sample(observation)` 正好返回 `{"actions": (B,H,action_dim), ...信号}` —— **同一份契约**。故集成只需:
1. `create_trained_policy(...)` 照常构出标准 PyTorch pi05 `Policy` (transforms / norm_stats / model 全部原样)；
2. 在 `policy._model` 上建 `SpeculativeSampler` + 加载逐-ckpt 蒸馏的 `DraftChunkHead`；
3. **只替换 `policy._sample_actions`** 一个属性为投机 shim, 其余 (输入 repack/normalize、输出 unnormalize+14D 切片) 完全复用标准管线。

对外仍 emit 标准 `action_kind="joint"` 14D → 现有 `policy_inference_node --mode websocket` 客户端**零改动**接入。

### 7.2 新增文件 (全部 additive)
- `kai0/scripts/serve_policy_flash.py` —— FLASH server (`.venv_5090` PyTorch)。`--config/--dir/--asset-id/--draft/--port(默认8001)/--tau/--seed/--no-fallback`。draft blob 自描述 (`img_dim/chunk_m/out_dim`), 与 ckpt backbone 不配对时 `_build_draft` **大声报错** (hidden_size 不等)。
- `start_scripts/kai/start_autonomy_from_ckpt_v2.sh` —— 一键: 校验 PyTorch ckpt → 定位 draft (`--draft` 或 `<ckpt>/draft_head.pt`) → 选 GPU → 后台拉 FLASH server → 等端口就绪 → `start_autonomy.sh --mode websocket --ws-port`。`trap` 退出清理 server。
- `spec_pi0_pytorch.py` 新增 `full_denoise_from_observation()` (eager 全量 denoise)。

### 7.3 关键坑: 安全 fallback **不能**用模型的 compiled 路径
首版 shim 的 catastrophic fallback 写成 `orig = model.sample_actions`。冒烟实测在 5090 上**该 compiled 路径直接 CUDA-graph 崩** (`RuntimeError: invalid dtype for bias`, `_scaled_dot_product_efficient_attention` + cudagraph_trees) —— 正是 §2.2 记录的 5090 torch.compile 病。即"安全网"本身会崩。修正: 新增 eager 的 `full_denoise_from_observation()`, shim 异常时走它 (与 `SpecArgs.full_fallback` 的 eager `_full_denoise` 同源, 5090-safe)。

### 7.4 冒烟核验 (GPU3, pure200 step50000 + R1-d draft, 真 `Policy.infer` 全管线)
- **FLASH 路径**: `policy.infer(obs)` → 输出 **shape (50,14)** (经标准 unnormalize+切片); 信号 accept=**50.0/50**, fallback=False, radius=**0.0171** (与 R1-d 离线 0.018 一致), draft=16.3ms + verify=102.1ms (K=2 步) vs full-10-步 denoise ≈510ms → 实测提速。
- **Fallback 路径**: 强制 `spec.sample` 抛错 → eager `full_denoise_from_observation` 仍返回合法 **(50,14)**, 不崩 → 安全网验证通过。
- 两文件 + spec 改动 ruff All-checks-passed; `v2.sh` `bash -n` OK; GPU3 用后回基线。

### 7.5 旧代码无影响核验
未改: `policy.py` / `policy_config.py` / `policy_inference_node.py` / `start_autonomy.sh` / 任何 ROS2 / 训练栈。`spec_pi0_pytorch.py` 仅**追加** `full_denoise_from_observation` (旧 `sample`/`_spec_core`/CPU 16 单测路径不动)。集成靠运行时替换 `policy._sample_actions` 属性, 不 monkeypatch 类。JAX/V1/XVLA 服务端口 (8000/8002/8003) 与 flash (8001) 不冲突。

### 7.6 vs v1 (Triton) pipeline 对比 (别误读 v2=更快)
两者**同构** (都 server-only websocket, 都不改 ROS2/训练栈), 但加速轴正交:

| 维度 | v1 (Triton) | v2 (FLASH) |
|---|---|---|
| 加速思路 | kernel 级: TRT 跑*同套* 10-step denoise | 算法级: draft 猜整条 + K=2 步 verify 替 10 步 |
| 无损? | ✅ lossless | ⚠️ lossy (radius 启发式, fallback 兜底) |
| 源 ckpt | JAX (orbax) → `v1_p200.pkl` 转换 | PyTorch (safetensors) → `draft_head.pt` 逐-ckpt 蒸馏 |
| 后端 | TRT kernel (~3-4ms/步), `.venv_5090_trt`, :8002, SHM+ws | **eager** PyTorch (compiled 5090 崩), `.venv_5090`, :8001, ws |
| 延迟 | server infer **P50≈32ms / 8×** | prefill+draft16+verify102 ≈ **120-150ms+** (eager 税) |
| 副产物 | 无 | **accept/radius 信号** → R5 开环探针 / R3 触发器 |

**关键**: v2 当前走 eager, 每步 denoise ~51ms ≫ v1 TRT 每步 → 少跑步数省的被 eager 吃回, **wall-clock v2 未必胜 v1**。v2 现价值 = (1) 免 TRT 转换直吃 PyTorch ckpt; (2) 接受率/半径信号 (v1 给不了); (3) FLASH 研究载体。两者正交可叠加: v1 压每步成本、v2 压步数, 终态 **R6** 把 FLASH 投机挂到 Triton kernel (draft + K=2 步 TRT verify) 才是组合拳。选型: 要稳要快上真机干活→仍用 v1; 要 PyTorch 免转换 / 开环探针 / 推 FLASH 研究→v2。

### 7.7 诚实边界 / 上机前必读
- 冒烟用**随机合成图**, 只证"形状对+管线通+不崩+有提速", **非真机行为正确性**。draft 是 pure200 专属, 换 ckpt 必须重蒸 (`spec_draft_r1d.py --out` → 拷成 `<ckpt>/draft_head.pt`)。
- FLASH 是 **lossy** 投机; 闭环漂移会拉低接受率→起 fallback (慢但不错)。**首次上机务必先看 server 日志 `mean_accept / fallback%`, 并与 v0 A/B 对比 EE 抖动**, 通过后才 commit (部署红线: 真机验证前不提交)。draft `/tmp/draft_r1d_pure200.pt` 为 ephemeral, 长期部署需固化到 ckpt 目录。

---

## 8. R5: 接受率/radius 能否当"免费视觉敏感度探针"? —— **证伪 + 机理** (2026-06-09)

`spec_r5_probe.py` (additive, 复用 R1-d sampler+draft, 不改旧码)。每帧两条件 real vs **全相机置黑**, 解耦测两路: 模型 vision-SNR (从 eager teacher `_full_denoise` 算, = `eval_vision_ablation` 同义) + FLASH 免费信号 (accept/radius)。

### 8.1 结果 (GPU3, pure200 step50000, 160 帧 holdout, real vs all-black)
| 量 | real | black | Δ / 相关 |
|---|---|---|---|
| 模型 vision-SNR (Δblack/floor, 臂) | — | — | **mean 7.36× / median 6.53×** (p10 3.76, p90 12.5) |
| FLASH accept | 50.0/50 | 50.0/50 | **Δ = 0.00 (完全不动)** |
| FLASH radius | 0.0174 | 0.0150 | Δ = −0.0024; corr(SNR, Δradius)=−0.09, corr(SNR, radius_black)=+0.05 |

### 8.2 结论: **接受率/radius 不是开环探针 (证伪), 且能解释为什么**
- pure200 **确实在用视觉** (臂 SNR 7.4× ≫ 1) —— 不是开环 ckpt, 对比有效。
- 但**置黑全相机 (模型输出位移 7.4× SNR), FLASH 接受率纹丝不动 (Δ=0.00), radius 也几乎不动** (corr≈0)。
- **机理**: verify-from-draft 接受率是**自洽 (self-consistency) 量** —— 测的是"draft 与模型**在同一输入上**的去噪是否一致", 而非"该输入是否有信息"。置黑后 draft 与 verify **同时**吃到置黑 prefix, 两者照样彼此一致 → 接受率对输入退化**结构性失明**。接受率是 *consistency* 指标, 不是 *information* 指标。

### 8.3 对 R5 的修正 (重要)
- 原 R5 设想"接受率可能是免费开环代理" → **被证伪**。
- R5 真正论点被**坐实并锐化**: `接受率 × SNR` 必须是**真乘积**, SNR 项要**外接、独立** (周期性视觉消融探针), 不能指望从接受率/radius 里"白捡"出来。两轴正交: 接受率 = "投机捷径可不可信 / 模型自不自洽"; SNR = "模型用不用视觉"。
- **部署落地**: 开环门禁仍走既有 vision-ablation SNR ([[reference_vision_ablation_openloop]]); FLASH server 若要出"接受率×SNR"联合健康度, 需在 server 内**周期性插一帧置黑前向**算 SNR (廉价, 每 N 帧一次), 与每帧接受率联合 —— 列为 R5-followup。
- ⚠️ 边界: 这是**全相机置黑**的极端消融 + 单 ckpt; "好 ckpt vs 已知开环 ckpt 同输入下接受率是否都饱和"仍需第二个 (开环) ckpt + 其 draft 来直接验 (R1-d 已证同任务接受率饱和, 此处补上"对输入退化也失明"这一面)。

### 8.4 旧代码无影响核验
仅新增 `spec_r5_probe.py`; 复用 `spec_pi0_pytorch.SpeculativeSampler` + R1-d draft; 未改任何旧码/训练栈。

### 8.5 R5-followup ①: server 内"接受率×SNR"健康探针 (落地, 2026-06-09) ✅
把 §8.3 的修正变成 server 里可用的在线信号。`serve_policy_flash.py` 新增 `_FlashHealthPolicy` 包装 (opt-in `--health-probe-every N`, **默认 0=关 → 与裸 v2 逐字一致**):
- 每 N 次 infer, 对**同一帧**做一次 raw-image-blacked 重前向 (faithful: 在 transforms **之前**置黑, 故包 `Policy.infer` 而非 `_sample_actions` seam), 算 vision-SNR = Δblack/floor (3 次 eager teacher denoise), 与该帧接受率联合日志 `accept x SNR`, 并挂进 response 的 `flash_health`。
- 设计要点: SNR 项**外接独立** (§8.3 铁律, 不从接受率白捡); inline 在 probe 帧 (~3 次 denoise) → N 取大 (默认建议 60), 靠 node chunk buffer 吸收慢帧; 探针异常自动**关掉自己**, 永不影响出动作。
- 冒烟 (GPU3, pure200, probe_every=2, **合成随机图**): probe 帧正常触发, 日志 `accept=50.0/50 | vision-SNR=1.74x | joint=1.7`, 动作路径仍 (50,14)。SNR 1.74× 是因合成随机图 (置黑相对随机图位移小); 真机/真帧应回到 §8.1 的 ~7× 量级 —— 此处只验**管线通+不崩+SNR 外接生效**。

### 8.6 R5-followup ②: 第二个开环 ckpt 直验 — ⏸ 阻塞 (缺 PyTorch 开环 ckpt)
"好 ckpt vs 已知开环 ckpt 同输入下接受率是否都饱和"需要第二个**开环 PyTorch ckpt** + 为其蒸 draft。现状: 全仓库仅 `pytorch_pure200_step50000` 一个 PyTorch (safetensors) pi05 ckpt; FLASH sampler 仅支持 PyTorch。已知开环 ckpt (如 5day_recent, 见 [[reference_vision_ablation_openloop]]) 均为 JAX → 需先 JAX→PyTorch 转换再蒸 draft, 属独立工程, **本轮不做**。机理上 §8.2 已预判: 接受率从不跨 ckpt、不对 GT 比较, 每个 ckpt 的 draft 自蒸馏只测自洽 → 两者都该饱和; ②只是把这条直接量出来。待有第二个 PyTorch 开环 ckpt 时复用 `spec_r5_probe.py` 即可。

---

## 9. 真机幅值欠驱动诊断 + draft v2 重训 (2026-06-18) ✅

> ⚠️ **记录恢复说明**: 6-09~6-13 期间在工作树里写过的 §9–§14 (R3 证伪 / R6 Triton-on-5090 量测 / v2 trace 工具 / 真机开环诊断 / draft v2 配方) 均为**未提交的 tracked-file 改动**, 在一次 `pull --rebase` + `reset` + merge (见 reflog HEAD@{0,2-5}) 中被丢弃; `draft.py` 的多层改动同时被回退。**未提交的 untracked 新脚本全部幸存**, 故那几轮分析的权威记录现存于各脚本 docstring: `spec_r3_probe.py` (R3 证伪), `optimize/v1_triton/bench_v1_per_step_5090.py` (R6: prefill-skip 下 FLASH+v1≈3.83x), `flash_trace_phase_report.py` (v2 真机 accept×相位), `spec_draft_magnitude_probe.py` (本节诊断)。本节把当前唯一可复现的活产物 (draft v2) 落档, 并已重新应用 `draft.py` 多层支持 (向后兼容, `num_layers` 默认 1)。

### 9.1 真机现象 → 离线根因
用户两次真机 rollout: `--no-spec` (纯 eager 10 步) 正常完成; **带 spec 动得慢、欠完成, 但整体轨迹趋势对**。接受率 50/50 查不出问题 —— 因为 verify-from-draft 锚在 draft 上 (x_t = t·noise + (1-t)·x0_draft), 衰减的 draft **自洽地被接受** (又一个 §8 自洽盲点)。

`spec_draft_magnitude_probe.py` (逐 chunk 比 spec vs full-denoise, arm 非夹爪维, MC=4 均值轨迹) 确诊 **draft v1 幅值欠驱动**:
- v1 (1 层 + Huber δ=0.1): **spec/full(均值)=0.61**, 趋势 cosine 0.52 → 1 层回归蒸馏退化为条件均值预测, 方向对幅值被压到 61%。

### 9.2 修复配方 (蒸馏脚本 `spec_draft_r1d.py` 新增, 全 opt-in)
`--num-layers 2`(从 1)、`--loss mse`(替 huber)、`--vel-weight 1.0`(逐帧速度差 MSE)、`--mag-weight 1.0`(逐维 std 匹配)、`--step-decay`。`draft.py` 配套加 `_gemma_blocks_tail` ModuleList + `init_from_vlm_layers()` 用 VLM 前 N 层 warm-start; 1 层路径键集 ⊂ 2 层键集, 旧 blob 照常加载 (CPU 冒烟核验通过)。

### 9.3 结果 (GPU1, pure200 step50000, 1600 train / 240 holdout, 300 epoch, train_loss 0.00282)
| 指标 | v1 (1L+Huber) | v2 (2L+MSE+vel+mag) |
|---|---|---|
| spec / full(均值) motion | **0.61** 欠驱动 | **1.16** |
| spec / full(单样本) motion | — | 1.13 |
| 趋势 cosine (vs 均值) | 0.52 | **0.71** |
| 逐 arm 维 spec/mean | ~0.6 | 1.0–1.5 (每维都回到位) |

采样自身 full/均值=1.02, 故 draft 1.16 处于健康区间 (略激进而非欠驱动)。幅值已非主因。产物 `/tmp/draft_r1d_pure200_v2.pt` (num_layers=2)。

### 9.4 下一步 (待用户真机验证)
用户用 `start_autonomy_from_ckpt_v2.sh <ckpt> --draft /tmp/draft_r1d_pure200_v2.pt` 重跑真机 (建议先不 `--execute` 看动作幅度, 再 `--execute`)。若仍偏慢, 候选: 提 `--mag-weight` / 加第 3 层 / spec 路径保留少量采样性 (verify 不锚 draft)。

### 9.4b ⚠️ 关键 loader bug: 多层 draft 被静默截断 (2026-06-18, 与 §9 同源回退连带)
用户首次用 v2 (2 层) draft 上真机 → **机械臂几乎不动**, chunk `Δ(0→49)=0.01` (输出≈保持当前姿态), infer 175ms。复盘发现 `serve_policy_flash.py::_build_draft` 也在那次 rebase 里被回退掉了 `num_layers` 支持: 它用 1 层 head + `load_state_dict(strict=False)` 加载 2 层 blob → 2 层 blob 的 `_gemma_blocks_tail.0.*` 全成 "unexpected" 被**静默丢弃**, 只装进 layer 0 → 半截 draft 出退化/近静止 chunk。这极可能就是真机不动的**真因** (而非 §9.4 推测的纯 OOD 欠驱动)。
修复 (`_build_draft`): 读 `num_layers=blob.get("num_layers",1)` 正确构造; 且 missing/unexpected 非空时**直接 SystemExit**, 拒绝服务半截 draft (宁可不起也不发垃圾动作)。下次起服会装全 2 层。

### 9.4c A/B trace 工具复原 + 增强 (诊断"真机仍欠驱动 vs loader bug")
`serve_policy_flash.py` 与 `start_autonomy_from_ckpt_v2.sh` 的 `--trace` / `--no-spec` 也随回退丢失, 已重新实现并加强:
- `--trace-out` 每帧写 JSONL: accept/fallback/radius **+ 新增 `arm_motion`** (臂维 max-min 之和, 与 §9 offline probe 同度量) + 夹爪 net/range。**`--no-spec` 下也记 arm_motion** → spec vs 纯 eager 基线可直接比真机 chunk 幅值 (offline in-dist probe 看不到 OOD 欠驱动, 真机 trace 能)。
- `--no-spec`: 跳过 draft+verify 走纯 eager `full_denoise_from_observation` (= 用户 --no-spec 真机路径)。
- `flash_trace_phase_report.py` 加 `_ab_compare`: 给多个 trace 文件 (spec + nospec) 时按文件列 arm_motion 中位数 + 相对比, 一眼看出 spec 是否 < nospec。
- 全部 opt-in (默认关 = 裸 v2 逐字一致); mock 冒烟通过 (spec 0.6 < nospec 1.1, A/B 比值与 verdict 正确); ruff + py_compile + bash -n 全绿。

### 9.6 真机 A/B (spec vs --no-spec) → 重新定性: draft OOD **过驱动 + 方向偏**, 非欠驱动 (2026-06-18)
9.4b loader 修复后 (`draft built: num_layers=2`, accept 50/50 fallback 0%), 真机现象变成 **spec 动不对 + 回退**; `--no-spec` 则**能抓衣服、大致跑完叠衣流程**(仍有些回退不流畅)。用 `--trace` 录两次, `flash_trace_phase_report.py <spec.jsonl> <nospec.jsonl>` (传多文件自动按文件 A/B) 对比 (归一化动作空间):

| | arm_motion (chunk 内 max-min 和) | 夹爪 chunk 间方向翻转 gl / gr |
|---|---|---|
| spec (draft) | **2.94** | 14% / 24% |
| no_spec (真模型) | **1.70** | 30% / 29% |

**三条结论**:
1. **draft 是 spec 失败主因** (no_spec 能做任务, spec 不能) ✅。
2. **机制与 §9.4 推测相反**: 不是 chunk 间翻转更多 (spec 翻转率反而**更低/持平**), 而是 **spec 过驱动 — arm_motion 是 no_spec 的 1.73×**。draft 在真机 OOD 帧上**动太猛**, 方向又 ~30% 偏 (离线 cosine 0.71)。**§9 加的 mag-weight=1.0 在 OOD 上过校正了** (in-dist 1.16x → OOD 1.73x)。
3. **"回退/不流畅"是另一个早存在的问题**: spec/no_spec 翻转率都 24-30%, 连真模型都这样 → 基模型 + RTC 的 chunk 间振荡 ("走3退1" / [[real_machine_oscillation_data_tail]] 的 0.99Hz), **与 FLASH 无关**。
- 读数提醒: 合并 trace 报告里 `fallback=33.5%` 是假象 (no_spec 行打 fb=1.0 标记, 70/209=33.5% 只是占比; spec 实 fallback=0)。

**根本张力**: accept 50/50 + radius 0.015 **只证 draft 自洽, 挡不住 OOD 过驱动 + 方向偏** (§13 盲点的闭环版)。光调幅值治不好 → 这是 draft **方向保真度 + OOD 泛化**问题。

### 9.7 draft v3 配方: cosine 方向损失 + 降 mag-weight (进行中, 2026-06-18)
`spec_draft_r1d.py` (也随同次 rebase 被回退, 已连同 v2 改动一并复原) 新增/改默认:
- **`--cos-weight 1.0` (新)**: demeaned 轨迹形状 `(1 - cosine(pred-mean, tgt-mean))`, 正是 magnitude probe 报的 "trend cosine" 度量 → 直接压方向误差 (0.71→目标≥0.9)。
- **`--mag-weight` 默认 1.0 → 0.3**: v2 的 1.0 在 OOD 过冲 1.73x, 降权让方向损失主导。
- 复原: `--num-layers 2`、`--loss mse`、`--vel-weight 1.0`、`--step-decay`、多层 `init_from_vlm_layers`、blob 存 `best_train_loss` + 全部 loss 元数据。
- 时序一致性 (相邻**帧**间 chunk 一致) 需重建 cache (现 cache 不保帧邻接), 本轮先不做; chunk **内**的 vel-weight 已覆盖帧内平滑。
- 验收 (与 §9 同): 离线 `spec_draft_magnitude_probe.py` 看 spec/full(均值) 回到 ~1.0 (不再 1.73 过冲) **且 cosine ≥0.9**; 达标才上机真机 A/B (arm_motion 应贴近 no_spec, 翻转率不恶化)。
- 训练: GPU0, `--reuse-cache` (1600/240), 300 epoch, `/tmp/draft_r1d_pure200_v3.pt`。

### 9.5 旧代码无影响核验
- `draft.py` 改动纯加性: `num_layers` 默认 1 → 单层路径与回退版逐字等价 (tail 为空 ModuleList, forward 循环退化为单 block); 旧 1 层 blob 键集是 2 层超集的子集, `load_state_dict` 照常。
- 蒸馏脚本所有新损失项 weight>0 才生效, 默认值即 v2 配方; 不触碰任何旧推理/训练路径。
- 全部新文件 untracked, 不进 git 直到用户明示。

---
</content>
