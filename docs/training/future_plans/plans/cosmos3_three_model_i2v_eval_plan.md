# Cosmos3 三模型 I2V 世界预测评测 plan

> **目的**: 在叠衣 fold 验证集上，用统一的 Image→Video 世界预测任务，横比 **Cosmos3-Nano / Cosmos3-Super / Cosmos3-Super-Image2Video** 三个模型的效果。
> **状态**: 📝 待评审（设计已定稿，P0 环境阻塞已解，待开跑）
> **创建**: 2026-06-05 ｜ **更新**: 2026-06-05（scope fix: 只 3 episode；horizon=1s/3s/7s **滑窗覆盖整段**，每窗 teacher-forced 独立 rollout）
> **资源**: 当前主机(=b2) 8×A100 80G + b1 8×A100 80G = **16×A100**
> **关联**: 复用 `tau0_wm/finetune/eval_gigaworld_dist.py` 的 GPU 度量；数字与 tau0 eval 可横比。

---

## 1. 背景与定位

三个已下载模型（`deepdive_kai0/cosmos/models/modelscope/`）均为 `cosmos3_omni` Diffusers 流水线（`Cosmos3OmniPipeline`）：

| 模型 | 规模 | transformer(bf16) | 定位 |
|---|---|---|---|
| Cosmos3-Nano | 16B | 29G | omni 基座（T2V/I2V 皆可，I2V 非专精） |
| Cosmos3-Super | 64B | 120G | omni 基座（更大，I2V 非专精） |
| Cosmos3-Super-Image2Video | 64B | 120G | **I2V 专精**（预期此任务最强） |

**唯一可三者对齐的公共评测面 = Image→Video「世界预测」**：给首帧 + 任务文本，预测后续视频，与 GT 视频比。

**为何不做 action-conditioned 评测**（已决策排除）：
- Cosmos3 action 接口是统一「末端位姿」模态：9D(3D 平移+6D 旋转)+1D 夹爪，官方 embodiment 仅 AV/DROID/UMI，**全是单末端**。
- 前向动力学 fd **官方只支持 Cosmos3-Nano**；**Super 不支持 action**。后端是 cosmos_framework / vLLM-Omni（非 diffusers）。
- 本数据是 **Agilex 双臂 Piper、14D 关节角**，需 FK 转末端 9D+夹爪，且**无双臂 embodiment → 分布外**。
- → 三者无法统一加 action；结论：**只做无 action 的 I2V 横比**（如需，Nano fd 可另作单独探索实验）。

---

## 2. 验证集

`kai0/data/wam_fold_v1/visrobot01_val/`（LeRobot v2.1）：
- **200 episodes**，~295,854 frames，单任务 prompt = `"Flatten and fold the cloth."`
- 三相机：`observation.images.cam_high` / `cam_left_wrist` / `cam_right_wrist`
- 480×640（H×W），30fps，h264
- （自带 `vae_latent/vae_latent_c9/t5_embedding` 是 tau0 训练用，本评测不需要。）

**抽样**：固定 seed 抽 **3 episodes**，**三相机都评**，每个 horizon **滑窗覆盖整段**（步长=H，非重叠平铺）。
生成次数 = Σ_horizon ⌈L_sec / H⌉（L≈episode 秒数）。以 ~50s episode 估：1s≈50 窗 + 3s≈17 窗 + 7s≈7 窗 ≈ **74 窗/(ep,cam)** → 每模型 9 (ep×cam) × 74 ≈ **~666 次生成**，三模型共 **~2000 次生成**。（实际数取决于 3 条抽样 episode 的真实长度。）

---

## 3. 评测设计

### 任务（滑窗覆盖整段 × 多 horizon）
对每个 (episode, camera) 和每个 horizon H ∈ {1s, 3s, 7s}：把整条 episode 按**步长=H 非重叠平铺**成多个窗口；**每个窗口从它的 GT 锚帧（teacher-forced，即条件帧取该窗起点的真实帧）独立 rollout H 秒**，与该窗 GT 比。

| horizon | num_frames @24fps | prompt `duration` | 窗口数(~50s episode) | 每窗 GT(30fps 源) |
|---|---|---|---|---|
| **1s** | 24 | `"1s"` | ~50（起点最多→最慢） | 锚 .. 锚+30 |
| **3s** | 72 | `"3s"` | ~17 | 锚 .. 锚+90 |
| **7s** | 189（原生 7.875s，模型按 `int(189/24)=7s` 标注） | `"7s"` | ~7 | 锚 .. 锚+236 |

要点：
- 每个 horizon **都 ≤ 模型原生单 chunk(7.875s)** → **每个窗口 = 单次生成**，窗口内无需多 chunk 续接。
- **窗口间 teacher-forced 重锚（条件取 GT）→ 无跨窗漂移、互相独立 → 可全并行**。
- 把同一 horizon 的各窗预测**按锚点拼接**即得"每 H 秒重锚的整段预测视频"，与 GT 整段并排。
- **metric-vs-horizon {1s,3s,7s}**：1s/3s/7s 各自聚合全部窗口，得"短/中/长程预测精度"；并可附 **metric-vs-episode-phase**（窗口在 episode 中的位置）。

> 说明 1s 为何最慢：滑窗下 1s 起点最多（~50 窗），总帧数三 horizon 近似相等，但 1s 调用次数最多 → 固定开销/VAE 编解码损耗最大。

### 生成配置（统一，取 I2V 模型自带 `scripts/gen_video.py` 设定）
```
num_frames ∈ {24, 72, 189}（按 horizon）, height=480, width=832, fps=24,
num_inference_steps=50, guidance_scale=6.0, flow_shift=5.0,
add_resolution_template=False, add_duration_template=False, 固定 seed
```

### 分辨率对齐
模型原生 bucket 832×480 (16:9) 生成（质量最好）；评分时把 GT 与生成都**中心裁到 4:3 → 缩放到公共网格（如 256²）**，避免拉伸失真。GT 重采样 30→24fps，取与生成重叠的 horizon。

### 指标
- 复用 `tau0_wm/finetune/eval_gigaworld_dist.py` 的 GPU 实现：**PSNR / SSIM / LPIPS / temporal_absdiff_ratio**（逐帧 + 每 horizon 聚合）。
- **按 horizon 分别报**：1s / 3s / 7s 三组指标 → metric-vs-horizon 曲线（头号科学产出，区分三模型的衰减速度）。
- 加分布级 **FVD**（在 7s 桶上算，样本最足）。
- 定性：逐 (episode,camera) 的 **gen vs GT 并排画廊**，每条含 1s/3s/7s 三个预测（复用 `make_report_html.py` / `gen_video_compare.py`）。
- 动作指标 **N/A**（I2V 不输出 14D 动作）。

### 公平性说明
I2V 专精模型预期占优；结论框定为「**哪个 Cosmos3 对叠衣折叠动力学预测最准**」，非绝对优劣。

---

## 4. prompt text（手写丰富 caption）

I2V prompt 是含 `temporal_caption/duration/fps/resolution/aspect_ratio` 的 JSON。条件首帧已交代外观/视角，caption 重点写**双臂 flatten→fold 时序动作**，并给一句视角提示。三相机共用时序段、只改首句视角。

**cam_high（俯视主视角）**
```json
{
  "temporal_caption": "A top-down overhead view of a tidy tabletop workspace where a dual-arm Agilex Piper robot — two black articulated arms with parallel-jaw grippers entering from the top of the frame — manipulates a single piece of cloth lying flat on a light-colored mat. The camera is fixed directly above, looking straight down. The sequence begins with both arms moving toward the cloth; the grippers pinch opposite edges and pull outward to flatten the fabric, smoothing wrinkles until it lies as a clean rectangle. The two arms then coordinate: one gripper lifts a corner while the other holds the opposite side, folding the cloth in half toward the center so the edges align, then pressing the crease flat. Motion is smooth, deliberate and physically plausible; the fabric deforms naturally, casts soft shadows and settles into a neat folded shape. Lighting is even and diffuse, the mat and background stay static, only the arms and the cloth move.",
  "duration": "7s",
  "fps": 24.0,
  "resolution": {"H": 480, "W": 832},
  "aspect_ratio": "16,9"
}
```
**cam_left_wrist** — 首句换为：`"A close first-person wrist-mounted view from the robot's left arm, the left parallel-jaw gripper prominent in the foreground looking down and forward at a piece of cloth on a light mat, while the right arm assists from the side. The sequence begins ..."`（后接同样时序段）
**cam_right_wrist** — 同理镜像为右臂腕部视角。

`negative_prompt` 复用 I2V 模型 `assets/example_prompt.json` 中的标准负面提示，三相机三模型统一。

> caption 的 `duration` 字段按 horizon 改为 `"1s"/"3s"/"7s"`（与 `num_frames` 24/72/189 对应），其余 `temporal_caption` 文本三 horizon 复用同一段。

---

## 5. 三模型分别如何测试

**完全相同的脚本与 I2V 调用，只换 checkpoint 与 GPU 摆放**：

| 模型 | GPU 摆放 | 并行度(16 GPU)；并行单元 = (ep×cam×horizon×window) ~666/模型 |
|---|---|---|
| Nano 16B | 单卡 `device_map="cuda:i"` | 16 路 |
| Super 64B | 2 卡分片 `device_map="balanced"` | 8 路 |
| Super-I2V 64B | 2 卡分片 | 8 路 |

> **每 worker 只加载一次模型**，循环跑分到的所有 (ep,cam,horizon,window) 单元——Super 120G 加载 ~5min，绝不能每 clip 重载。窗口 teacher-forced 独立 → 可任意分片到 16 GPU。

统一调用（差别仅 `MODEL`/`device_map`；窗口循环按 horizon 步长滑过整段）：
```python
pipe = Cosmos3OmniPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map=PLACEMENT)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)
for H_frames, dur, src_step in [(24,"1s",30), (72,"3s",90), (189,"7s",236)]:   # horizon: 帧数/标注/源步长
    cap = {**CAPTION[cam], "duration": dur}
    for anchor in range(0, ep_len_src, src_step):                  # ← 非重叠滑窗覆盖整段
        gt_anchor_frame = read_frame(ep, cam, anchor)              # teacher-forced 锚帧 = GT
        result = pipe(prompt=json.dumps(cap), negative_prompt=NEG,
                      image=gt_anchor_frame,
                      num_frames=H_frames, height=480, width=832, fps=24,
                      num_inference_steps=50, guidance_scale=6.0,
                      add_resolution_template=False, add_duration_template=False,
                      generator=torch.Generator("cuda").manual_seed(SEED))
        # 与 GT[anchor : anchor+src_step] 比 → 该 (horizon,window) 指标
```

---

## 6. 环境（P0 已解决）

- 本机 8×A100 + b1 8×A100；`/mnt/pfs` 共享 gpfs，模型/venv 在所有节点可见。
- **节点**：`b2 就是当前主机` → 用 **当前主机(b2) rank0 + b1 rank1**（与 `tau0_wm/finetune/launch_gweval_2node.sh` 同 2 节点模式）。b1 = `ssh -p 429 root@120.48.99.93`（同一把 `~/.ssh/id_rsa` 已可达）。
- **venv**: `/mnt/pfs/p46h4f/cosmos/cosmos3-venv`（diffusers 0.39.0.dev0 + Cosmos3OmniPipeline，torch 2.6.0+cu124）。
- **P0 阻塞已解**：`transformers 5.10.2` 在 import 时无条件引用 `torch.float8_e8m0fnu`（torch≥2.7 才有），而本机锁 torch 2.6/cu124（driver 535）。fp8 量化路径在 bf16 推理中不会触发，故在脚本顶部加**一行 shim** 即可（不动 torch/transformers 版本，避免版本矩阵泥潭）：
  ```python
  import torch
  if not hasattr(torch, "float8_e8m0fnu"):
      torch.float8_e8m0fnu = torch.float8_e4m3fn   # fp8 路径推理不用，安全
  from diffusers import Cosmos3OmniPipeline        # ← 实测 6.1s import OK，image= 参数在
  ```

---

## 7. 耗时估算（smoke 后校准）

滑窗覆盖整段：每模型 ~666 窗 (9 ep×cam × ~74 窗)。三 horizon 总生成帧近似相等（各≈整段长度），但 **1s 调用最多→开销最重**。估算（smoke 校准）：

| 模型 | 权重加载(gpfs) | 吞吐 | ~666 窗/模型 @16 GPU |
|---|---|---|---|
| Nano 16B(单卡×16) | ~1–2 min | 快 | ~40–60 min |
| Super 64B(2卡×8) | ~4–6 min | 慢 | ~3–4 h |
| Super-I2V 64B(2卡×8) | ~4–6 min | 慢 | ~3–4 h |

**三模型合计 ≈ 7–9 h**（1s 档占大头）。如需压缩，可在 §9 的"加大 stride"或"7s 切前缀"两个旋钮上权衡。

---

## 8. 执行步骤

0. ✅ **P0 修 env**（shim 实测通过）。
1. **采样器**：固定 seed 抽 **3** val episode；对每 (cam, horizon) 按步长=H 非重叠平铺，枚举全部窗口的 GT 锚帧 + GT 窗口（teacher-forced）。
2. **写 `eval_cosmos3_i2v_dist.py`**：模型循环 × (episode,camera,horizon,window) 分片；每 worker 只 load 一次模型；顶部加 float8 shim；复用 `eval_gigaworld_dist.py` 度量函数；输出 per-model per-horizon per-window JSON。
3. **三相机 caption 文件**（`duration` 按 horizon 注入）+ 复用 `example_prompt.json` 的 negative。
4. **Nano smoke**：1 (episode,camera) 三 horizon 各跑 1-2 窗，端到端 rollout→度量→拼接→画廊，**实测各 horizon 单窗耗时**（据此精算总时长）。
5. **16 GPU 全跑**：3 模型 × ~666 窗（2 节点 accelerate launch，参考 `launch_gweval_2node.sh`）。
6. **聚合**：三模型 × 三 horizon 对比表 + metric-vs-horizon 曲线 + 每 horizon 重锚整段预测 vs GT 并排画廊；结果写入 `docs/training/history/experiments/`。

---

## 9. 风险 / 待确认

- **短 horizon 生成质量**：模型按 7s/189f 训练，强制生成 1s/3s（24/72f + duration 模板）可能略偏离训练分布——这正是 metric-vs-horizon 想测的，smoke 时目检 1s 输出是否退化。
- **成本旋钮（当前=滑窗整段、stride=H、~7–9h）**：① 加大 stride（>H）抽稀起点 → 仍覆盖整段但不密铺，1s 档省最多；② 退化为"每 (ep,cam) 只生 1 条 7s rollout，按 1/3/7s 前缀算指标"（起点=1，最省 ~1h，但失去整段覆盖与多起点统计）。两者均可随时切换。
- **Super off-bucket 4:3**：原生 16:9 832×480，对 4:3 场景生成可能有黑边/构图偏移；已用中心裁对齐缓解，smoke 时目检。
- **caption 分布贴合度**：手写 caption 若仍偏弱，可改用自带 `upsample_prompt.py` 逐帧 LLM upsample（需额外 LLM/端点）。
- **guardrails**：cosmos 默认有 guardrail；diffusers 路径若引入需确认不误伤（叠衣场景无人脸/敏感内容）。
- **b2 授权**：b2 publickey 当前对外被拒，但已确认 b2=当前主机，故无需额外授权。
