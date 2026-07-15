# DINOv3-base 时序一致视频解码器（可视化，2026-07-11）

> **目标**：为 DINOv3-base 特征训解码器做可视化。硬要求：**视频流解码的时序一致性**（连续帧要一致、不闪烁）。
> **两种方法都实现**：**pooled**（768D 每帧一向量 → 软原型，轻量）+ **grid**（768×16×16 空间 token → 锐利）。二者都加时序一致约束。
> **结论**：① 时序约束相对逐帧 baseline **静态区抖动 −73%**、运动同步 0.76→0.88；② **grid 比 pooled 锐 3.4×**（val_recon 0.051 vs 0.177），空间布局贴合真实。

---

## 1. 为什么逐帧解码不够

现有 `train_dinov3h_decoder.py` 逐帧独立：`pooled → fc → 5×上采样 → 3×128×128`。pooled 特征对相邻帧的微小变化敏感 → 解码细节逐帧跳变 → 播成视频**闪烁**。pooled 无空间 grid，解码本就是"软性可读原型"，逐帧独立会让这个原型抖动。

## 2. 设计（可迭代）

```
clip of pooled feats (B,T,768)
  └─① 时序上下文编码 Conv1d over TIME (kernel tk=5, 2 层, ctx=512)
        每帧解码看到 ±邻帧 → 相邻帧共享重叠感受野 → 天然平滑
  └─② 逐帧图像头 ctx → fc → (512,4,4) → 5×上采样 → 3×R×R (Tanh)
损失（clip 上）:
  recon    = |D−I| + 0.5|D−I|²
  temporal = |(D_t−D_{t-1}) − (I_t−I_{t-1})|   ← 解码帧间运动 == 真实帧间运动
  gdl      = 空间梯度差(锐度)
  loss = recon + tc_weight·temporal + gdl_weight·gdl
```

**关键**：temporal 项匹配**真实**帧间 delta，而非零运动先验 → 一致性**不以抹平真实运动为代价**（真动它就动，真静它才静）。

推理：整条 episode 序列一次过 Conv1d（每帧得到完整时序上下文）→ 逐帧解码 → 连贯视频。

## 3. 结果（kai，ep100，823 帧）

**训练收敛**：val_recon 0.352→**0.177**，val_tc 稳定 ~**0.034**（clip=8, n_clips=2500, 50 ep）。

**A/B 时序一致性**（vs 逐帧 baseline clip=1/tc=0，同数据量）：

| 指标 | per-frame | **temporal** | 更优 | 说明 |
|---|---|---|---|---|
| **static_flicker** | 0.0234 | **0.0062** | ↓ **73%** | 场景静止时的解码抖动——核心一致性指标 |
| tc_error | 0.0527 | **0.0316** | ↓ 40% | 解码运动 vs 真实运动的偏差 |
| motion_corr | 0.760 | **0.884** | ↑ | 解码视频运动与真实同步度 |
| dec_motion | 0.0405 | **0.0148** | 更贴真实 | per-frame 0.041 >> 真实 0.025=乱动/闪；temporal 0.015 稳定 |
| real_motion | 0.0247 | 0.0247 | — | 参考 |

→ 逐帧解码器在静止场景**乱动**（dec_motion 远超真实），时序解码器把静态抖动压掉 73%、运动与真实同步。**硬证明时序设计有效。**

## 3.5 两种方法：pooled vs grid（锐度）

pooled 糊的**根因是特征本身**：pooled 768D 是整帧一向量、无空间布局 → 解码欠定 → 软原型（调损失/GAN 也只能锐化这个糊原型）。**grid 方法**改用 DINOv3-base 的 **16×16×768 patch grid**（空间 token 保留）→ 解码器能重建空间细节。

| | **pooled** (`train_dinov3base_video_decoder.py`) | **grid** (`train_dinov3base_grid_decoder.py`) |
|---|---|---|
| 输入 | pooled 768D（crave/data 现成） | DINOv3-base `encode_grid` → 768×16×16（需重编码，srpo 环境） |
| 解码器 | Conv1d 时序上下文 + fc→5×上采样 | `make_decoder(768,"big")` grid 16→128 + conv 细节 |
| 时序 | 时序上下文 + tc 损失 | tc 损失（grid 天然帧间连续，逐帧+tc 已够） |
| **val_recon** | 0.177 | **0.051**（锐 3.4×） |
| val_tc | 0.034 | 0.027 |
| 画质 | 软"可读原型"（gist 级） | **锐利**，布形/夹爪/褶皱可辨、贴合真实布局 |
| 成本 | 轻（无需 DINOv3 前向） | 需 DINOv3-base 编码 grid（GPU，srpo） |

**何时用哪个**：
- **grid** → 要看清内容（演示、审阅、锐利可视化）。**首选**。
- **pooled** → 只要 gist / 极轻量 / 直接可视化 LMWM 预测的 **pooled 子目标**（预测器输出就是 pooled，pooled 解码正好同空间）。

视觉对比见 `docs/assets/dinov3base_pooled_vs_grid_ep100.mp4`（real|pooled|grid）：pooled 是青色糊团，grid 清晰还原布形与双夹爪。

## 4. 交付物

- 训练：[`train_dinov3base_video_decoder.py`](../scripts/train_dinov3base_video_decoder.py)（pooled）· [`train_dinov3base_grid_decoder.py`](../scripts/train_dinov3base_grid_decoder.py)（grid, srpo 环境）
- 视频生成：[`make_dinov3base_decode_video.py`](../scripts/make_dinov3base_decode_video.py)（pooled，H.264）· [`make_grid_vs_pooled_video.py`](../scripts/make_grid_vs_pooled_video.py)（三联对比, srpo）
- A/B 时序量化：[`compare_decoder_temporal_consistency.py`](../scripts/compare_decoder_temporal_consistency.py)
- ckpt：`dinov3base_decoder/kai_video_dec.pt`（pooled 时序）· `kai_grid_dec.pt`（grid）· `kai_perframe_dec.pt`（逐帧对照）
- 视频（本地，git-ignored *.mp4，均 H.264/avc1 可 VS Code 播）：
  - `dinov3base_decode_ep100.mp4`（real|pooled）
  - `dinov3base_decode_compare_ep100.mp4`（real|per-frame|temporal，看时序 flicker 差）
  - `dinov3base_pooled_vs_grid_ep100.mp4`（real|pooled|grid，看锐度差）
- **视频编码**：cv2 只能写 mp4v（VS Code/浏览器不认），生成脚本已内置 `write_h264()` 自动 ffmpeg 转 H.264(avc1)。

## 5. 用法

```bash
# 训练（从 repo 根，绝对路径；kai0/.venv 有 torch+cv2）
CUDA_VISIBLE_DEVICES=0 kai0/.venv/bin/python lmvla/lmwm/scripts/train_dinov3base_video_decoder.py \
  --feature_dir lmvla/crave/data/kai_dinov3base --dataset_root kai0/data/Task_A/kai0_base \
  --clip 8 --n_clips 2500 --epochs 50 --tc_weight 1.0 --gdl_weight 0.3 \
  --out lmvla/lmwm/checkpoints/dinov3base_decoder/kai_video_dec.pt
# grid 版训练（srpo 环境，需 DINOv3-base 编码 grid）
CUDA_VISIBLE_DEVICES=0 /home/tim/miniconda3/envs/srpo/bin/python lmvla/lmwm/scripts/train_dinov3base_grid_decoder.py \
  --feature_dir lmvla/crave/data/kai_dinov3base --dataset_root kai0/data/Task_A/kai0_base \
  --clip 6 --n_clips 1600 --epochs 60 --dec big --tc_weight 0.5 \
  --out lmvla/lmwm/checkpoints/dinov3base_decoder/kai_grid_dec.pt
# 解码成视频：pooled 单版 / grid-vs-pooled 三联对比
CUDA_VISIBLE_DEVICES=0 kai0/.venv/bin/python lmvla/lmwm/scripts/make_dinov3base_decode_video.py \
  --ckpt lmvla/lmwm/checkpoints/dinov3base_decoder/kai_video_dec.pt \
  --feature_dir lmvla/crave/data/kai_dinov3base --dataset_root kai0/data/Task_A/kai0_base \
  --episode 100 --out lmvla/lmwm/docs/assets/dinov3base_decode_ep100.mp4
CUDA_VISIBLE_DEVICES=0 /home/tim/miniconda3/envs/srpo/bin/python lmvla/lmwm/scripts/make_grid_vs_pooled_video.py \
  --pooled_ckpt lmvla/lmwm/checkpoints/dinov3base_decoder/kai_video_dec.pt \
  --grid_ckpt lmvla/lmwm/checkpoints/dinov3base_decoder/kai_grid_dec.pt \
  --feature_dir lmvla/crave/data/kai_dinov3base --dataset_root kai0/data/Task_A/kai0_base \
  --episode 100 --out lmvla/lmwm/docs/assets/dinov3base_pooled_vs_grid_ep100.mp4
```

## 6. 迭代旋钮 / 局限（诚实）

- **pooled 768D 无空间 grid → 解码软（可读原型）**。要锐度：提取 DINOv3-base **grid 特征**(16×16×768) 训 grid 解码器（空间信息足），或接检索(medoid)出锐样例。当前 pooled 匹配数据集现状 + LMWM 预测的是 pooled 子目标，故 pooled 解码正好可视化预测。
- **tc_weight 权衡**：越大越稳但会略微抹平真实运动（当前 dec_motion 0.015 < 真实 0.025，偏稳）。要更贴真实运动可降到 0.3–0.5。
- 更强一致性备选（未做）：**前一解码帧自回归条件**（decode P-frame 式，静态区完全沿用上一帧）——一致性最强但训练需 scheduled sampling 防曝光偏差。当前"时序上下文 + tc 损失"已达 −73% flicker，够用且训练稳。
- 其它数据集（coffee/vis/xvla）同法可训，换 `--feature_dir`/`--dataset_root`/camera。
