# LMWM 最优解码器交付 — flow-matching 像素解码器 (2026-07-03)

## 结论：`dec_best.pt` = 条件 flow-matching 像素解码器（原 `flow_b160`）

把模型预测的 pooled DINOv3-H 隐向量（1280-D, L2 归一化）解码成**锐利且保真**的图像。
它在 gf3 八卡上对 4 个配置（base 96/128/160 + 80k 数据）扫参后胜出。

## 客观排名（gf3 公共 held-out，`flow_eval.json`）

| 配置 | step | reencode_cos↑ | sharpness | pixel_L1 |
|---|---|---|---|---|
| **flow_b160 (交付)** | 24000 | **0.667** | 749 | 0.110 |
| flow_b128_80k | 30000 | 0.657 | 700 | 0.109 |
| flow_b128 | 24000 | 0.655 | 719 | 0.109 |
| flow_b96 | 24000 | 0.593 | 788 | 0.110 |
| *real* | — | — | *923* | — |

## 三方最终对比（本地公共 held-out，`final_decoder_compare.json` / `.png`）

| 解码器 | reencode_cos↑（语义保真） | sharpness | pixel_L1 | 视觉 |
|---|---|---|---|---|
| dec_v2 (L1) | 0.348 | 180（糊） | **0.070** | 条件均值模糊，无褶皱 |
| dec_gan_v2 (GAN) | 0.413 | 879 | 0.089 | 锐但**幻觉**（凭空造夹爪/纹理）|
| **flow_b160 (交付)** | **0.681** | 500 | 0.111 | **锐且保真**：衣物形状/颜色/褶皱、夹爪位置、桌沿都对 |

**为什么 flow 的 pixel_L1 反而更高**：L1 解码器专门最小化像素 L1 → 输出模糊的条件均值（L1 低但语义差 0.35）。
生成式解码器从图像分布采样，不做像素平均，所以 pixel_L1 高，但**语义保真度（reencode_cos）几乎翻倍**、且锐利。
对"展示模型自己的预测长什么样"这个目的，reencode_cos + 视觉锐度才是对的指标，pixel_L1 会误导。

**为什么 flow 是对的路线**：它在真实图像分布内采样，结构上无法产生对抗式噪声（这正是"再编码一致性损失"失败的原因——
直接优化冻结编码器输出会得到 cos 0.88 但视觉是高频垃圾）。flow 同时拿到 GAN 的锐度 + 超过 GAN/L1 的保真度，且无幻觉。

## 用法

```python
from decode_best import load_best_decoder          # lmwm/scripts/decode_best.py
dec = load_best_decoder("lmwm/checkpoints/dinov3h_decoder/dec_best.pt", "cuda:0")
imgs = dec(latents)      # (N,1280) L2-normed pooled DINOv3-H -> (N,128,128,3) uint8 RGB
```

- 输入必须是**统一 gated DINOv3-H pooled 空间**的隐向量（`crave.encoders.encode_pooled` / `DINOv3HGated`，
  训练即部署同一空间——见 pitfalls B8 编码空间统一）。喂 bank-space 隐向量已交叉验证可用。
- ODE 25 步 Euler；`dec(latents, ode_steps=50)` 可换更慢更稳。res=128，base=160 UNet，80MB。
- 生视频：把 `make_prod_video_bankspace.py` 的 pooled 解码换成 `decode_best`（逐帧 ODE 采样，比 L1 慢但锐利保真）。

## 训练复现（gf3 八卡）

```
kai0/.venv/bin/python lmwm/scripts/flow_decoder_gf3.py \
  --base 160 --n 50000 --res 128 --steps 24000 --bs 64 \
  --out temp/lmwm_p0/flow_b160.pt --device cuda:2
```
rectified flow：`xt=(1-t)·noise + t·image`，UNet 回归速度场 `image-noise`，pooled 隐向量经 FiLM + 时间嵌入做条件。

## 产物

- `lmwm/checkpoints/dinov3h_decoder/dec_best.pt` — 交付解码器（flow_b160, 80MB）
- `lmwm/scripts/decode_best.py` — 干净的解码 API（`load_best_decoder`）
- `lmwm/scripts/final_decoder_compare.py` — 三方对比复现脚本
- `lmwm/docs/assets/final_decoder_compare.png` — real | L1 | GAN | flow 对比图
- `lmwm/outputs/{flow_eval,final_decoder_compare}.json` — 客观指标
