# LMWM 像素解码器交付 — 最终收敛 (2026-07-04)

## ✅ 最终决定：两个可选解码方案（milestone 提示场景）

经两条硬性要求（**R1 确定性**：同 latent→同图；**R2 连续性**：近 latent→近图，无跳变）筛选 + 三 episode 实测（见下），**收敛为两个可选方案**，均满足 R1+R2：

| 方案 | ckpt | 特性 | 何时用 |
|---|---|---|---|
| **① flow fixed-noise（首选）** | `dec_best.pt` + `seed=0` | 确定性+连续+**锐利保真**（sharp 510–661），帧间跳变 ≈L1 | 要看清结构（褶皱/夹爪/细节）的提示与展示 |
| **② dec_v2（L1，兜底）** | `dec_v2.pt` | 确定性+连续，绝对稳但**糊**（sharp 97–510） | 只需稳定色块、不在意细节、算力吃紧（前馈单次比 ODE 快） |

**用法**：`make_prod_video_vis_fwd.py --decoder dec_best.pt`（flow 分支已默认 `seed=0` 固定噪声）或 `--decoder dec_v2.pt`。库调用 `decode_best.py: BestDecoder.__call__(..., seed=0)`。

**已排除**：逐帧重采样 flow（违反 R1+R2，闪烁）；裸检索 NN（违反 R2，Voronoi 边界跳变）；`dec_gan_v2`（幻觉）；`dec_reencode_v2`（噪声纹理，游戏度量）。

**已知边界（decoder OOD）**：两个解码器都在 kai0（米/黑衣物）上训练；对**训练未见的颜色**（如橙色 T 恤 vis_base/2026-06-28）会**串色**（自重建即错色，证明是解码器 OOD 非预测错）。修复须用含目标颜色的帧重训/微调解码器。

**最终证据资产**（保留）：
- `assets/decoder_compare_kai0base_testep8.{png,mp4}` — kai0_base 留出测试集(ep8)：域内两方案对照，颜色正确、flow-fixed 更锐。
- `assets/decoder_temporal_stability.png` — R1/R2 四方案实测胶片（含被否的 fresh-flow / 检索 NN）。
- `assets/selfrecon_orange_ood_0628.png` — 橙色 OOD 串色诊断（自重建证明是解码器域外）。

三 episode 实测汇总：

| episode | 域 | dec_v2 跳变/锐 | flow-fixed 跳变/锐 | 颜色 |
|---|---|---|---|---|
| kai0_base ep8（测试集） | 域内 | 0.030 / 510 | 0.029 / **661** | ✅ 正确 |
| vis 2026-04-23（黑衣） | 近域 | 0.080 / 97 | 0.085 / **619** | ✅ 大致对 |
| vis 2026-06-28（橙 T 恤） | 域外 | — | — | ❌ 串色(OOD) |

---

# （背景）LMWM 最优解码器交付 — flow-matching 像素解码器 (2026-07-03)

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

## ⚠️ 逐帧视频 / milestone 提示场景：为什么改用确定性解码器（2026-07-04 修正）

上面的排名是**单帧保真**的结论——flow 确实最锐利最保真。但**做 milestone 提示（把预测 milestone 解码成图给策略/人看）时最终没有用 flow**，原因是**时序稳定性**：

- 我预测的 milestone 是**分段常值信号**：连续多帧对应同一个 milestone（同一个目标隐向量）。
- flow 是**生成式**解码器，每次调用从新采样的噪声出发跑 ODE → 即使输入隐向量**完全相同**，逐帧解码也会得到**不同**的图 → 提示在同一个 milestone 内部**来回跳变 / 闪烁**。这对"稳定提示下一步该长什么样"是致命的。
- 确定性解码器（`dec_v2` L1，单次前向、无采样）：**同隐向量 → 同像素** → 同一 milestone 内提示**纹丝不动**，只在 milestone 切换时变化。这正是想要的行为。

**交付选择**：milestone 提示视频用 `dec_v2`（L1，确定性、保真、无幻觉，代价是糊）。`dec_best`（flow）仍是**单帧/静态展示**的最佳解码器。见 `make_prod_video_vis_fwd.py --decoder`（`load_any_decoder()` 自动识别 flow vs pooled）。

**关键认知：稳定性 与「生成式与否」是两个正交轴**。闪烁的根因是**每帧重采样噪声**，不是"生成式"本身。若 (a) 每段只解码一次并保持、或 (b) 固定噪声种子，flow 也能变成时序稳定——即不必为了稳定牺牲 flow 的锐度。更优方案见下节。

## 📋 更优解码方案调研（sharp + faithful + 时序稳定，2026-07-04）

**两条硬性要求**（用户明确，缺一不可）：
- **R1 确定性**：同一隐向量重复解码必须得到**完全相同**的图（可重复）。
- **R2 连续性**：`milestone+1` 在当前 stage 下目标本身不变，变的只是 LMWM 对它的**预测**；连续帧上预测的 `milestone+1` 隐向量**近似相同**（非严格相等），所以解码函数必须**把邻近隐向量映到邻近图**——不允许"隐向量微变→图巨变"。

R2 是关键判据：它要求解码器对隐向量是**连续/Lipschitz** 的，而不只是分段稳定。**这否决了检索式解码**——最近邻是隐向量的**分段常值/不连续**函数，Voronoi 边界处隐向量微动就会跳到另一张真实帧 → 巨大跳变，恰恰违反 R2。

### 实测（40 连续帧，隐向量 consecutive-cos 0.947，即"近似相同"的真实序列；`temp/decoder_stability_test.py`）

| 解码器 | R1 重复解码Δ (需=0) | R2 帧间跳变 (需小) | 最大单帧跳变 | 锐度 |
|---|---|---|---|---|
| flow **fresh-noise**（现 `dec_best` 逐帧） | **0.149 ✗** | **0.116 ✗** | 0.172 | 693 |
| **flow fixed-noise（固定噪声）** | **0.000 ✓** | **0.047 ✓（最平滑）** | 0.095 | **610** |
| `dec_v2`（L1，当前交付） | 0.000 ✓ | 0.062 ✓ | 0.104 | 125（糊） |
| 检索式 NN | 0.000 ✓ | 0.066 | **0.147 ✗** | 508 |

（对照图 `assets/decoder_temporal_stability.png`：4 行×8 连续帧。fresh-noise 逐帧抖动/夹爪忽隐忽现；fixed-noise 平滑且锐利；L1 平滑但糊；检索式出现姿态离散跳变。检索 NN 索引 `428→359→368→359→428…` 直接暴露边界跳变。）

**结论：同时满足 R1+R2 的只有两个——`flow fixed-noise`（锐利，帧间最平滑）与 `dec_v2`（糊）。fresh-noise flow 两条都违反；检索式违反 R2。**

### 排序（满足 R1+R2 前提下）

1. **flow fixed-noise（固定噪声 flow）— 首选**。冻结 ODE 初始噪声跨帧（`seed=0`）→ flow 退化为隐向量的确定性连续函数：同隐向量→同图（R1），近隐向量→近图（R2，实测帧间跳变 0.047 比 L1 还小），且保留 flow 锐度（610）与保真无幻觉。已落地：`decode_best.py __call__(..., seed=0)`；视频脚本 `make_prod_video_vis_fwd.py --decoder dec_best.pt` 的 flow 分支已默认走 `seed=0`。文献支撑：latent reuse / consistent DDIM init / FastInit（固定/共享噪声做时序一致）。
2. **`dec_v2`（L1，当前交付）**。前馈网络 → 天然确定性(R1)+连续(R2)，帧间平滑无闪烁；唯一缺点是糊（锐度 125）。**安全兜底**（用户已选）。
3. **每段解码一次 + 保持**。仅当预测隐向量分段严格常值时成立；但本场景预测逐帧微变（近似而非相等），"保持"会丢掉真实的细微演化，不如 fixed-noise 的连续解码贴合。
4. **~~检索式解码（最近真实帧）~~ — 已否决**。虽满足 R1，但 NN 不连续 → 违反 R2（实测最大跳变 0.147，接近 fresh flow）。除非加重滞回/持续 N 帧/top-k 混合，但混合又回到糊。此前的推荐在只考虑 R1 时给出，R2 加入后作废。
5. **`dec_gan_v2`（GAN）**。确定性+连续、锐利，但**幻觉**（凭空造细节）→ 保真不过关。
6. **时序相干噪声 / 光流 warp（Go-with-the-Flow）**。比 fixed-noise 更重，仅在隐向量大幅漂移时才需要；本场景 fixed-noise 已足够。

**最终建议**：milestone 提示解码用 **flow fixed-noise**（`dec_best` + `seed=0`）——唯一同时满足 R1+R2 且锐利保真的方案；`dec_v2` 作为糊但绝对稳的兜底。**不要**用逐帧重采样 flow（违反 R1+R2），**不要**用裸检索 NN（违反 R2）。

参考文献：
- Go-with-the-Flow: Motion-Controllable Video Diffusion Using Real-Time Warped Noise — https://arxiv.org/abs/2501.08331
- FastInit: Fast Noise Initialization for Temporally Consistent Video Generation — https://arxiv.org/abs/2506.16119
- TokenFlow: Consistent Diffusion Features for Consistent Video Editing — https://arxiv.org/abs/2307.10373
- Warped Diffusion: Solving Video Inverse Problems with Image Diffusion Models (NeurIPS 2024) — https://arxiv.org/abs/2410.16152

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
