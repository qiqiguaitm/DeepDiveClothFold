# 阶段性落档:LaWM 实测对比 · 可测性分析 · ckpt 现状(2026-07)

承接 web 报告 §03/§04 的 LaWM-vs-LMWM 对比表。本文把"哪些 `—` 是可测的、需要什么、以及当前 ckpt 卡点"固化下来,避免下次重新推导。

---

## 1. 对比表里 `—` 的三类可测性

把 LaWM 侧留白的指标按"能不能补、补的成本"分三类:

### ① 原生可得(改评测脚本即可,零训练)
- **oracle grid-cos** = 0.770 —— 已有(`eval_lawm_lam.py`,recon vs tgt)。
- **persistence** = 0.627 —— 已有(dec_in vs tgt)。
- **lift over persistence** = +0.143 —— 已有,**唯一同口径可硬比的数**。
- **recon L1**(可选补)—— `out["recon"]` 已暴露,加一行 L1 即可。

### ② 需建 deploy-predm 附加件(~30–60min 小训练)
LaWM 的 LAM **原生没有 predm**——部署码由下游 VLA policy 产出,所以这些是"—"是**架构选择**,不是不能测:
- **deploy grid-cos**
- **deploy 净增益**(deploy − persistence)
- **deploy best-of-8**
- **model-lag 欠射 ratio**

补法(与我们 `optimize_subgoal.py` 的 predm 协议**完全对齐**,保证公平):
1. 冻结 LaWM,用它的 `encoder`(inverse,Perceiver-query Transformer)在 [当前, 未来] pair 上取 teacher code `quantized`;用它的 `decoder(features=dec_in, actions=code)`。
2. 在冻结码空间上训一个小 predm:`predm(dec_in) → ẑ`,loss = `smooth_l1(decoder(dec_in, ẑ), tgt)`(重建式,镜像我们的 l2)。带 VAE 头做 best-of-8。
3. 一次算出上面 4 个数,填表,标注"同/异编码器空间 + 训练充分度"。

API 可行性(已从 `lam_model.py` 确认):`get_latent_action(...)` 返回 dict 暴露 `recon/dec_in/tgt/quantized`;`decoder(features=dec_in, actions=quantized)`(line 453)是 **AdaLN-DiT**:current grid 当空间基底,code 通过 shift/scale/gate 调制 LayerNorm(zero-init gate)。`vision_encoder.encode` 可单独调。→ deploy-predm 完全可搭。

### ③ 真·N/A(不是 LaWM 的模型属性)
- **corr(value, time)**、**progress 单调 / 负滞后比例** —— 这些是 **CRAVE 目标构建**的性质(进度标签质量),不是 world-model 的性质。LaWM 没有"进度目标"这个设计,填 N/A 是**诚实**的,不是缺测。
- **下游 SR**(LIBERO/RoboTwin/真机)—— LaWM 论文有;我们没接 VLA,是我们的缺口(不是 LaWM 的)。

---

## 2. ckpt 现状(卡点)

| 候选 | 是什么 | 空间 | 可得性 |
|---|---|---|---|
| `LaWAM/ckpts/pytorch_model.pt` | **官方 LaWM**(vendored `LatentLAMModel`,ViT-B/16 768) | 异空间(ViT-B 768 vs 我们 ViT-H+ 1280)| **本地/gf3 都已被清理**,重下需 ~2.9GB(走 gf3 aria2c) |
| `checkpoints/grid_predictor/G_lawm.pt` | 17.4M **CNN**(`fc/up` 层)—— 红鲱鱼,**不是** LaWM | — | 本地有但无意义 |
| `train_lawm_gf3.py` 产出 `temp/lmwm_p0/lawm_members/member_*.pt` | 真·vendored `LAMEncoder`,`input_dim=1280`=**我们 DINOv3-H 同空间**,ctx768/6层/12头,milestone 条件 | **同空间(1280)** | 本地 `lawm_members/` 目前为空 → ckpt 疑在 gf3 或别机;**这最可能就是"训了一半的 LaWM"** |

**结论/推荐**:同编码器空间(DINOv3-H 1280)的 half-trained LaWM,比"换编码器的收敛官方 ckpt"**更适合做 deploy 对比**——它消掉了"编码器空间不同"这个最大免责声明,deploy 0.700 能真·同口径比。代价是半训练=LaWM 下界(表里标注即可)。

**待用户提供**:half-trained LaWM 的 ckpt 绝对路径(哪台机)。拿到后即可跑 §1.② 的 deploy-predm。官方 ViT-B 收敛 ckpt 作为"异空间收敛参照"第二行,可选(需重下 2.9GB)。

---

## 3. web 报告已落地内容(回执)
- 标题去"最终":`Milestone 预测器 — 版本`。
- §04 加了 LaWM-vs-LMWM 结构对比 SVG(冻结编码器/inverse/码瓶颈/解码器/部署 predm 五行,Transformer vs CNN)。
- §03 加了"实测对比 · LaWM vs 我们 V3.1+VAE"表 + 脚注 ¹²³⁴ 解释每个留白原因。
- 单文件版 `lmwm_report_standalone.html`(2.06MB,hero+2 视频内联)已 gitignore。
- 首页 `marked.min.js` 已本地 vendored(GFW 下 jsdelivr 挂 → 首页持续加载,已修)。
