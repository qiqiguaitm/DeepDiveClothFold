# AC-Cosmos3 L3 改进方案：动作注入架构重设计（对标 PAIWorld + Ctrl-World）

> 2026-06-27。背景：头对头已证实 AC-Cosmos3 控制性 ΔPSNR(GT−错误)=**+0.16**，Ctrl-World=**+8.17**，两轴全面落后（见 `wam_fold_wm_runs/reports/ladder_eval_summary.tsv`）。L1/L2 增量阶梯（抽帧/历史/IDM损失）补不平 → 问题是**架构性的**。本文深调研 PAIWorld(arxiv 2606.18375v3) + Ctrl-World 的动作注入设计，定位 Cosmos3 的根因，给可落地的 L3 方案。

## 1. 三方设计对比（grounded in code/paper）

| 维度 | **AC-Cosmos3（我们）** | **Ctrl-World** | **PAIWorld** |
|---|---|---|---|
| 骨干 | Cosmos3-Nano 16B **MoT**（统一 token 序列，joint/two-way self-attn）| **SVD UNet**（~1.5B，spatio-temporal）| **Cosmos-Predict2.5 14B DiT** + flow-matching |
| Latent | Wan2.2 VAE | SVD-VAE | Wan2.1 VAE |
| **动作注入** | **DomainAwareLinear `action2llm`→动作 token 塞进统一序列**，靠 joint self-attn 自己去关联（无专用通路、无空间锚定，动作被稀释）| **帧级动作 token → cross-attention**（`encoder_hidden_states=action_hidden`，`frame_level_cond=True`）| **动作渲染成空间 action map（EVAC 式）→ 沿 channel 维 concat 到 noisy latent**（像素/latent 空间锚定）|
| 外观/历史条件 | 历史帧作为序列里的 clean token（sequence_packing）| **当前帧 latent 沿 channel concat 到每个 noisy 帧**（`cat([...,condition_latent],dim=2)` = 外观/运动解耦）+ 6 帧 clean 历史 | context 帧多视角 concat；AdaLN 注入 text |
| 时间分辨率 | 训练用原生帧率（L1 才抽到 ~10hz）| down_sample=3 → **10hz 粗时间**，预测5帧/skip∈{1,2} | 任务相关 |
| 训练目标 | flow/diffusion（video-only），无动作辅助损失 | diffusion + 外观条件 | **L_diff + 0.5·L_REPA**（Depth-Anything-3 token 关系蒸馏，空间+时间）|
| 控制性专门机制 | 无 | cross-attn 专用通路 + 外观解耦 | 空间 action map + **Geo-RoPE**（ray/pose 子空间）+ **几何感知 cross-view attn**（AdaLN-Zero 门控）+ 3D-REPA |
| 实测控制性 | **+0.16 ΔPSNR（WEAK）** | **+8.17 ΔPSNR（STRONG）** | WorldArena Controllability **87.16**，EWMScore 72.31（1st）|

## 2. 根因：为什么 Cosmos3 几乎不跟随动作

**两个强模型有一个共同点，恰是我们缺的：动作走"专用、结构化、锚定"的通路。**

- **Ctrl-World**：动作经 cross-attention 注入——动作是 query/key 的**显式外部条件**，每个去噪层都被动作直接调制；且帧级（每帧一个动作 token）。
- **PAIWorld**：动作**渲染成空间图**（末端轨迹投影到每个相机视角的像素），沿 channel concat 到 latent——动作与"它该影响画面的哪个位置"在**空间上对齐**，监督信号极强。

**而 AC-Cosmos3**：动作只是被 `action2llm` 投影成几个 token，**平铺进统一 self-attention 序列**，和成千上万视觉 token 混在一起竞争注意力。没有专用通路、没有空间锚定、没有逐层强制调制 → 在**弱信号**（叠衣服微动作）下，模型发现"忽略动作、只做视频续写"loss 更低（自回归续写本身就能拿高 PSNR）→ 动作被学成噪声。这正是 ΔPSNR≈0 的机制（+ domain16/17 头是 NVIDIA 跳过的空槽、fresh-init 无先验，雪上加霜）。

## 3. 关键洞见：PAIWorld 证明"在 Cosmos 骨干上就能拿到强控制性"

> **PAIWorld 的骨干就是 Cosmos（Predict2.5 14B DiT）**，和我们的 Cosmos3 同源。它没换骨干，只是**把动作注入改成"空间 action map + channel concat"**，就在 WorldArena 拿到 Controllability 87.16 / 第一名。

这把 L3 从"移植 SVD 的 cross-attn（架构差异大、风险高）"降级成**"在 Cosmos3 DiT 上改动作注入方式"**——一个**同骨干、可落地**的改造。这是本方案的核心：**不追 Ctrl-World 的 SVD 设计，而是抄 PAIWorld 在 Cosmos 上验证过的 channel-concat 空间动作条件。**

## 3.5 ⭐ 精确实现源头 = EVAC（EnerVerse-AC, arXiv:2505.09723），非 PAIWorld

**核查纠正**：PAIWorld 动作注入是**照搬 EVAC**（原文 "we follow EVAC [52] and render actions into spatial action maps..."），它自己**无动作注入独创**（贡献是 3D 多视角一致性）。要抄就抄 EVAC。EVAC 的精确机制（可照写）：

- **Action map（EEF 投影图,channel-concat）**：EEF 世界坐标 →（标定内外参 K）→ 像素坐标；roll/pitch/yaw 画成沿轴**单位向量箭头**；夹爪用**单位圆**（浅=开/深=合）；黑底、左右手异色 → 编码成 latent → channel-concat。
- **Delta Action Attention（逐帧运动,cross-attn）**：相邻帧 EEF delta 位姿 → linear projector → 定长 token → 与 reference image 融合 → **cross-attention** 注入 UNet。
- **Ray map（相机运动,channel-concat）**：ray (o_r,d_r) 与 trajectory map 拼接；腕相机随臂动 → 隐式编码 EEF 运动。
- **精确通道布局（EVAC Table 2,UNet 输入 19 通道）**：latent 4 + **condition image(外观) 4** + **action map 4** + **ray map 6** + dropout mask 1（concat）**＋** delta action token（cross-attn）。

→ **我们的 L3-A/B/C 恰好就是 EVAC 的三路**（action-map concat + 外观 concat + delta cross-attn），应合成一套实现。我们有 `config/calibration.yml`（内外参）+14维关节态→FK 得 EEF，完全可复现。

## 4. L3 改进方案（按 性价比 排序）

### L3-A（首选，最高性价比）：空间 action map + channel-concat（抄 PAIWorld/EVAC）
- 把 14 维动作（双臂关节+夹爪 → 正运动学得末端 6DoF 轨迹）**渲染成空间图**：末端位置投影到各相机视角（top_head/hand_left/hand_right，用 `config/calibration.yml` 的内外参），画成 heatmap/轨迹通道。
- VAE/下采样到 latent 分辨率 → **沿 channel 维 concat** 到 noisy latent（每帧都拼）。
- 改 patch-embed 输入通道数（+ action-map 通道），新增通道 **zero-init**（不破坏预训练，AdaLN-Zero 同理）。
- **代码位置**：`omni_mot_model.py` 的 latent 拼接处 + patchify 输入投影；数据侧 `wam_fold_dataset.py`/`transforms.py` 产 action-map。
- **为什么有效**：动作与画面空间对齐 = 监督信号从"几个被稀释的 token"变成"逐像素对齐的强条件"，弱信号也压得住。

### L3-B（叠加，Ctrl-World 式外观解耦）：当前帧 channel-concat
- 把**首/当前帧 latent 沿 channel concat 到每个 noisy 帧**（Ctrl-World `condition_latent`）→ 外观（静态，来自当前帧）与运动（动态，来自动作）解耦，模型只需学"动作如何改变外观"，不用从零重画外观。
- 同样 zero-init 新通道。**代码**：同 L3-A 的拼接点。

### L3-C（若仍不足）：帧级动作 cross-attention / AdaLN-Zero 专用通路
- 若坚持 token 路线：把动作从"塞进 joint self-attn"改成**专用 cross-attention 层**（或 AdaLN-Zero 调制），帧级（每帧一动作 token）。新层 zero-gate 初始化保预训练。比 L3-A 改动大、更接近 Ctrl-World，但同骨干可做。

### L3-D（可选增强，长程/多视角）：REPA 3D 一致性损失
- PAIWorld 的 `L_REPA`（Depth-Anything-3 token 关系蒸馏，λ=0.5）提升多视角 3D 一致性、Scene Consistency 0.90。叠衣是 3 相机多视角 → 可加；但属锦上添花，控制性主要靠 L3-A/B。

## 5. 落地次序与验证
1. **先 L3-A + L3-B**（一起，都是 channel-concat + zero-init，改动集中在 latent 拼接 + patch-embed，最小风险）。
2. 用现成头对头裁判：`Ctrl-World/scripts/eval_clothfold_ctrl.py` 同协议（ΔPSNR GT vs 错误动作）+ `ladder_eval_summary.tsv`。**目标**：把 ΔPSNR 从 +0.16 推向 Ctrl-World 的 +8.17 量级（先到 +2~+3 就证明架构对了）。
3. 不行再加 L3-C（cross-attn 专用通路），最后 L3-D（REPA）。
4. 资源：本机 8卡先 smoke + 小步验证 action-map 通路是否"动作一改、画面就变"（ΔPSNR 早期信号），再上集群全量。

## 6. 一句话结论
我们的动作注入（token 塞进 joint self-attn）是三者里最弱的形态。**两个强模型都用"专用+空间锚定"的动作通路**；PAIWorld 还证明这能在 **Cosmos 同骨干**上实现。L3 的正解 = **抄 PAIWorld 的 channel-concat 空间 action map（L3-A）+ Ctrl-World 的外观 channel-concat（L3-B）**，zero-init 不破坏预训练，同骨干可落地——这是把 +0.16 真正推向 +8 的唯一架构路径。
