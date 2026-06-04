# X-VLA 相机鲁棒精确抓取 — 最终建议与方案认证 (probe-validated)

> **目的**: 在 [`../../analysis/xvla_innovation_directions.md`](../../analysis/xvla_innovation_directions.md) 提出的方向上, **自己跑离线小实验验证可行性**, 给出 probe 证据支撑的最终建议 + 可执行方案 + GO/NO-GO 认证。
> **建立**: 2026-06-04 · **方法**: 离线数据探针 (深度可用性 / 相机内参FOV / 图像统计 / 样帧对比图), 不训练、不碰任何已有代码与文档 (探针脚本/产物在 gitignored `_xvla_innovation_probe/`)。
> **关联**: X-VLA 论文 arXiv 2510.10274 · `cross_embodiment_strategy.md` §0.2/§5.6 (相机 gap 主线) · `project_kai0_vis_camera_gap` (memory)。

---

## 0. 最终建议 (一句话, 经实测修正)

**主攻 = Direction A「跨相机感知适配」, 用 P1 外观增强 co-train + vis(D405) 感知监督修复抓取不准, 以 camera/sensor-conditioned soft prompt 为创新点。**
经离线探针验证, 原 "A+B 合体" 中的 **B (深度抓取) 降级** (wrist 深度未采集), **P2 纯 FOV 裁剪降级** (gap 主因是 appearance 不是 FOV)。

---

## 1. 我跑了哪些小实验 (认证证据)

| Probe | 做法 | 结果 | 对方向的影响 |
|---|---|---|---|
| **P1 深度可用性** | 读 `config/camera_depth_flags.py` + 扫数据 videos 流 | **wrist D405 depth = OFF** (`ENABLE_DEPTH_HAND_LEFT/RIGHT=False`, 因 USB 带宽/存储/"policy 不消费"); **仅 top_head(D435) 有 depth** | 🔴 **Direction B (D405 腕深抓取) 近期不可行** —— 需重启采集+解带宽 |
| **P2 内参/FOV** | 从 `config/intrinsics.yaml` 算 FOV | vis hand(D405) fx=393 → **78.3°×62.9°**; top_head(D435) fx=604 → 55.8°×43.3°; kai0 腕 D435 ~69° → **P2 center-crop=84%** (每边裁 ~8%) | P2 几何对齐**很小** |
| **P3 图像统计** | 抽 kai0 腕 vs vis 腕各 8 帧算亮度/对比/锐度 | kai0: bright135 contrast42 sharp29; **vis: bright119 contrast93 (2.2×) sharp66 (2.3×)** | 🔴 **appearance gap 大** (D405 全局快门+近焦 → 更锐更高对比) |
| **P4 样帧对比图** | kai0腕 \| vis腕RAW \| vis腕P2crop 拼图 (`_xvla_innovation_probe/wrist_gap_montage.png`) | 视觉上: kai 灰/软/浅布 vs vis 高对比/锐/**深布白桌**, 连 **gripper 硬件外观都不同**; **P2 crop 几乎无变化** | 🔴 **FOV 不是主因**; 主因是 appearance + 内容(布料颜色/夹爪外观/场景) |

> **核心认证结论**: kai0(D435)→vis(D405) 腕部 gap = **appearance 主导 (对比/锐度 2×+ + 布料颜色 + 夹爪外观), FOV 次要 (仅 8%)**。这解释了"纯 kai0 模型在 vis 抓取不准": 模型在 kai 的灰/软图上学的衣角定位, 到 vis 高对比/深色图上对不上。**→ 修复必须做 appearance 域适配, 而非单纯 FOV 裁剪或深度。**

---

## 2. 验证如何改变了原推荐

| 方向 | 原评 (directions 文档) | 实测发现 | 修正 |
|---|---|---|---|
| **A 跨相机感知适配** | ⭐⭐⭐ | gap 大且 appearance 主导, vis 数据(1940 ep)可做监督 | ✅ **升为唯一主攻** |
| **B 深度精确抓取** | ⭐⭐⭐ | **wrist depth 未采** (只 top_head) | ⬇️ 降级 (需重采, medium-term) |
| **P2 相机FOV对齐** | 先做诊断 | FOV 仅差 8%, montage 证明几乎无效 | ⬇️ 仅作 sanity, 不作解 |
| **P1 外观增强 co-train** | 主力之一 | 实测 appearance 是主 gap → P1 正中要害 | ✅ **升为主路径** |
| **C camera-conditioned prompt** | 中 | 同 robot 异 camera 是干净 testbed, 论文 soft prompt 只编码 robot | ✅ **作创新点并入 A** |
| **D 衣角 keypoint 辅助** | 中 | 低成本直接补 grasp 定位 | ✅ 次选 (可叠加) |
| E 推理加速 / F loss 自适应 | 低 | 与痛点无关 | 维持低/跳过 |

---

## 3. 最终方案 (Direction A 落地实验)

### 3.1 问题定义
模型在 vis(D405) 部署相机上**精确感知衣角并抓取**。kai0 提供操作技能, 但其感知绑死 D435; 真 gap 是 D405 的 appearance。

### 3.2 数据 (沿用已验证的健康路径)
- kai+vis **物理预合并单源** (绕开 broken datasets_yaml, 见 corrected Plan A) + per-source/合并 norm。
- vis 感知监督预算: vis_base/v3 **1940 ep / 2.53M frame** (D405)。

### 3.3 方法 (三个可叠加组件 + 创新点)
1. **P1 外观+几何增强** (主): 对图像加 **contrast/sharpness/brightness jitter** (量级对齐 P3 实测的 2× 差) + color jitter + RandomResizedCrop(scale 0.5-1.0 覆盖 8% FOV 差) → 逼视觉编码器 camera-robust。⭐ **关键是 appearance 增强, 实测证明比 FOV 重要。**
2. **vis D405 抓取监督** (主): vis 加权, 让感知学 D405 衣角定位。
3. **⭐ camera/sensor-conditioned soft prompt** (创新): X-VLA soft prompt 论文只编码 **robot embodiment**; 我们 kai/vis 是**同 robot 异 camera** → 把 prompt 拆成 **robot-prompt ⊕ sensor-prompt** (compositional)。**Ablation: robot-only prompt vs robot+sensor prompt** —— 干净回答"soft prompt 该编码 robot 还是 sensor"(论文 G1 未答, 可发表)。
4. (次选 D) **衣角 keypoint 辅助监督**: 加一个轻量 head 预测衣角/边 2D 点, 直接监督 grasp 定位 (论文 G2 grasp precision 无 metric)。

### 3.4 对照矩阵 (真机抓取精度为终判)
| 组 | init | 数据/方法 | 测什么 |
|---|---|---|---|
| B0 | pi05/xvla-base | vis-only | baseline (部署相机原生) |
| B1 | base | kai-only | 复现"跑通但抓不准" (motor OK perception 错) |
| A1 | base | 预合并 + **P1 外观增强** | 外观适配是否修抓取 |
| A2 | base | A1 + **camera-conditioned prompt** | sensor-prompt 增益 (创新点) |
| A3 | base | A1 + **衣角 keypoint 辅助** | grasp 定位辅助增益 |

### 3.5 评估 (填论文 gap)
- **新提 grasp precision metric** (论文 G2 缺): 抓取点与衣角真值的像素/3D 偏差 + 抓取成功率 + 进入下一阶段率。
- **cross-camera 协议**: train-on-D435 / deploy-on-D405 成功率掉多少, P1/prompt 修回多少。
- 真机为终判 (offline 只看健康 + 收敛)。

---

## 4. 认证 (GO / NO-GO)

| 项 | 结论 | 依据 |
|---|---|---|
| **Direction A 可做?** | ✅ **GO** | 数据 (vis 1940ep D405)、算力 (16GPU finetune)、资源 (dual-camera 真机 benchmark) 齐备; gap 性质已离线认证 (appearance 主导) |
| **B 深度抓取?** | ⏸️ **HOLD** | wrist depth 未采 → 需先重启 D405 wrist depth 采集 (解 USB 带宽) 再议; top_head D435 depth 可先用于场景级 |
| **P2 单独?** | ❌ **NO** | montage + 统计证明 FOV 仅 8%, 不是主 gap |
| **论文价值?** | ✅ 高 | camera/sensor-conditioned prompt + dual-camera cloth benchmark + grasp-precision metric = 填 X-VLA G1(多视角)+G2(grasp)+G7(failure) |

**不建议** (资源不匹配, 维持 directions 文档结论): 重做 0.9B/290K 预训练 · 纯架构替换 · 纯推理加速。

---

## 5. 执行 checklist (从认证到能跑)
- [ ] **S1** 预合并 kai+vis 单源 (含 sensor/camera 标记位) — 复用 corrected Plan A 的 merge 脚本思路
- [ ] **S2** 实现 P1 外观增强 (contrast/sharpness jitter 对齐 P3 实测分布 + RandomResizedCrop)
- [ ] **S3** (创新) compositional soft prompt = robot ⊕ sensor (改 soft_prompt_hub 索引 / 加 sensor embed)
- [ ] **S4** (次选) 衣角 keypoint 辅助 head + 标注/伪标注
- [ ] **S5** 训 B0/B1/A1/A2/A3, offline 健康闸门
- [ ] **S6** 真机 grasp precision + cross-camera 协议评估
- [ ] **S7** (条件) 重启 wrist D405 depth 采集 → 解锁 Direction B

---

## 附: probe 复现 (gitignored `_xvla_innovation_probe/`)
- FOV/crop 计算、图像统计、montage 生成脚本均在该目录 (不入 git)。
- 关键数: vis D405 wrist contrast 92.8 vs kai0 D435 41.9; sharpness 66 vs 29; P2 crop 0.844; wrist depth OFF。
