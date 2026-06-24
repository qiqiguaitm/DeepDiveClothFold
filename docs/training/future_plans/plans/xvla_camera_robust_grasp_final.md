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
- vis 感知监督预算: vis_base/v3 **1940 ep / 2.53M frame** (D405; ⚠️帧数为 pre-tailcap, v3 现含尾裁 Step 3 删~1.6%)。

### 3.3 方法 (三个可叠加组件 + 创新点)
1. **P1 外观+几何增强** (主): 对图像加 **contrast/sharpness/brightness jitter** (量级对齐 P3 实测的 2× 差) + color jitter + RandomResizedCrop(scale 0.5-1.0 覆盖 8% FOV 差) → 逼视觉编码器 camera-robust。⭐ **关键是 appearance 增强, 实测证明比 FOV 重要。**
2. **vis D405 抓取监督** (主): vis 加权, 让感知学 D405 衣角定位。
3. **⭐ camera/sensor-conditioned soft prompt** (创新): X-VLA soft prompt 论文只编码 **robot embodiment**; 我们 kai/vis 是**同 robot 异 camera** → 把 prompt 拆成 **robot-prompt ⊕ sensor-prompt** (compositional)。**Ablation: robot-only prompt vs robot+sensor prompt** —— 干净回答"soft prompt 该编码 robot 还是 sensor"(论文 G1 未答, 可发表)。
4. (次选 D) **衣角 keypoint 辅助监督**: 加一个轻量 head 预测衣角/边 2D 点, 直接监督 grasp 定位 (论文 G2 grasp precision 无 metric)。

### 3.4 对照矩阵 (真机抓取精度为终判)
| 组 | init | 数据/方法 | 测什么 |
|---|---|---|---|
| B0 | xvla-base | vis-only | baseline (部署相机原生) |
| B1 | base | kai-only | 复现"跑通但抓不准" (motor OK perception 错) |
| **A0 (skeptic baseline)** ⚠️ | base | 预合并, **vis 当独立 domain + vanilla X-VLA recipe** (原子 prompt + ColorJitter) | **X-VLA 现成机制够不够?** (必须设这组, 见 §3.6) |
| A1 | base | A0 + **P1 标定外观增强** (replace ColorJitter) | 外观适配是否修抓取 |
| **A2 (创新)** ⭐ | base | A1 + **compositional robot⊗sensor prompt** | sensor-prompt 解耦增益 |
| A3 | base | A1 + **衣角 keypoint 辅助** | grasp 定位辅助增益 |

### 3.5 评估 (填论文 gap)
- **新提 grasp precision metric** (论文 G2 缺): 抓取点与衣角真值的像素/3D 偏差 + 抓取成功率 + 进入下一阶段率。
- **cross-camera 协议**: train-on-D435 / deploy-on-D405 成功率掉多少, P1/prompt 修回多少。
- 真机为终判 (offline 只看健康 + 收敛)。

### 3.6 ⭐ 创新 vs X-VLA: delta 与必须打赢的 baseline

**X-VLA 如何处理异 camera (背景)**: 不专门处理, 把相机当 domain 异质性塞进 **per-domain 原子 soft prompt** (robot+camera+env 捆死) + ColorJitter + 共享 VLM (论文自承 multi-view 弱)。详见 [`../../analysis/xvla_innovation_directions.md`](../../analysis/xvla_innovation_directions.md) §1.1。

**我们的 delta (按新颖度诚实分级)**:
| # | 改动 | 相对 X-VLA 的新颖度 | 说明 |
|---|---|---|---|
| ① **compositional prompt** = robot ⊗ sensor 双因子 | ⭐ **真创新** | X-VLA prompt 原子化 (robot+camera 捆死); 拆成 robot(kai+vis 共享, 大数据) ⊗ sensor(只学相机 delta, 小数据) → "同 robot 异 camera"**数据高效迁移**, 扩展 §5.3 "cross-embodiment similarities" 到 cross-**camera** within embodiment |
| ② 标定外观增强 (P1) | 半增量 | X-VLA 只 ColorJitter(0.2); 我们按实测 2× contrast/sharpness 差标定 + RandomResizedCrop。工程改进, 非论文级 |
| ③ 共享 VLM 主动适配 D405 | 半创新 | 论文 VLM multi-view 弱却没解; 我们让视觉编码器在 vis 主动适配 |
| ④ 受控 dual-camera 叠衣 benchmark + grasp-precision metric | 贡献物 | X-VLA "domain" 把相机和别的混在一起, 无受控 camera ablation, 无 grasp metric (G2) |

> ⚠️ **诚实红线 (写死进实验)**: 怀疑者会说"X-VLA 把 vis 当一个 domain + 多点 ColorJitter 不就行?" —— 所以 **A0 (vis-as-domain vanilla X-VLA) 是必设 baseline**。**只有 A2 (compositional) > A1 (外观增强) > A0 (X-VLA 原生) 才证明 delta 真有增益**; 若 A0 已够好, 说明 X-VLA 原机制足够, 创新①不成立, 据实记录。

### 3.7 compositional prompt 的实现 (① 的代码改法)
- X-VLA domain 身份载体 (已实测): `soft_prompt_hub.weight (num_domains, 32*1024)` + `action_encoder/decoder` 的 DomainLinear `.fc/.bias (num_domains, ·)`。
- **改法**: 把单一 `domain_id` 索引换成 **(robot_id, sensor_id) 两个 embedding 相加/拼接**:
  - 新增 `robot_prompt_hub = Embed(num_robots, 32*1024)` + `sensor_prompt_hub = Embed(num_sensors, 32*1024)`, soft prompt = robot[r] + sensor[s] (或 concat 后投影)。
  - kai0=(robot=agilex, sensor=D435), vis=(robot=agilex, **sensor=D405**) → robot 因子共享, 仅 sensor 不同。
  - 部署固定 (agilex, D405)。
- ⚠️ 需核对参数 path 命名 + 兼容 X-VLA-Pt 权重加载 (robot 因子可用原 agilex domain 槽 warm-init, 见 [`xvla_domain_slot_init_ablation.md`](xvla_domain_slot_init_ablation.md))。

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

## 6. 是否需要从头预训练 + compute 预算

**结论: 不需要从头预训练。** 直接微调开源 **X-VLA-Pt** (官方就是这么设计的; 论文证明 LoRA 9M(1%)≈全量, 10 demo→91%)。本方案 (P1 + compositional prompt + keypoint) **全是微调 / 小改架构级**, 无一需从头预训练。

| 路线 | 卡 | 时间 | 建议 | 备注 |
|---|---|---|---|---|
| **从头预训练 (像 X-VLA)** | 64 GPU | 1–2 周 + 数据后勤 | ❌ **不做** | 估 **~1–2 万 GPU-h** (官方未披露; 锚 OpenVLA-7B=64×A100×14d≈2.15万 A100-h, X-VLA 0.9B 更小同量级)。还要下载/清洗/存几十 TB 跨本体数据 (Droid/RoboMind/Agibot)。**赢不了它的数据规模, 无收益** |
| **continual / 域适应续训** | 16–32 GPU | 几天 | ⚠️ 仅当加大量新数据/新模态 | = Track X 的 X3 跑法 |
| **微调 X-VLA-Pt (本方案主路径)** ✅ | **16 GPU** | **~半天/次** | ✅ **走这个** | A0/A1/A2/A3 各一次, 全在 uc01/02/03 + Volc 资源内 |

> **锚点对照**: 你现有 X-VLA 微调 ≈ 16 GPU × ~12h(30k step)/次; **从头预训练 ≈ 一次微调的 50–100×**。把卡省下来多跑几组对照 + 真机评估, 远比烧 1–2 万 GPU-h 重训一个赢不了的 base 划算。
> ⚠️ compositional prompt(§3.7)新增的 robot/sensor 因子是**少量新参数**, 从 X-VLA-Pt warm-init (robot 因子借 agilex 槽) 后**微调即可**, 不触发预训练需求。

---

## 附: probe 复现 (gitignored `_xvla_innovation_probe/`)
- FOV/crop 计算、图像统计、montage 生成脚本均在该目录 (不入 git)。
- 关键数: vis D405 wrist contrast 92.8 vs kai0 D435 41.9; sharpness 66 vs 29; P2 crop 0.844; wrist depth OFF。
