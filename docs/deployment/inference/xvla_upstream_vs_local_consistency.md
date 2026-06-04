# X-VLA 上游 (DeepDive-XVLA) vs 本地一致性分析

> **目的**: 把上游官方 repo `qiqiguaitm/DeepDive-XVLA` (fork of `2toinf/X-VLA`) 与本地 `deepdive_kai0` 的 XVLA 训练/部署管线逐契约对比, 找出不一致与可参考点, 为 Track X 真机部署兜底正确性。
> **建立**: 2026-06-04 · **方法**: clone 上游 (blob 过滤, 留 `workspace/DeepDive-XVLA`) + 逐文件读 + 验证本地 lerobot XVLAPolicy 实际行为。
> **关联**:
> - 部署 bring-up: [`xvla_inference_bringup.md`](xvla_inference_bringup.md) (R1–R4 契约)
> - 训练侧 curriculum: [`../../training/future_plans/plans/xvla_track_x_curriculum.md`](../../training/future_plans/plans/xvla_track_x_curriculum.md)
> - 上游源: https://github.com/qiqiguaitm/DeepDive-XVLA (本地 clone `/vePFS/tim/workspace/DeepDive-XVLA`, HEAD d151487)

---

## 0. TL;DR

| 类别 | 项 | 结论 |
|---|---|---|
| ✅ **已对齐 (别动)** | rot6d interleaved / gripper 二值化 / 20D 布局 / ImageNet 归一 / ColorJitter / fp32 / 加权采样 / use_proprio / chunk30·domains30·ee6d | 本地正确镜像上游 |
| ⚠️ **真实差异 (可参考)** | **I2** 图像 letterbox vs 拉伸 · **I3** domain_id 冷启 vs 复用已训 agilex 槽 · **I4** EE 来源 (录制 vs PiperFK link6) · **I5** 冗余双 resize (256→224) | 见 §2 |
| ❌ **已证伪 (别再担心)** | **I1** "本地 256 致视觉 OOD" — 实测 `_prepare_images` 把 256 统一降到 224, 模型实际见 224 | 见 §3 |
| 📌 **本地文档 stale** | bringup R4 说"不做 ImageNet 归一" — 与上游 + 本地 P0 代码全矛盾 | 见 §4 |

---

## 1. ✅ 已对齐的契约 (逐位核对, 这些是正确性基石, 勿改)

| 契约 | 上游 (file:line) | 本地 (file:line) | 核对 |
|---|---|---|---|
| **Rot6D = interleaved** `[r00,r01,r10,r11,r20,r21]` | `datasets/utils.py:53` `R.from_quat(q).as_matrix()[...,:,:2].reshape(...,6)` | `train_scripts/xvla/data/joint_to_ee6d.py:41` `R[:,:2].flatten()` + `kai0/scripts/xvla_action_codec.py:26` | ✅ 编解码逐位一致 |
| **gripper 二值化** `raw*50<1.0 → 闭(1)` | `datasets/domain_handler/real_world.py:42` `(eef[:,7:8]*50<1.0)` | `joint_to_ee6d.py:48` `gripper*50.0<1.0` | ✅ 完全相同 |
| **20D EE6D 布局** `xyz(3)+rot6d(6)+grip(1)` ×2 臂 | `real_world.py:33` | `joint_to_ee6d.py` | ✅ |
| **ImageNet 归一** `(0.485,0.456,0.406)/(0.229,0.224,0.225)` | 训练 `datasets/dataset.py:68` + 部署 `models/processing_xvla.py` | 训练 `multi_domain_dataset.py:30,177` + 部署 `serve_policy_xvla.py:80,258` (P0, 2026-06-01) | ✅ P0 修复后对齐 |
| **ColorJitter(0.2,0.2,0.2,hue=0)** | `dataset.py:65` | `multi_domain_dataset.py:106` | ✅ 完全相同 |
| **加权多域采样 (非复制实例)** | `dataset.py:106` `random.choices(weights=ws)` | `multi_domain_dataset.py:214` `WeightedRandomSampler` + `xvla_train.py` `weight=N` | ✅ 同范式 |
| **use_proprio=True** (吃 20D EE6D 当前态) | `models/configuration_xvla.py:53` + `modeling_xvla.py:147` forward 含 `proprio` | serve `joint_to_ee6d_row` → 20D state | ✅ 结构一致 |
| **chunk=30 / num_domains=30 / action_mode=ee6d / fp32** | `configuration_xvla.py:51-52` + `deploy.py:79` | `lerobot_base/config.json` + serve `--dtype float32` | ✅ |

> ⭐ **正面发现**: 本地 XVLA 侧 `weight=7.0` 是 **WeightedRandomSampler 权重**, 不是 ConcatDataset 复制实例 —— 与上游 `random.choices` 一致, 也**和 pi05/kai0 那条崩掉的 datasets_yaml 复制路完全不同**。XVLA 的加权采样可作 pi05 侧的修复参考 (corrected Plan A 方案 C)。

---

## 2. ⚠️ 真实差异 (可参考 / 可能要改)

### I2. 图像几何: 本地 **letterbox 补边**, 上游 **直接拉伸** ⭐ (真正的图像差异)
- **上游** `dataset.py:64`: `transforms.Resize((224,224))` —— **直接缩放到方形, 改变长宽比, 无补边**。xvla-base 在"拉伸 224"图上预训练。
- **本地**: `resize_pad`/`resize_with_pad` —— **保持长宽比 + 0 填充 (黑边)**。模型 (经 `_prepare_images`, modeling_xvla.py:317) 最终见 **letterbox 224**。
- **差异本质**: 模型 pretrain 见"拉伸图", 本地喂"带黑边的 letterbox 图" → **几何分布偏离 pretrain** (有效物体尺度 + 黑边都不同)。letterbox 本身更不失真, 但偏离了 base 见过的分布。
- **可参考**: 想最大化复用 pretrain 先验 → 改用上游 `Resize((224,224))` 直接拉伸; 或保留 letterbox 但知晓这是一个 pretrain-domain 偏离源 (与当初 ImageNet-norm 缺失同类, 只是程度轻)。**真机若仍有轻微视觉 drift, 这是候选因子。**

### I5. 冗余双 resize (256 → 224) (新发现, 清理项)
- **本地** `multi_domain_dataset.py:91` 先把主相机 `resize_pad` 到 **256**, 做 ImageNet 归一, 然后 lerobot `_prepare_images` (config `resize_imgs_with_padding=[224,224]`, `lerobot_base/config.json:196`) 又把它 **letterbox 降到 224**。
- → 主相机经历 **raw→256(letterbox)→224(再 letterbox)** 双重缩放+嵌套补边, 轻微 lossy 且浪费; 256 这一级**实际是死的** (最终都被压到 224)。
- **可参考**: dataset 直接产 224 (与 `resize_imgs_with_padding` 对齐), 省一次缩放、避免嵌套黑边。**注意: serve 与 train 都过 `_prepare_images`→224, 二者一致, 非 train/serve 失配** (已验证)。

### I3. domain_id: 本地冷启 **19/20/21**, 上游 agilex 有**已训槽位** ⭐ (最有价值的可参考)
- **上游** `datasets/domain_config.py`: 真机 agilex 类落在**已预训练**的槽: `AIR-AGILEX=10`, `AIR-AGILEX-HQ=5`, `robomind-agilex=16` (这些 domain soft-prompt 在 xvla-base 里有权重)。
- **本地** `xvla_train.py:37-39`: kai=19 / **vis=20** / xvla_sf=21 —— 在 base 里**随机初始化、从未训练**的冷槽位。
- **可参考 (建议试)**: vis(20) 的 domain embedding 从零学。可试 **用一个已训 agilex 槽 (10 或 16) 初始化 vis 的 soft-prompt**, 借 pretrain 的 agilex 先验, 而非冷启 20 → 小数据下可能加速收敛 / 提升泛化。低成本 ablation。

### I4. EE 来源: 上游读**录制 eef**, 本地从关节算 **PiperFK link6**
- **上游** `real_world.py:41`: EE 来自 hdf5 `/observations/eef_quaternion` (机器录制值)。
- **本地**: LeRobot kai/vis 只有关节 → 必须 `joint_to_ee6d` 算 EE, 选 **link6** (CalFK `result[-1]`)。
- 这是**必要适配, 非 bug**。bringup R1 的硬约束 (部署 IK 必须解到 link6, 否则 13.58cm 偏差) 正源于此。
- **可参考**: 若将来录制数据带 eef 字段, 须核对其 frame 与 PiperFK link6 是否一致再混用。

---

## 3. ❌ 已证伪 (记录以免重复踩坑)

### I1. "本地 main 256 → 视觉 OOD" — **证伪**
- 初看: `multi_domain_dataset.py:91` `image_size_main=256` vs 上游统一 224, 疑似喂 256 给 224-pretrain 的 Florence2 → OOD。
- **验证**: 本地走 lerobot `XVLAPolicy._prepare_images` (modeling_xvla.py:317-323), `resize_imgs_with_padding=[224,224]` (config.json:196) → **所有图统一 letterbox 到 224 再 stack 喂模型**。
- **结论**: 模型**实际输入 224**, 与上游/pretrain 一致。256 只是中间量 (见 I5 冗余)。**I1 不成立, 不是 OOD 来源。**

---

## 4. 📌 本地文档需更新

### bringup `xvla_inference_bringup.md` 的 **R4 已 stale**
- 文档 (2026-05-31, §0★/R4/架构图) 反复强调"图像 /255 [0,1], **不做 ImageNet 归一**"。
- 但 **P0 修复 (2026-06-01)** 已加 ImageNet 归一 (`multi_domain_dataset.py:177` + `serve_policy_xvla.py:258`), 且**与上游一致** (上游训练 `dataset.py:68` + 部署 processor 都做)。
- → 文档 R4 与现行代码 + 上游全部矛盾, **应更新**: 训练/部署**都做** ImageNet 归一, 这是对齐上游 pretrain 的必需项 (P0 之前缺失正是真机抖动根因之一)。

---

## 5. 行动清单 (按价值排序)

| 优先 | 项 | 动作 |
|---|---|---|
| 🟡 中 | **I3** domain_id | ablation: vis soft-prompt 用上游已训 agilex 槽 (10/16) init vs 冷启 20, 比真机 + 收敛 |
| 🟡 中 | **I2** letterbox vs 拉伸 | 若真机有视觉 drift, 试改 dataset 用上游式 `Resize((224,224))` 直接拉伸对齐 pretrain |
| 🟢 低 | **I5** 双 resize | dataset 直接产 224, 去掉 256 中间级 (清理 + 省算力, 不改模型输入) |
| 🟢 低 | **文档** | 更新 bringup R4 (ImageNet 归一已做且必需) |
| ✅ 复用 | 加权采样 | XVLA 的 WeightedRandomSampler 范式回填 pi05/kai0 (修 datasets_yaml 复制坑) |

> ⚠️ **真机为终判**: 以上 I2/I3 是否真影响成功率/抖动, 离线无法定论, 需真机 A/B。I4/I5 是正确性/清理, 不依赖真机。

---

## 附: 上游 repo 关键文件 (本地 clone `/vePFS/tim/workspace/DeepDive-XVLA`)

| 功能 | 上游路径 | 本地对应 |
|---|---|---|
| 图像预处理 + 归一 | `datasets/dataset.py:63-70` | `train_scripts/xvla/data/multi_domain_dataset.py` |
| EE6D/gripper/rot6d 转换 | `datasets/domain_handler/real_world.py` + `datasets/utils.py:53` | `train_scripts/xvla/data/joint_to_ee6d.py` + `kai0/scripts/xvla_action_codec.py` |
| 加权采样 | `datasets/dataset.py:100-111` | `multi_domain_dataset.py:214` |
| domain id 表 | `datasets/domain_config.py` | `train_scripts/xvla/launch/xvla_train.py` (本地重映射) |
| 模型 forward (proprio/image) | `models/modeling_xvla.py:141,175` | lerobot `policies/xvla/modeling_xvla.py` (X-VLA-env) |
| 部署入口 | `deploy.py` (`XVLA.from_pretrained` + `XVLAProcessor`) | `kai0/scripts/serve_policy_xvla.py` (lerobot `XVLAPolicy`) |
