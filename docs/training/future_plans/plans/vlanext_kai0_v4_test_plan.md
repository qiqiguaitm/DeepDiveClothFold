# 用自有数据集训练 VLANeXt —— 可行性测试 plan

> **建立**: 2026-06-27
> **目的**: 把开源 **VLANeXt**(*Recipes for Building Strong VLA Models*,arXiv 2602.18532,`github.com/DravenALG/VLANeXt`)拉进本仓,**用我们的 Task_A 叠衣数据集(v4 base+dagger)在其上跑一次训练做可行性测试** —— 验证这套 Qwen3-VL-2B + 流匹配动作头的 VLA 配方能否吃下我们的真机双臂可变形数据并收敛。
> **状态**: 📋 **plan 草拟**(repo 已 clone 到 `external/VLANeXt`)。**本文件只做规划,不直接执行训练。**
> **定位**: 这是**外部模型的探索性测试**(不是主线 pi05/AWBC),目标是低成本摸清 VLANeXt 在我们数据上的可训练性 + 与 pi05 的对照价值。
> ⚠️ **铁律**(本项目):真机为终判;VLA 报告先看 val MAE(不是 train loss);idle 轨迹 MAE 反指。

---

## 0. VLANeXt 是什么(已查 repo + 论文 + config)

| 维度 | 事实 |
|---|---|
| 论文 | *VLANeXt: Recipes for Building Strong VLA Models*,arXiv **2602.18532**;系统消融 VLA 设计空间(foundational / perception / action modeling 三维,12 条 findings)|
| **backbone VLM** | **`Qwen/Qwen3-VL-2B-Instruct`**(finetune,不冻)|
| **视觉编码器** | **`google/siglip2-base-patch16-256`**(256 分辨率,patch16)|
| **动作头** | **扩散 / 流匹配**(`loss_type=diffusion`,`scheduler_type=flow_match`,train 1000 步 / infer 10 步);policy transformer(depth 29 / hidden 1024 / heads 16);`num_queries=16`,soft condition,transformer connector |
| **动作维度** | `action_dim` **可配**(DROID/LIBERO 默认 **7** = 6D pose + 1 gripper,**单臂**)|
| **时序** | `history_len=8`(历史观测帧)/ `future_len=8`(动作 chunk = 预测未来 8 步)|
| **多视角** | `view_mode=multi`:主相机 `image` + 腕相机 `wrist_image`(loader 里可扩到更多腕视角)|
| **本体输入** | `use_proprio_input_vlm=true`(proprioception 喂进 VLM)|
| **数据格式** | **TFDS / RLDS**(`tfds.builder_from_directory`);LIBERO 用 `openvla/modified_libero_rlds`,DROID 用 OXE RLDS |
| 训练 | AdamW(wd 0.01)· batch **256** · LR **1e-4** · warmup 500(libero)/5000(droid)· DROID 100k step / LIBERO 10k step · 数据增强(random resized crop + 色彩抖动)· `gradient_checkpointing` |
| 评测 | **LIBERO / LIBERO-Plus(仿真)** + 真机;脚本 `scripts/libero_bench_eval.py` 等 |
| 代码 | `scripts/train.py` · `src/datasets/{droid,libero}_act.py` · `src/models/VLANeXt.py`(connector/encoder/generator/policies)· config `config/*.yaml` · 设计空间教程 `DESIGN_SPACE.md` |

**环境**:`conda create -n VLANeXt python=3.10` + `torch==2.4.0/cu124` + `requirements.txt` + `flash-attn` + ffmpeg。

---

## 1. 我们的数据 vs VLANeXt 期望(差异 = 工作量来源)

| 维度 | VLANeXt 默认(DROID/LIBERO)| **我们的数据(Task_A v4)** | 差异处理 |
|---|---|---|---|
| 本体 | 单臂 7-DoF | **双臂 14-DoF**(2×[6 关节 + 1 夹爪])| `action_dim` 7→**14**;⚠️ VLANeXt 官方只验过单臂,双臂是**扩展未验证** |
| 动作表示 | delta pose 6D + gripper | **absolute joint 14D**(v4:夹爪 action≠state 取主臂指令)| 扩散动作头是 generic 的 → 只需 **per-dim 重算 normalization min/max**,不需改头 |
| 相机 | main `image` + 单 `wrist_image`(2 路)| **3 路**:`top_head` / `hand_left` / `hand_right` | loader 扩到 3 视角(main=top_head,wrist=hand_left,加 `second_wrist`=hand_right)|
| proprio | 8D(libero)| **14D** | 改 state 维度 + 夹爪抽取索引(我们夹爪在 **6 和 13**)|
| 语言 | per-task instruction | 固定 `"Flatten and fold the cloth."` | 直接填 |
| 格式 | **TFDS/RLDS** | **LeRobot v2**(parquet + mp4)| ⚠️ **必须转换**(本 plan 最大工作量,§3)|
| 终止信号 | `reward`(loader **只保留 reward=1 结尾的轨迹**)| LeRobot 无 reward | ⚠️ **转换时务必给每条轨迹末帧写 reward=1**,否则全被丢弃 |

**数据量**(已落地,见 [[project_tos_sync_paused_restructure]]):v4 base 13 日期/1207ep/1.35M 帧 + v4 dagger 12 日期/789ep/1.02M 帧 ≈ **1996ep / 2.37M 帧**。

---

## 1.5 action_dim=14 与 3 相机:具体怎么改(已读源码确认)

> ⭐ **结论:这两件都不是"改模型",而是改 config + 数据 loader。模型本身天然支持任意 action_dim 和任意视角数。**

### ① action_dim 7→14 —— 改 1 行 config + loader 出 14 维
- **模型 100% 由 config 驱动**:`scripts/train.py:358,373` 用 `action_dim=config['model']['action_dim']` 建模型;所有层(`encoder.py:7,52` input/output proj、`policies.py` input_proj/final_layer、`VLANeXt.py`)都按这个数 size。→ **config 里 `action_dim: 14` 一改,整个模型自动变 14 维,零模型代码改动。**
- **真正要做的在数据 loader**(我们的 `kai0_act.py`):
  1. 输出 `future_actions` / `history_actions` / `proprioception` 都是 **14 维**(不是 7)。
  2. **替换 7 维的 `action_min/action_max`**(libero 在 `libero_act.py:51-69` 按 suite 硬编码 7 维)→ 用**对 v4 全集 per-dim 统计的 14 维 min/max**(归一化是 `2*(a-min)/(max-min)-1`,必须 14 维 bound)。
  3. **删掉 libero 的夹爪专用处理**(`libero_act.py:135` `gripper_qpos=raw_state[:,6:8]` + 0.04 宽度逻辑,是 libero 硬件特定)。我们 v4 state 已是 14 维关节,**两个夹爪在 idx 6 和 13**,直接透传 / per-dim 归一化即可。
- ⚠️ **唯一硬约束:必须用 `loss_type: diffusion`(flow_match,默认就是)**。因为**分类头**假设"最后一维 = 唯一的夹爪"(`policies.py:548 pose_dim=action_dim-1`、`VLANeXt.py:694 pose_logits=[...:action_dim-1]`)—— 双臂有 **2 个夹爪(idx 6、13)**,不在末位 → 分类头会算错。扩散头对 14 维**一视同仁**,正确。**别切到分类头。**

### ② 2 相机→3 相机 —— 改 loader 出第 3 路 + collate 加 3 行
- **模型已动态支持任意视角数**:`VLANeXt.py:449` `num_views = image_embeds.shape[0] // B`(从堆叠的图像 embedding 反推视角数)→ **零模型改动。**
- "2 视角"只硬编码在**两处**:
  1. **数据 loader**(`libero_act.py:83-84`):`main_key="image"` / `wrist_key="wrist_image"`,只产出 `image` + `image_wrist`。→ 在 `kai0_act.py` 里读 3 路:`top_head→image`、`hand_left→image_wrist`、`hand_right→image_second_wrist`(新键)。
  2. **train.py collate**(`scripts/train.py:206-214`,Qwen image 分支)只拼 im0+im1。→ 加第 3 路:
     ```python
     im2 = self._augment_frames_uint8(sample["image_second_wrist"])
     content.extend([{"type":"image","image":im0},{"type":"image","image":im1},{"type":"image","image":im2}])
     images.extend([im0, im1, im2])
     ```
- 处理器(Qwen)会把 3 张图都编码,模型 `num_views` 自动 =3。**就这些。**

> 小结:**TODO① = config 改 1 行 + loader 出 14 维动作&14 维 norm;TODO② = loader 出第 3 路 + collate 加 3 行。** 都不碰模型主体。真正的大头还是 §3 的 LeRobot→数据接口转换。

---

## 2. 测试范围(先小后大,控制成本)

> "for test" = 先验证**管线能跑通 + 能收敛**,不追求 SOTA。分两档:

- **T0 烟雾测试(必做,先行)**:取 **v4 base 1~2 个日期(~100ep)**,转换 → 1~2 卡跑 **~2k step**,验证:数据 loader 不报错、forward/backward 通、loss 下降、能存 ckpt。**通过才进 T1。**
- **T1 正式测试**:**全 v4 base+dagger(~1996ep)**,8 卡跑 **~30k step**(VLANeXt LIBERO finetune 才 10k;我们数据更难 + 双臂,先 30k 看 val MAE 曲线再决定续不续)。

⚠️ 与 pi05 不同,VLANeXt 是 **PyTorch + RLDS + Qwen3-VL** 的独立栈,**不复用** kai0/openpi 的任何代码 —— 整条链路要新搭。

---

## 3. 数据转换 LeRobot v4 → TFDS/RLDS(核心工作)

VLANeXt 的 `src/datasets/libero_act.py` 用 `tfds.builder_from_directory()` 读 RLDS。每个 step 期望字段:

| 字段 | 含义 | 我们填什么 |
|---|---|---|
| `image` | 主相机 RGB(uint8)| `top_head` 解码帧 |
| `wrist_image` | 腕相机 | `hand_left` |
| `second_wrist`(扩展)| 第二腕相机 | `hand_right` |
| `state` | proprioception | 14D 关节状态 |
| `action` | 动作 | **14D**(v4 absolute,夹爪取主臂)|
| `language_instruction` | 任务文本 | `"Flatten and fold the cloth."` |
| `reward` | 终止 | **末帧=1.0,其余=0.0**(⚠️ loader 靠它过滤)|

**两条转换路线(二选一)**:
- **路线 A(推荐,改动小):写 TFDS dataset builder** `kai0_fold_rlds`:遍历 v4 LeRobot(parquet 读 state/action,mp4 解码 3 路图像),按 episode 组装 RLDS `steps`,末帧 reward=1,`tfds build` 落 TFDS 目录。参照 `openvla/modified_libero_rlds` 的 builder 结构 + RLDS dataset builder 模板。
- **路线 B(更省,但要改 VLANeXt):写 LeRobot 直读 loader** `src/datasets/kai0_act.py`,绕过 RLDS 直接读 parquet+mp4 → 返回同样的 sample dict。省掉 TB 级 TFDS 中间产物,但要吃透 VLANeXt 的 sample 接口(history/future chunk 逻辑)。

> 建议:**T0 用路线 B**(小数据、快迭代、不落 TFDS);**T1 若要忠实复刻官方管线再考虑路线 A**。

**无论哪条路,都要做的适配**(参照 libero_act.py 的 §"Integrating a Custom Dataset"):
1. `action_min/max` 改成 **v4 14-DoF 的 per-dim 统计**(扫全集算,别用 libero 的 7D bound)。
2. 相机键扩到 3 路(main + wrist + second_wrist)。
3. `state` 维度 8→14;夹爪抽取索引改成 **[6, 13]**。
4. ⚠️ **v4 铁律照搬**:动作 normalization 必须对 v4 重算(夹爪 action≠state);**夹爪不裁**。

---

## 4. config(新建 `config/kai0_fold_train_config.yaml`)

克隆 `config/libero_train_config.yaml`,改:
```yaml
data:
  dataset_name: "kai0_fold"          # 触发 §3 的自定义 loader / builder
  data_root: "<v4 转换后路径>"
  history_len: 8
  future_len: 8                       # 动作 chunk;若要对齐 pi05 horizon 可调
  view_mode: "multi"                  # 3 视角(loader 内扩 second_wrist)
  batch_size: 256                     # 8 卡;显存不够先降 128
model:
  lmm_path: "Qwen/Qwen3-VL-2B-Instruct"
  vision_encoder_path: "google/siglip2-base-patch16-256"
  action_dim: 14                      # ← 双臂(7→14)
  loss_type: "diffusion"
  scheduler_type: "flow_match"
  use_proprio_input_vlm: true
train:
  learning_rate: 1.0e-4
  warmup_steps: 500
  distributed: true                   # 8 卡 torchrun
  pretrained_checkpoint: ""           # 可填 VLANeXt 官方 DROID 预训练 ckpt 做暖启(若放出)
project:
  save_interval: 2000
  max_steps: 30000                    # T1;T0 用 2000
```

⚠️ **权重就位**:`Qwen3-VL-2B-Instruct` + `siglip2-base-patch16-256` 需从 HF 下载(集群离线则先缓存,`HF_HUB_OFFLINE=1`)。

---

## 5. 训练 + 评测

**训练命令**(8 卡):
```bash
conda activate VLANeXt
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29505 \
  -m scripts.train --config config/kai0_fold_train_config.yaml
```
- **集群**:可走 cnbj Robot-North-H20 8 卡闲时(同 pi05 习惯,`submit-training-job` skill);或 gf0 本地 2 卡先跑 T0 烟雾。

**评测**:
- ⚠️ **LIBERO/LIBERO-Plus 评测脚本是仿真,对叠衣无意义,不用**。
- **Tier 1 offline**:v4 留出 val 逐 ckpt **val MAE**(整体 + 夹爪维单列)+ loss 收敛曲线。
- **Tier 3 真机(终判)**:best ckpt 部署叠衣,看成功率 + 夹持稳定性。
- **对照**:同数据(v4)的 pi05 AWBC([`pi05_v4_awbc_from_paligemma_plan.md`](pi05_v4_awbc_from_paligemma_plan.md))→ 比 "VLANeXt(Qwen3-VL+流匹配)vs pi05(PaliGemma+流匹配)" 在同一叠衣数据上的真机表现。

---

## 6. 落地步骤
1. ✅ **clone repo** → `external/VLANeXt`(本轮已做)。
2. **建 env** `VLANeXt`(conda + torch2.4/cu124 + flash-attn);下载 Qwen3-VL-2B + SigLIP2 权重(集群离线预缓存)。
3. **读透接口**:`src/datasets/libero_act.py`(sample 字段 + history/future chunk)+ `src/models/VLANeXt.py`(action_dim/多视角接入点)+ `DESIGN_SPACE.md`。
4. **T0**:写 `kai0_act.py`(路线 B)读 v4 base 1~2 日期 → 算 14D action min/max → `kai0_fold_train_config.yaml`(action_dim=14, 3 视角)→ **gf0 2 卡 ~2k step 烟雾**(loss 下降 + 存 ckpt)。
5. **T1**:扩到全 v4 base+dagger → 8 卡 ~30k step → val MAE 曲线 → 决定续不续。
6. **eval**:offline val MAE → 真机(对照 pi05 v4 AWBC)。
7. 回填 results + 更新 master history。

---

## 7. 风险 / 注意
- **双臂未验证**:VLANeXt 官方只在单臂 7-DoF(DROID/LIBERO)上验证;14-DoF 双臂是扩展 —— 动作头维度可改但**配方(LR/chunk/normalization)是否仍最优未知**,这正是"test"要答的。
- **reward=1 过滤陷阱**:转换时漏写末帧 reward=1 → 轨迹全被 loader 丢 → "0 样本"假象。**T0 第一件事就核验样本数 > 0**。
- **3 视角接入**:官方默认 main+wrist 两路;第三路 `hand_right` 需在 loader + 模型 embed 处确认能吃(看 `VLANeXt.py` 视觉接入点)。
- **RLDS/TFDS 重**:路线 A 会产生 TB 级中间产物;路线 B 省盘但要改 VLANeXt 数据栈。先 B 后 A。
- **动作语义**:官方 delta-pose,我们 absolute-joint;扩散头本身不挑,但**normalization 必须 per-dim 对 v4 重算**(夹爪 action≠state),否则静默训坏。
- **权重下载**:Qwen3-VL-2B / SigLIP2 走 HF,集群离线需预缓存(本机出口限速,大文件用 gf3 aria2c 见 [[reference_gf3_fast_download]])。
- **独立栈成本**:不复用 openpi 任何代码,env + 数据 + eval 全新搭 → 比 pi05 内部实验贵;故先 T0 小成本验证。

---

## 关联
- repo:`external/VLANeXt`(`github.com/DravenALG/VLANeXt`)· 论文 arXiv 2602.18532 · 主页 dravenalg.github.io/VLANeXt
- 关键源:`src/datasets/libero_act.py`(数据接口模板)· `src/models/VLANeXt.py` · `config/libero_train_config.yaml`(finetune config 模板)· `DESIGN_SPACE.md`(12 findings 教程)
- 数据:`kai0/data/Task_A/vis_base/v4`(13)+ `vis_dagger/v4`(12);v4 框架背景见 [[project_tos_sync_paused_restructure]]
- 对照(同 v4 数据的 pi05 路线):[`pi05_v4_awbc_from_paligemma_plan.md`](pi05_v4_awbc_from_paligemma_plan.md) · [`pi05_v4_awbc_validation_plan.md`](pi05_v4_awbc_validation_plan.md)
- LeRobot→RLDS 参考:`openvla/modified_libero_rlds`(HF dataset)
