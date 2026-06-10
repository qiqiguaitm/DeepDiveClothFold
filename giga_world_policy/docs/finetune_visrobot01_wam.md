# 基于 kai0 数据微调 GigaWorld-Policy(WAM)——目标本体 visrobot01 定稿方案

> 目标:用 kai0 的叠衣服数据,在 GigaWorld-Policy 的 World-Action Model(Wan2.2-TI2V-5B backbone)上微调,**最终在 visrobot01 上测试/部署**。
> 任务文本统一:`"Flatten and fold the cloth."`
> 本文档为落地定稿,所有路径/参数均已对照真实代码核实。

---

## 1. 数据全景(已核实)

| 本体 | 数据集 | 结构 | episodes | 视频编码 | fps | 维度 | robot_type |
|------|--------|------|---------|---------|-----|------|-----------|
| **visrobot01**(目标) | `vis_base/v2/*`(20 子集)+ `vis_dagger/*`(3 子集) | 每个日期目录是一个独立 LeRobot 集 | **2101** | h264 | 30 | 14 | agilex |
| **kairobot01**(辅助域) | `kai0_base` + `kai0_dagger` | 单一 LeRobot 集 | **6512** | av1 / h264 | 30 | 14 | agilex |

- 数据根:`/home/tim/workspace/deepdive_kai0/kai0/data/Task_A/`(= `/vePFS/...` 同一份)
- 两套都是双臂 piper,相机 key 一致:`observation.images.{top_head,hand_left,hand_right}`,480×640。
- 两套 rig 差异:**相机外参、臂间距、相机配置不同** → 视觉分布 + 动作几何不同,但**关节动作空间语义相同**(6 左臂 + 6 右臂 + 2 夹爪,index 6/13 为夹爪)。

---

## 2. 架构决策:双 embodiment 联合训练

**结论:visrobot01 = embed_id 0(目标),kairobot01 = embed_id 1(辅助),联合训练一个模型,推理时走 embed_id 0。**

理由:
- visrobot01 单独只有 2101 ep,对 5B 模型偏少;kairobot01 的 6512 ep 任务完全相同,能提供叠衣服技能 + 视频动力学的强监督。
- 两套 rig 动作分布因臂间距不同而有差异,**各自一份 norm_stats** 比合并更贴合各本体;视觉差异交给模型 + Wan2.2 视频先验吸收。
- WAM 的 `robotype_to_embed_id` + 多份 `norm_path`(按 embed_id 索引)原生支持。
- 推理只在 visrobot01,用其专属 norm_stats(embed_id 0)→ 归一化最优。

**数据配比**:kai:vis ≈ 6512:2101 ≈ 3:1。为避免目标域被淹没,建议对 visrobot01 子集上采样到与 kai 相当(在 sampler / data entry 重复或加权,见 §5.3)。

> 备选(若想先要最简 baseline):只用 visrobot01(embed_id 0,单份 norm_stats),其余步骤相同、去掉 kai 部分。建议先跑这个冒烟,再升级到双 embodiment。

---

## 3. 必须的改动(代码 + 数据,均为轻量、非破坏)

### 3.1 本体标识:config 按数据集传参(方案 A,已实施 ✅,完全不碰 info.json)
已给 `third_party/giga-datasets/giga_datasets/datasets/lerobot_dataset.py` 的 `LeRobotDataset` 新增 `embodiment` 参数:config 里每个数据集 entry 写 `embodiment="visrobot01"/"kairobot01"`,`open()` 中优先作为 robotype 注入(`data_dict["robotype"]`),不读 info.json → **零污染共享数据集**,config 自描述。
- **生效前提**:giga-datasets 必须 editable 安装 —— `pip install -e ./third_party/giga-datasets`(否则改动不生效)。
- 映射:`robotype_to_embed_id = {"visrobot01": 0, "kairobot01": 1}`(必须精确命中;两名都不含 agibot/aloha/agilex 子串,务必写全,否则 fallback default 0)。

### 3.2 delta_mask(已重构 → 内嵌 norm_stats,2026-06-09)
> ⚠️ 历史:曾在 `wa_transforms_lerobot.py` 内置 `delta_mask_templates={0,1}` 硬编码 mask
> (vis/kai 同为 14 维 piper)。**该硬编码已删除**,改为从每份 norm_stats json 内嵌的
> `delta_mask` 读取(单一真值源,`resolve_delta_mask()`)。abs = mask 全 False。
> 现在不再改代码;mask 跟着 stats 走。完整设计+用法见
> [`action_repr_delta_abs_compat.md`](action_repr_delta_abs_compat.md)。

14 维 piper mask = `[True]*6 + [False] + [True]*6 + [False]`(关节 delta、夹爪 idx6/13 绝对);
vis/kai 相同。配合 `model_action_dim = 14`、`models.action_dim = 14`。新算的 stats 自动写入该
mask;旧 stats 无字段时回退 piper14。

### 3.3 训练端相机 key(config 传参,零改数据)
`WATransformsLerobot` 接受 `view_keys`,仅当 `None` 才 fallback 到 `cam_high`(`wa_transforms_lerobot.py:31,67`)。
→ 在 transform config 显式传:
```python
view_keys = ["observation.images.top_head",
             "observation.images.hand_left",
             "observation.images.hand_right"]
```

### 3.4 推理端相机 key(改 3 个脚本,仅推理阶段需要)
以下硬编码了 `cam_high/cam_left_wrist/cam_right_wrist`,**训练不受影响**,做开环验证/上机前再改:
- `scripts/inference_server.py:76-78`
- `scripts/inference_client.py:104-106`
- `world_action_model/pipeline/utils.py:97-99`

最简做法:在喂入 observation 时把 visrobot01 的 `top_head/hand_left/hand_right` 映射成上述键名(改 client / 上机 adapter 即可,不必动 pipeline)。

---

## 4. 阻塞项 / 待确认(需提前处理)

1. **✅ 权重已就位(无需下载)**:从本地已有权重拷贝到 `checkpoints/`:
   - backbone(Diffusers,`models.pretrained`):`../checkpoints/Wan2.2-TI2V-5B-Diffusers`(32G,源 `/vePFS/HuanQian/giga-world-policy/models/`)
   - T5(`compute_t5_embedding --wan_path`):`../checkpoints/Wan2.2-T5`(`models_t5_umt5-xxl-enc-bf16.pth` + `google/umt5-xxl`,源 `/vePFS/zundong/ViVa/weights/`)
2. **av1 解码验证**:`kai0_base` 是 av1,`LeRobotDataset` 用 `video_backend="pyav"`。需确认环境的 `av`/ffmpeg 带 av1 解码器(libdav1d)。若不支持 → 仅 kai 部分需转码 h264,或先用 kai0_dagger(h264)。vis 全 h264,无忧。
3. **fps 一致性**:两套均 30fps,`delta_info={"action":48}` → 48 帧 @30fps ≈ 1.6s 动作块,跨集一致,无冲突。无需抽帧。

---

## 5. 落地步骤

设环境变量:
```bash
cd /vePFS/tim/workspace/deepdive_kai0/giga_world_policy   # repo root,下面全部相对路径(checkpoints/kai0 均为 repo 同级目录)
DATA=../kai0/data/Task_A
CKPT=../checkpoints
PRETRAINED=$CKPT/Wan2.2-TI2V-5B-Diffusers          # models.pretrained
WAN=$CKPT/Wan2.2-T5                                 # compute_t5_embedding --wan_path
OUT=./assets_visrobot01
mkdir -p $OUT
```

### 5.0 代码改动(已完成 ✅,无需改任何 info.json)
- `LeRobotDataset` 加 `embodiment` 参数(§3.1)—— 已改;**需 `pip install -e ./third_party/giga-datasets` 使其生效**。
- `wa_transforms_lerobot.py` delta_mask 模板 embed_id 1 → 14 维 piper(§3.2)—— 已改。

### 5.1 计算 norm_stats(两份,按 embed_id 索引)
```bash
# embed_id 0 —— visrobot01(目标本体)
python -m scripts.compute_norm_stats \
  --data_paths $DATA/vis_base/v2/* $DATA/vis_dagger/* \
  --output_path $OUT/norm_stats_vis.json \
  --embodiment_id 0 --delta-mask "1,1,1,1,1,1,0,1,1,1,1,1,1,0" \
  --sample-rate 1.0 --action-chunk 48 --action-dim 14

# embed_id 1 —— kairobot01(辅助域)
python -m scripts.compute_norm_stats \
  --data_paths $DATA/kai0_base $DATA/kai0_dagger \
  --output_path $OUT/norm_stats_kai.json \
  --embodiment_id 1 --delta-mask "1,1,1,1,1,1,0,1,1,1,1,1,1,0" \
  --sample-rate 1.0 --action-chunk 48 --action-dim 14
```
→ config 里 `norm_path = [$OUT/norm_stats_vis.json, $OUT/norm_stats_kai.json]`(index 0=vis, 1=kai,顺序即 embed_id)。

### 5.2 预计算 T5 embedding(每个子集各跑一次,共 25 个)
```bash
for d in $DATA/vis_base/v2/* $DATA/vis_dagger/* $DATA/kai0_base $DATA/kai0_dagger; do
  python -m scripts.compute_t5_embedding \
    --repo_id "$(basename $d)" --root "$d" --wan_path $WAN \
    --device cuda --text_len 512 --t5_folder_name t5_embedding
done
```
生成 `<each>/t5_embedding/episode_*.pt`,并写回各 `episodes.jsonl`。

### 5.3 训练 config(`world_action_model/configs/visrobot01_fold.py`)
基于 `configs/example.py` 改,关键字段:
```python
import glob
DATA = "../kai0/data/Task_A"          # 相对 repo root(运行 `python -m scripts.train` 时 cwd=repo root)
vis_paths = sorted(glob.glob(f"{DATA}/vis_base/v2/*")) + sorted(glob.glob(f"{DATA}/vis_dagger/*"))
kai_paths = [f"{DATA}/kai0_base", f"{DATA}/kai0_dagger"]
num_frames = 48
view_keys = ["observation.images.top_head","observation.images.hand_left","observation.images.hand_right"]
image_frame_offsets = [0, num_frames//4, num_frames//2, 3*num_frames//4, num_frames]

def _entry(p, emb):
    return dict(_class_name="LeRobotDataset", data_path=p, data_size=None,
                embodiment=emb,                       # 方案A:本体标识 → WAM 路由 norm_stats/delta_mask
                delta_info={"action": num_frames},
                delta_frames={k: image_frame_offsets for k in view_keys})

config = dict(
  project_dir="runs/visrobot01_fold",
  runners=["world_action_model.trainer.wa_casual_trainer.CasualWATrainer"],
  launch=dict(gpu_ids=[0,1,2,3,4,5,6,7], distributed_type='DEEPSPEED',
              deepspeed_config=dict(deepspeed_config_file='accelerate_configs/zero2.json'),
              until_completion=True),
  dataloaders=dict(train=dict(
      # vis 上采样 ~3x 以平衡 6512:2101 ≈ 3:1;先验证可改 1x
      data_or_config=[_entry(p,"visrobot01") for p in vis_paths]*3 + [_entry(p,"kairobot01") for p in kai_paths],
      batch_size_per_gpu=8, num_workers=8,
      transform=dict(type='WATransformsLerobot',
          robotype_to_embed_id={"visrobot01":0, "kairobot01":1},
          dst_size=(256,192), num_frames=num_frames, is_train=True,
          norm_path=["./assets_visrobot01/norm_stats_vis.json","./assets_visrobot01/norm_stats_kai.json"],
          model_action_dim=14, num_views=3, t5_len=64,
          view_keys=view_keys,
          image_cfg=dict(mask_generator=dict(max_ref_frames=1, start=1, factor=4))),
      sampler=dict(type='DefaultSampler'), collator=dict(is_equal=True))),
  models=dict(pretrained="../checkpoints/Wan2.2-TI2V-5B-Diffusers", strict=False,
              action_dim=14, flow_shift=5.0, expand_timesteps=True, state_repeats=1),
  optimizers=dict(type='CAME8Bit', lr=2**(-14.5), weight_decay=1e-2),
  schedulers=dict(type='ConstantScheduler'),
  train=dict(resume=False, max_steps=100000, gradient_accumulation_steps=1,
             mixed_precision='bf16', checkpoint_interval=5000, with_ema=True,
             log_with='tensorboard', log_interval=1),
)
```
> `view_keys` 必须在 transform dict 里(否则 fallback cam_high)。所有路径相对 repo root,运行时确保 cwd 在 `giga_world_policy/`。

### 5.4 冒烟 → 全量训练
```bash
# 冒烟:临时改 max_steps=200、只挂少量 vis 子集,确认 visual_loss/action_loss 正常下降
python -m scripts.train --config world_action_model.configs.visrobot01_fold.config
# 通过后恢复全量 max_steps≈100k
```

### 5.5 开环推理验证(visrobot01)
先按 §3.4 处理推理端相机名,再:
```bash
python -m scripts.inference_server \
  --model_id ../checkpoints/Wan2.2-TI2V-5B-Diffusers \
  --transformer_path runs/visrobot01_fold/checkpoint-XXXX/transformer \
  --stats_path ./assets_visrobot01/norm_stats_vis.json \
  --t5_embedding_pkl <某个 vis 子集>/t5_embedding/episode_000000.pt \
  --state_dim 14 --action_dim 14 --action_chunk 48 \
  --delta_mask "1,1,1,1,1,1,0,1,1,1,1,1,1,0"
# 另开终端:
python -m scripts.inference_client --dataset_paths <vis 子集> --action_chunk 48 --save_dir ./vis
```
用开环 action MSE / 轨迹可视化对比 GT,作为收敛 sanity check;通过后再上 visrobot01 实机。

---

## 6. 验证标准
- 训练:`action_loss` 稳定下降(本任务主目标);`visual_loss` 同步下降但非主指标。
- 开环:visrobot01 验证集上预测 action 与 GT 的逐维 MSE 收敛到合理水平,可视化轨迹贴合。
- 实机:visrobot01 叠衣服成功率(对照 kai0 内部 pi0.5 基线)。

## 7. 若改为「在 kai0 内部用 pi0.5 微调」
则本方案不适用——直接用现成 `pi05_flatten_fold_*` config + `kai0/scripts/train.py`,几乎零预处理。本文档仅针对 WAM 路径。
