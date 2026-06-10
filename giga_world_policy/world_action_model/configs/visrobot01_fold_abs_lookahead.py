"""abs-LOOKAHEAD —— world-model lookahead 接入 action 解码(架构改动,受控 A/B vs abs-best)。

动机(见 docs/action_repr_delta_abs_compat.md 的定量分析 + 本仓 eval):abs-best @45k 在 mae@1/@10
远胜 pi0.5,却在 mae@24/@48 反超(长程更差)。根因 = causal mask 把 action token 与 noisy-video token
**切断**(`CasualWorldActionTransformer` 的 `mask[s_r_end:action_end, action_end:]=-inf`),所以推理期
action 只看 当前帧(ref)+state,**完全用不到模型自己预测的未来视频** → 长程退化成"无前瞻前馈策略",
video 目标只是吃容量的辅助损失、不回报长程。同时 @24→@48 三模型斜率几乎一致(信息受限)——差距全在
@1→@24 这段,正是"有/无未来视觉"该起作用的区间。

本改动:`action_attends_video=True` → 训练 + 全量推理 都**不再切断** action→noisy-video,action token
通过 attend 到(去噪中的)未来视频拿到真正的 world-model 前瞻。其余配方与 `visrobot01_fold_abs_best`
**逐字相同**(abs stats、5:1、bs/gpu=8×5节点=batch320、5×LR、warmup5000、EMA off、50k)——唯一变量 = mask。
这是与 abs-best 的干净 A/B。

资源:稠密 SDPA 对被 mask 的 pair 仍照算后 softmax 清零,去掉 -inf **不增 FLOPs/显存** → 资源画像与
abs-best 完全一致(bs/gpu=8 峰值 ~70G),沿用同一 5n8g AIHC spec。

风险(已知):train 期 action attend 的是 noised-GT 未来视频,推理期换成模型**自己预测**的未来视频 →
world-model exposure bias(这正是当初加 mask 的原因)。先用本 A/B 测"前瞻是否净增益";若退化,再上
two-pass cascade / partial-lookahead / scheduled-sampling。

部署影响:lookahead ckpt **不能**走 action_only 快路径(那条路丢掉 video token);serve/eval 必须用
全量去噪(pipeline action_only=False)。会失去 prefix-KV / action-only 提速(serving 速度需重评)。
模型 forward 已对 action_only+flag 组合抛错兜底。若算力紧 max_steps 可减(30k 即可先读趋势)。
"""
import copy

from world_action_model.configs.visrobot01_fold_abs_best import config as _base

config = copy.deepcopy(_base)

PROJ = "runs/visrobot01_fold_abs_lookahead"
config["project_dir"] = PROJ
config["models"]["view_dir"] = PROJ

# 唯一变量:打开 world-model lookahead(action token 可 attend noisy-video)。
# trainer 通过 register_to_config 写进 transformer config → ckpt 自描述;默认 False=原切断行为。
config["models"]["action_attends_video"] = True
