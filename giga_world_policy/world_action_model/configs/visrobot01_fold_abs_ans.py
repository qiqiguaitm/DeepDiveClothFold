"""abs-ANS —— X-WAM 异步噪声采样(arXiv 2604.26694 Eq.4)× action_attends_video。

诊断依据(docs/wam_mae_root_cause_and_optimization.md):TF 探针证明 lookahead 的 action 解码器
给 GT 未来视频时 mae@48=0.0598(−61%),但同步噪声训练只覆盖 (t_a=t_O) 对角线 → 推理时
"动作低噪、自预测视频高噪"的状态分布外,exposure bias 吃掉全部增益(naive lookahead ≈ 打平,
复现 GigaWorld Table 6)。ANS 训练覆盖整个上三角(t_O ≥ t_a 恒成立)→ 推理轨迹在分布内,
动作只学提取高噪视频中可靠的低频信息 ≈ 对自预测误差鲁棒;且动作 T_a=5 步先出,延迟减半。

与 `visrobot01_fold_abs_lookahead` 的唯一差别 = async_noise(噪声耦合方式),attention/配方
逐字相同 → 与 naive lookahead 构成第二个干净 A/B。X-WAM 消融(RoboCasa, 同 Wan2.2-TI2V-5B
backbone):ANS 67.8 SR vs 同步 66.4 vs naive 解耦 67.2(视频崩),4.5× 提速。
eval:episode_report 自动识别 async_noise=True → T_a=5/T_O=10、metric eps 提前返回。
ans_p(分支A概率)论文未公开,取 0.1 起步;若视频质量退化可升。
"""
import copy

from world_action_model.configs.visrobot01_fold_abs_lookahead import config as _base

config = copy.deepcopy(_base)

PROJ = "runs/visrobot01_fold_abs_ans"
config["project_dir"] = PROJ
config["models"]["view_dir"] = PROJ

# 唯一变量:X-WAM 异步噪声采样(action_attends_video=True 继承自 lookahead 基配置)
config["models"]["async_noise"] = True
config["models"]["ans_p"] = 0.1            # 分支A:t_a=0、t_O~U(动作作为干净条件)
config["models"]["ans_beta"] = (1.5, 1.0)  # 分支B:t_O = t_a + (1−t_a)·Beta(1.5,1)
