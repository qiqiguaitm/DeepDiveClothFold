# CRAVE 最终效果图索引(按方法 × 类型)

> 每类挑一张代表"首图"放在本目录,方便快速查找。完整图集见上级目录;结论说明见 [../../../README.md](../../README.md)。
> 同步视频(不入 git)在 `temp/`,文件名见每行"视频"。

| # | 类别 | 图文件 | 一句话结论 | 同步视频(temp/) |
|---|---|---|---|---|
| 01 | **核心方法·demo域** | `01_method_demo_milestones.png` | 自动挖出 20 个进度均匀 milestone | — |
| 01 | 核心方法·value | `01_method_demo_value.png` | 干净 0→1 阶梯,前段无误判 | — |
| 01 | 核心方法·撞色兜底 | `01_method_colorclash_value.png` | 橙色衣物三路冗余不误判 | — |
| 01 | 核心方法·退步 | `01_method_rollout_regression.png` | 真机 rollout 两次回落 0 + 恢复 | `rollout_v24_sync.mp4` |
| 02 | **泛化·新本体** | `02_generalize_xvla_newrobot.png` | XVLA soft_fold 168ep corr 0.956 / 100%≥0.7 | `temp/generalization_value_eval/xvla/sync_ep*.mp4` |
| 02 | 泛化·真实ALOHA | `02_generalize_coffee_realaloha.png` | lerobot coffee 50ep corr 0.988 / 单调100% | `temp/generalization_value_eval/coffee/sync_ep*.mp4` |
| 03 | **vs监督AE·同域** | `03_vs_ae_kai0base_indist.png` | kai0_base ep2047 与 AE 打平(corr0.82)更平滑 | — |
| 03 | vs监督AE·OOD | `03_vs_ae_autonomy_ood.png` | 真机 rollout CRAVE 干净退步, AE 欠读 end0.33 | `crave_vs_ae_autonomy_sync.mp4` |
| 04 | **三档打标** | `04_3level_vs_ae_labels.png` | CRAVE 三档结构清晰(normal72%); AE 强行三档=噪声 | `crave_3level_vs_ae3_ep808.mp4` |
| 04 | AWBC标签分布 | `04_awbc_label_distribution.png` | CRAVE 天然 neg4% vs AE 41%(专家数据 AE 负是噪声) | — |
| 05 | **在线读出·离线vs在线** | `05_offline_vs_online_30hz.png` | 固定滞后在线≈离线 corr0.89, 窗标定后抖动降8× | `crave_ep2047_final_offline_vs_online.mp4` |
| 05 | 频率·2×2解耦 | `05_freq_2x2_mine_x_output.png` | 挖矿频率不改质量(动曲线), 输出频率=分辨率 | `crave_freq_4way_ep2047.mp4` |
| 05 | 因果DP | `05_causal_dp_fixedlag.png` | 固定滞后 Viterbi 修复朴素前向粘滞, corr0.94 | — |
| 06 | **斜坡读出** | `06_ramped_vs_staircase.png` | 斜坡 τ 升源于"抹退步去噪", 非斜坡本身(诚实更正) | — |
| 07 | **簇间流转·2D** | `07_cluster_flow_2d.png` | 单 episode 在特征空间沿 milestone 链流转 | — |
| 07 | 簇间流转·3D | `07_cluster_flow_3d.png` | 3D PCA 旋转彗星 | `crave_cluster_flow_3d_ep2302.mp4` |
| 08 | **臂噪声影响** | `08_armmask_noise_impact.png` | 去臂 jitter 降40%, 但窗标定是更大杠杆 | — |

## 按方法分类速查
- **零训练 value(主线)**:01(方法)· 02(泛化)
- **对标监督/RL**:03(vs pi0-AE)· 04(AWBC 打标)
- **读出工程**:05(频率/在线/因果)· 06(斜坡)
- **可解释性/鲁棒**:07(簇间流转)· 08(臂噪声)

## 按类型分类速查
- **value 曲线**:01_value, 01_colorclash, 03_*, 06
- **打标/分布**:04_*
- **泛化总览**:02_*
- **频率/在线工程**:05_*
- **结构/几何**:07_*
- **消融**:08, 05_freq, 06
