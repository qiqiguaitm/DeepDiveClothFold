# 各类模型 MAE 横向对比(fold 任务,2026-06-15)

> 指标:`raw_mae@h` = 动作 chunk 前 h 步的**累计平均**关节误差(`ae[:h].mean()`,见
> `episode_report.py`),visrobot01_v3_val / v1_val 60-ep 全量,越低越好。
> `@1` 是伪指标(数据约定 `action[0]≡state`,地板为 0,只测 state 透传);**以 `@48` 为准**。
> 绝对值仅在同评测集内可比;FastWAM 与 GWP 评测脚本一致(8-shard `eval_offline_fold` / `episode_report`)。

## 一、总览(按 MAE@48 排序)

| 模型 | 架构 / 表示 | 数据 | MAE@48 | 延迟 | 状态 |
|---|---|---|---|---|---|
| **FastWAM opt NFE=5 exact** | 独立 ActionDiT(378M,MoE)/ abs | v3 | **0.0905** | **85ms** (RTX5090) | ✅ 部署选型 |
| **FastWAM v4** (step25000) | 独立 ActionDiT / abs | v3 | **0.0910** | NFE=20 | ✅ 最终可用 ckpt |
| **FastWAM v3** (step25510) | 独立 ActionDiT / abs | v3 | 0.0912 | NFE=20 | ✅ |
| gwp_ori(官方切断基线) | 共享 DiT | — | 0.0916 | 532ms | 参考 |
| gwp_ans(异步噪声采样) | 共享 DiT | — | 0.0918 | 283ms | 参考 |
| **delta-5x**(gwp_ori 复现) | 共享 DiT / **delta** | v1 | 0.1128 | — | ✅ 生产 |
| abs_50k | 共享 DiT / **abs** | v1 | 0.1492 | — | ✅(abs A/B) |
| π₀.₅(kai0) | openpi | — | 0.1155 | — | 参考 |
| v3_abs | 共享 DiT / abs | **v3 损坏** | 0.4372 | — | ❌ 数据损坏 |
| gwp_abs_v4 | 共享 DiT / abs | **v3 损坏** | 0.41–0.45 | — | ❌ stale-index 损坏 |

## 二、可训练模型的 horizon 曲线

| 模型 | @1 | @10 | @24 | @48 |
|---|---|---|---|---|
| FastWAM v4 (25000) | 0.0038 | 0.0299 | 0.0600 | **0.0910** |
| FastWAM v3 (25510) | 0.0038 | 0.0297 | 0.0595 | 0.0912 |
| gwp_ori(官方) | 0.0053 | 0.0298 | 0.0595 | 0.0916 |
| gwp_ans | 0.0063 | 0.0288 | 0.0574 | 0.0918 |
| delta-5x(gwp_ori 复现) | 0.0028 | 0.0347 | 0.0720 | 0.1128 |
| abs_50k(共享 DiT abs) | 0.0094 | 0.0513 | 0.1090 | 0.1492 |
| v3_abs(损坏) | 0.4537 | 0.4457 | 0.4377 | 0.4372 |
| gwp_abs_v4(损坏) | 0.42–0.52 | — | — | 0.41–0.45 |

FastWAM v4 收敛曲线(MAE@48,NFE=20):
`15000=0.1028 → 17500=0.0939 → 20000=0.0931 → 22500=0.0913 → 25000=0.0910`(已饱和)。
step25510 ckpt 落盘被截断(312KB vs 12GB),不可用;**step25000 即最终部署 ckpt**。

## 三、结论

1. **最佳可训练结果 ≈ 0.091** —— FastWAM v4/v3 与官方 gwp_ori 全部收敛到 ~0.091。
   FastWAM 的独立 378M ActionDiT 追平官方共享 DiT 模型。

2. **共享 DiT 内 delta 优于 abs**:同一 v1 数据 / 配方 / batch 下,delta-5x(0.1128)
   < abs_50k(0.1492),唯一差别是表示。印证架构论点——共享 action+video transformer 里
   delta 与 video token 的速度语义对齐,abs 还需模型额外吸收绝对尺度。FastWAM 用**独立**
   ActionDiT 才让 abs 达到 0.091。详见 [`action_repr_delta_abs_compat.md`](action_repr_delta_abs_compat.md)。

3. **v3_abs / gwp_abs_v4 的 0.42 不是表示问题,是数据损坏**:v3 训练 parquet 携带
   合并前的 stale `episode_index`/`index` 列(val 集已重写修复,train 集只在代码侧绕过但未根治),
   latent↔action 错位 → 模型预测均值级常量。flow_shift/training_weight 是误诊;
   `abs_50k` 证明 abs 在 flow_shift=5 + **干净数据**下可正常收敛到 0.1492。
   损坏 run 已重命名 `runs/_corrupt_gwp_abs_v4_staleidx`。

4. **部署**:FastWAM opt NFE=5 exact = **0.0905 @85ms(RTX5090)**,与 stock NFE=20
   (0.0912 @931ms)精度等价,10.9× 加速。详见
   [`inference_speed_optimization.md`](inference_speed_optimization.md)。

## 四、来源

- FastWAM v3/v4:`fastwam/runs/visrobot01_fold_uncond_1e-4/aihc_5n8g_{v3,v4}/report_step*/summary.json`
- delta-5x / abs_50k / v3_abs:`giga_world_policy/runs/{visrobot01_fold_aihc_latent_5x,visrobot01_fold_abs_50k,visrobot01_v3_abs}/report_step*/summary.json`
- gwp_ori / gwp_ans / pi05 参考值:`fastwam` eval 日志 ref 行(8-shard 一致评测)
- FastWAM opt:`docs/inference_speed_optimization.md`、记忆 `wam-abs-lookahead-experiment`
