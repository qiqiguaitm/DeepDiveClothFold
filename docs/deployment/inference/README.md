# inference/ — 推理与真机部署

> **场景**: 把训完的 ckpt 部署到真机 / sim01 / 推理时延优化 / RTC / IPC / ROS2 校验 / web 推理服务。

## 目录结构

```
inference/
├── README.md                              ← 你在这里
├── rtc_implementation.md                  RTC (Real-Time Chunking) 算法实现 (π₀.5/JAX)
├── xvla_rtc_design.md                     X-VLA RTC 移植设计 (EE6D 前向 inpaint, 设计未实施)
├── ipc_inference_deployment_review.md     IPC 推理服务架构 review
├── ros2_image_inference_validation_review.md  ROS2 图像处理与推理校验
├── sim01_deployment.md                    sim01 部署文档
├── fixed_noise_inference_fix.md           G0 fixed-noise — vis_v2_full 真机 oscillation 修复 (2026-05-27)
├── build_web_venv.md                      web 推理服务通用 venv (多框架解耦)
└── realtime_vla/                          实时推理优化 series (4 文件)
    ├── README.md
    ├── strategy.md                        决策摘要 + 上下文 + 优化排序 + Fallback (选项 Y)
    ├── roadmap.md                         5 阶段实施路线 + 真机测试方案
    ├── v1_triton_log.md                   V1 Triton 已实施日志 (P50 76→32ms)
    └── layer_b_plan.md                    Layer B 系统级优化 未来 plan
```

## 文件清单

| 文件 | 行数 | 用途 |
|---|---|---|
| [`rtc_implementation.md`](rtc_implementation.md) | ~249 | RTC (Inference Real-Time Chunking) 算法实现, 4 schedules + jax.vjp guidance |
| [`deploy_speed_analysis.md`](deploy_speed_analysis.md) | ~analysis | **部署后机械臂运行速度分析** — RTC/EMA/V0-V1 对真机速度影响; 场景不可控故用峰值/分布(非均值); 发现 RTC 削手臂峰值~0.62×、夹爪~0.53× |
| [`rtc_ema_speed_ablation.md`](rtc_ema_speed_ablation.md) | ~design | RTC/EMA 速度消融实验设计 — 开关方法 (RTC 热改/EMA 需重启) + 2×2 控制变量协议 + 结果表 |
| [`speed_ablation_episodes.md`](speed_ablation_episodes.md) | ~ledger | 速度消融采样 episode 登记表 (A0-A3 v0 / B0-B3 v1) + 跨日期检索法 |
| [`xvla_rtc_design.md`](../../../xvla/docs/deployment/inference/xvla_rtc_design.md) | ~design | X-VLA RTC 移植设计 — EE6D 原生空间 + **前向过程 inpaint** (vs 算术平均的 IK/decode 失败率差异) + 多层 IK 兜底; 异步双时钟前置; gated 于 E0 读视觉 |
| [`ipc_inference_deployment_review.md`](ipc_inference_deployment_review.md) | ~178 | IPC 推理服务架构 — 与原版差异分析 |
| [`ros2_image_inference_validation_review.md`](ros2_image_inference_validation_review.md) | ~430 | ROS2 图像处理 + 推理结果校验策略 review |
| [`sim01_deployment.md`](sim01_deployment.md) | ~611 | sim01 仿真机器人部署 / 完整 step-by-step |
| [`fixed_noise_inference_fix.md`](fixed_noise_inference_fix.md) | ~150 | G0 fixed-noise inference 修复 — vis_v2_full 真机 oscillation 诊断 + sim01 端代码补丁 (RTC 兼容) |
| [`build_web_venv.md`](build_web_venv.md) | ~137 | 通用 web 推理服务 venv — 支持多框架, 与代码解耦, 可用于 data_manager + 推理 host |
| [`xvla_inference_bringup.md`](../../../xvla/docs/deployment/inference/xvla_inference_bringup.md) | ~210 | X-VLA ckpt 真机 bring-up 计划 (修订版) — server-only `ee` 16D; 端到端审计后 4 层阻塞 (旧ckpt作废/新ckpt训练中/R4 server改lerobot类/客户端ee链缺失) + R1-R4 正确性契约 (link6/interleaved/二值gripper/lerobot预处理) |
| [`realtime_vla/`](realtime_vla/README.md) | series | 实时 VLA 推理优化 (P50 76→32ms 已达成, Layer B 下一阶段) |

## 按需求找文件

| 你想做什么 | 去 |
|---|---|
| 看实时推理优化路线 / 选项 X 战略 | realtime_vla/strategy.md |
| 看 5 阶段实施路线图 | realtime_vla/roadmap.md |
| 看 V1 Triton 已实现日志 (异步流水线 / FP8 / SHM) | realtime_vla/v1_triton_log.md |
| 看下一阶段 Layer B 系统级优化 | realtime_vla/layer_b_plan.md |
| RTC 算法 (chunk overlap / inpainting / guidance) 怎么实现 | rtc_implementation.md |
| **把 RTC 移植到 X-VLA** (EE6D / 前向 inpaint / 求逆失败率 / IK 兜底) | xvla_rtc_design.md |
| IPC 推理服务 (kai0-style) 架构 | ipc_inference_deployment_review.md |
| ROS2 image topic / 推理 input 校验 | ros2_image_inference_validation_review.md |
| sim01 上完整部署 ckpt 推理 | sim01_deployment.md |
| **vis_v2_full 真机 oscillation 怎么修** (走几步退几步 / 夹爪犹豫) | fixed_noise_inference_fix.md |
| 起一个 web 服务接口供其他框架/真机调用推理 | build_web_venv.md |
| **把训好的 X-VLA ckpt 在真机跑通** (server-only / ee 协议 / 坐标 codec) | xvla_inference_bringup.md |

## 跨场景跳转

- 部署前需要先拿 ckpt → `../training_ops/data_sync_tos.md` (ckpt 训练→TOS→sim01)
- 推理时实时可视化 → `../visualization/inference_visualization.md`
- 推理时硬件故障/相机问题 → `../incidents/` (查历史)
- 跨本体推理 routing (Soft Prompt domain_id) → `../strategy/cross_embodiment_strategy.md` §5.2
