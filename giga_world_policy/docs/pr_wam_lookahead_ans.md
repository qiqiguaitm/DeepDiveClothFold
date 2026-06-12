> 命名约定:gwp_ori = 官方切断基线(旧称 abs-best);gwp_ans = 异步噪声采样模型(旧称 ANS 模型)。

# PR: WAM world-model lookahead — action_attends_video + X-WAM ANS + MAE 根因排查

分支 `wam/abs-lookahead` → `main`。PR 链接:
https://github.com/qiqiguaitm/deepdive_kai0/pull/new/wam/abs-lookahead

## 概要
针对 abs WAM 的长程 mae@48(0.1285)弱于 pi0.5(0.1155),做了根因排查 → 架构改动 → 优化方案三步。

### 1. action_attends_video(9ea6fdf)
打开 action↔video 注意力(world-model lookahead),向后兼容 config 旗标(默认 False=原因果切断,
旧 ckpt 不受影响),serverless AIHC 提交/查询工具(`scripts/aihc/submit_raw.py`、`job_status.py`)
——granted serverless 队列无法用 `aihc job create`(pool name→id 404),改走 raw OpenAPI POST。

### 2. MAE 根因定量排查(2abb437)
4 个 GPU 探针(`_diag_deep.py` 逐维/NFE/best-of-N、`_diag_teacher_force.py`):
- **mae@1 是伪指标**:stay-baseline(原地不动)@1=0(action≡state 约定);@48 各模型仅比
  stay 好 28-37%,误差主体=任务前向多模态(best-of-4 砍 18%/abs、33%/lookahead)。
- **teacher-forcing 实锤**:lookahead@25k 给 GT 未来视频 mae@48=0.0598 vs 0.1542(**−61%**),
  对照组(abs-best,mask 切断)给 GT 逐位不变 → action 解码器已重度依赖视频,
  瓶颈=世界模型预测质量(exposure bias)。NFE 10v30 无差、逐维无病理。
- 文献交叉验证(GigaWorld 官方 Table 6 / X-WAM / π0 / SIMPLER)+ 五方案对比 + X-WAM ANS 详解
  → `docs/wam_mae_root_cause_and_optimization.md`。

### 3. X-WAM 异步噪声采样 ANS(0fe92c7)
ANS(arXiv 2604.26694 Eq.4)训练耦合采样保证 σ_video ≥ σ_action(覆盖推理"动作低噪/自预测
视频高噪"的上三角)→ 修 exposure bias;推理动作 T_a=5 步先出、延迟减半;模型架构零改动
(Wan2.2 TI2V per-token timestep 原生支持)。
- `wa_casual_trainer.py` 双 σ 采样 + 分支A loss 掩码;`wa_pipeline.py` 解锁步 + T_a<T_O + 快档;
  `episode_report.py` ANS ckpt 自动 T_a=5/T_O=10;`configs/visrobot01_fold_abs_ans.py`。
- 验证:t_O≥t_a 不变量单测 0 违反;6 步真数据 smoke 通过;默认关闭时逐位等价。

## 实验(已收敛,200-ep 严格同协议终评见 docs/wam_mae_root_cause_and_optimization.md 终局结果 v2)
| 任务 | jobId | 结果(200ep:@1/@10/@24/@48,act 延迟) |
|---|---|---|
| naive lookahead | job-i3ngi7f23gi5 | 复现 GigaWorld Table 6:同步 attend 略差于切断,负迁移随 horizon 放大 |
| **gwp_ans(本 PR 核心,旧称 ANS 模型)** | job-3am80jbendcf | **.0063/.0288/.0574/.0918 @283ms** —— @10/@24/@48 全面超 delta-5x(.1128@48)与 pi0.5(.1155@48);与 gwp_ori(切断,原 abs-best,.0916@48)精度打平、延迟 −47%,延迟约一半;@1 输 delta(锚定结构优势) |

注:per-1k 曲线用的 60-ep 子集系统性偏难,跨家族对比一律以 200-ep 终评为准(勘误详见根因文档)。

## 后续
- ANS 终值出后补全 A/B(naive vs ANS vs abs-best);eval 指标拟换 best-of-N + 闭环 SR(MAE 是弱代理)。
- 备选:FLARE 式 latent 对齐(GR00T N1.5 路线)、deeper action decoder、ckpt soup。
