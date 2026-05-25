# Task_A 数据集诊断报告 (2026-05-14)

> 起因: SFT 后真机出现 **(a) cloth loop** (复杂场景抓不到衣角) + **(b) 桌面为空抖动** + **(c) `mixed_pure2_1800_6000` 真机抖动 > `pure_1200_new_norm`** 三大问题。
>
> 本文档整理对官方数据 (kai0_base / kai0_dagger) 与自采数据 (vis_base) 的全面对比分析、定位的核心 bug、修复方案。
>
> **结论**: 主要瓶颈是 (1) **vis_base 含 10% 采集 bug 帧** 污染 action 先验 → 真机抖; (2) **D435i vs D405 视觉域差** → cloth loop. 已生成 `vis_base_clean_v2` 清洁数据 (max_jump p99 从 1.05 降到 0.44), 当前 `task_a_new_smooth_800_new_norm` 训练验证中。

---

## 1. 数据集基本信息 (本地实测)

### 1.1 规模对比 (本地路径 `kai0/data/Task_A/`)

| 数据集 | episodes | frames | 任务描述 | 来源 |
|---|---:|---:|---|---|
| **kai0_base** (官方) | 3,055 | 3,362,369 | "Flatten and fold the cloth." | 官方公开采集 |
| **kai0_dagger** (DAgger) | 3,457 | 2,415,341 | "Flatten and fold the cloth." | 官方修正样本 |
| **vis_base** (自采 v2 总和) | 895 | 1,063,168 | "Flatten and fold the cloth." | 自有团队摇操 |

### 1.2 关键事实

- **`kai0_base/norm_stats.json` 与 `kai0_dagger/norm_stats.json` 字节级完全相同** — 两者都继承自 `Task_A_mixed_1` 的统计, 不是各自重算。
- 三方 `tasks.jsonl` 完全一致 — language prompt 无差异。
- `kai0_base` 数据布局: `data/chunk-{0..3}/episode_*.parquet` + `videos/chunk-XXX/observation.images.{cam}/episode_*.mp4` (LeRobot v2.1 标准)
- `vis_base` 数据布局: `vis_base/<date>-v2/data/chunk-000/...` + `videos/chunk-000/{cam}/...` (无 `observation.images.` 前缀, 每日期独立子目录)

### 1.3 vis_base 按日期分布 (v2 cleaned)

| Date subdir | episodes | frames | frames/ep | 备注 |
|---|---:|---:|---:|---|
| 2026-04-23-v2 | 22 | 16,905 | 769 | 早期 |
| 2026-04-24-v2 | 187 | 152,090 | 813 | 早期 |
| 2026-04-25-v2 | 100 | 67,539 | 675 | 早期 |
| 2026-04-28-v2 | 152 | 104,120 | 685 | 早期 |
| 2026-04-29-v2 | 100 | 118,429 | 1,184 | 中期 |
| 2026-04-30-v2 | 83 | 167,577 | 2,019 | 长 ep (复杂) |
| 2026-05-06-v2 | 100 | 185,446 | 1,854 | 复杂场景 |
| 2026-05-07-v2 | 20 | 37,430 | 1,872 | 复杂场景 |
| 2026-05-08-v2 | 101 | 161,766 | 1,601 | 复杂场景 |
| 2026-05-09-v2 | 30 | 51,866 | 1,729 | 复杂场景 |
| **总计** | **895** | **1,063,168** | — | — |

**观察**: vis_base 后期 (05-06 起) `frames/ep` 是早期的 2.4× — 用户确认是**更复杂场景** (折叠步骤更多)。

---

## 2. 三大真机问题及对应数据分析

### 2.1 问题一: **cloth loop** (复杂场景抓不到衣角)

#### 2.1.1 用户报告
- baseline `Task_A_mixed_1` (6000 数据训练): 复杂场景能抓到衣角, **但存在 cloth loop**
- 自采数据 SFT 后: **简单场景无 loop, 但复杂场景退化**
- **关 RTC 后真机 loop 仍然存在** (说明不是 RTC 反馈放大)

#### 2.1.2 离线诊断 (`~/tmp/kai0_diagnostic_scripts/diagnose_cloth_loop.py`)

对 mixed_1 在三个数据集上做闭环 rollout (200 ep, 3 detectors):

| Dataset | N | mean loop_score | median | high (>0.5) | rate |
|---|---:|---:|---:|---:|---:|
| kai0_base | 100 | 0.434 | 0.45 | 4 | **4%** |
| kai0_dagger | 50 | 0.448 | 0.45 | 7 | **14%** ← 最高 |
| vis_base | 50 | 0.429 | 0.40 | 4 | **8%** |

**关键观察**:
- 三个数据集 loop_score 均值差异 <5% — **loop 不是数据中已有的 pattern**
- 三方 autocorr 均值 0.485-0.520, p99 都 <0.7 — **训练数据本身没有 cloth loop 演示**
- 否定"kai0 数据污染"假说: kai0_base 反而最低 (4%)

#### 2.1.3 结论
**cloth loop ≠ 模型 imitate 数据中的 loop**, 而是 **OOD prediction 失稳**:
- 训练: D435i 视觉空间
- 部署: D405 视觉空间 (硬件差异)
- D405 输入对模型来说是 OOD → 预测不稳定 → 闭环累积漂移 → 看似 loop

---

### 2.2 问题二: **桌面为空时抖动**

#### 2.2.1 用户报告
- baseline `mixed_1` 在空桌面: **不抖**
- 用自采 600 ep + mirror 数据 SFT 后: **空桌面就抖**

#### 2.2.2 诊断逻辑

**OOD 假说被排除**: 如果纯 OOD, baseline mixed_1 也应该抖 (它训练时也没见过空桌面)。

**真因: vis_base 含高 jump 帧, 污染了 action 先验**:

Diffusion policy 的工作机制:
```
condition (obs, state) → 模型预测 action 分布
  ↓ 强 condition (训练分布内)
collapse 到示范 action
  ↓ 弱 condition (OOD)  
回归到边际分布 p(action) "prior"
```

`mixed_1` 训练于 kai0 (smooth, mean diff=0.0058) → prior 集中在小 action → 空桌面输出小动作 → 不抖

vis SFT 后, prior 被拉宽 (vis 含 p99=1.05 rad 的极端跳变) → 空桌面 diffusion 抽到大 jump → 真机抖动

---

### 2.3 问题三: **`mixed_pure2_1800_6000` 抖动 > `pure_1200_new_norm`**

#### 2.3.1 用户观察
两个混合训练的模型, 真机执行平滑度不同:
- `task_a_new_pure_1200_new_norm` (纯 kai0_base -new 子集): 真机执行最平滑
- `task_a_new_mixed_pure2_1800_6000_new_norm` (含 vis + base + advantage): val MAE 是 SOTA 0.0085 但真机抖动

#### 2.3.2 数据 smoothness 量化对比

| Dataset | overall diff mean | overall diff p99 | max jump per ep mean | **max jump per ep p99** |
|---|---:|---:|---:|---:|
| **kai0_base** | 0.0058 | 0.0523 | 0.1268 | **0.1945** |
| kai0_dagger | 0.0094 ⬆62% | 0.0848 ⬆62% | 0.1726 ⬆36% | 0.2937 ⬆51% |
| **vis_base** (原始) | 0.0061 | 0.0674 | 0.2752 ⬆117% | **1.0541** ⬆**442%** |

#### 2.3.3 解释吻合
- `pure_1200` = kai0_base -new 子集 → **最平滑** (0.0058)
- `mixed_pure2_1800_6000` = kai0_base (3055) + advantage (3055, 类比 dagger) + vis mirror (1790) → 平均抖动更高
- 模型学到了更大的 action 变化 → 真机抖动

**反推**: `val MAE@1 不是真机平滑度的好指标` — val MAE 衡量预测准确性, 不衡量轨迹连续性。

---

## 3. 视觉硬件差异 (D435i vs D405)

### 3.1 关键事实
| 数据 | 相机 | FOV | 工作距离 |
|---|---|---|---|
| **kai0_base / dagger** | Intel RealSense **D435i** | 87°, 立体 RGB | 0.3m–10m |
| **vis_base** | Intel RealSense **D405** | 87°, 近距特化 | 0.07m–0.5m |
| **真机部署** | **D405** | 87°, 近距 | 与 vis 同 |

### 3.2 量化对比 (100 ep 抽样)

| Dataset | mean RGB | mean brightness | sharpness | sharpness std |
|---|---|---:|---:|---:|
| **kai0_base** (D435i) | (131,135,132) | 132.7 | **46.3** | 21.2 |
| kai0_dagger (D435i) | (131,133,128) | 130.8 | 50.8 | 22.2 |
| **vis_base** (D405) | (116,118,115) ⬇12% | 116.5 ⬇12% | **31.9** ⬇**31%** | 15.1 |

### 3.3 解释
- D405 是**近距特化**相机, ISP/光学特性与 D435i 显著不同
- vis 图像系统性 **更暗 12%** + **锐度低 31%** — 这是硬件特征, 不是采集环境问题
- 训练于 D435i → 部署到 D405: 模型从未见过 (lower brightness, lower sharpness) 这个图像 mode → **真机视觉 OOD**

### 3.4 State 分布差异 (操作员习惯)

State std per joint (左臂 7 joint):
```
kai0_base: L=[0.197 0.514 0.482 0.305 0.238 0.274 0.033]
vis_base:  L=[0.258 0.553 0.430 0.492 0.316 0.501 0.030]
                                  ↑↑↑↑↑              ↑↑↑↑↑
                                  +61%               +83%
```

vis_base 操作员在 **joint 3 (腕滚转) 和 joint 5 (腕俯仰)** 上用了更大幅度 — 可能因 D405 窄视野需要更多手腕调整看清衣物。

---

## 4. vis_base 的"高 jump"采集 bug 详细分析

### 4.1 起始观察
`vis_base` max jump per ep p99 = **1.05 rad (~60°)** — 是 kai0_base (0.19 rad) 的 **5.4×**。

物理事实: piper 机械臂 30fps 单步不可能动 60°。这必是**采集 bug**。

### 4.2 全局扫描 (thresh > 0.5 rad)

```
扫描 895 个 vis_base ep
找到 91 个 ep 含 max_jump > 0.5 rad
─────────────────────────────────
高跳变率: 10.2%
最大跳变值: 2.17 rad (~125°!)
中位最大值: 0.84 rad (~48°)
```

### 4.3 按日期分布 (识别采集质量差的日子)

| Date | Bad eps | Total | Bad rate |
|---|---:|---:|---:|
| **2026-04-30-v2** | **24** | 83 | **29%** ⚠️ 最差 |
| 2026-04-24-v2 | 23 | 187 | 12% |
| 2026-05-06-v2 | 10 | 100 | 10% |
| 2026-05-08-v2 | 10 | 101 | 10% |
| 2026-04-28-v2 | 8 | 152 | 5% |
| 2026-04-25-v2 | 4 | 100 | 4% |
| 2026-04-29-v2 | 4 | 100 | 4% |
| 2026-05-09-v2 | 4 | 30 | 13% |
| 2026-05-07-v2 | 2 | 20 | 10% |
| 2026-04-23-v2 | 2 | 22 | 9% |

**04-30-v2 那天 29% 的 episodes 含故障** — 这天采集系统可能有硬件问题。

### 4.4 跳变特征 (50 个 top-jump ep 详细分析)

#### Type breakdown
| 类型 | 数量 | 比例 | 含义 |
|---|---:|---:|---|
| **Persistent step** (跳变后稳定) | 37 | **77%** | 操作员暂停-重定位事件 |
| **Single-frame spike** (孤立尖峰) | 11 | **22%** | 单帧异常, 下一帧回归 |
| **Tail** (末尾 3 帧) | 2 | 4% | episode 末尾 reset |

#### 多关节相关性 (强信号)
- **100% 至少 2 个关节同时跳变** (绝非单 joint 通信 bug)
- **85% 5+ 个关节同时跳变**
- **94% 双臂同时跳变** ← 决定性证据
- 平均一次跳变涉及 **6.4 个关节**

#### 跳变位置 (在 ep 中)
均匀分布 (前 20% / 中 62% / 后 16%), **不是录制启动问题**

#### 实例 (具体数据)
```
ep42 t=52→53: joint 1 突跳 0.49 → 2.32 (= 1.82 rad = 105°)
              joint 2 突跳 -0.95 → -1.78
              joint 3 突跳 0.25 → -0.49
              joint 5 突跳 -0.31 → 0.39
              (4 个 joint 同时, 不是单 joint 通信 bug)
              
t=54+: 稳定在 [-0.33, 2.32, -1.78, -0.49, ...] (新位置维持)
```

### 4.5 物理解释
最可能的根因: **采集帧丢失 + 操作员有意暂停-重定位**:
1. 操作员在 teleop 过程中**有意暂停**, 物理移动 master 手柄
2. 录制系统时间戳不连续, 中间几十帧未被记录
3. 录制日志显示"瞬间跳变 1-2 rad", 实际是**中间运动没被记录**
4. 跳变后稳定 (77%) 因为操作员到达新位置就停了

---

## 5. 残留跳变 (0.2-0.5 rad) 分析

### 5.1 用 thresh=0.5 清理后仍有残留

```
清理后 vis_base_clean: 827 ep
残留 0.2-0.5 jump: 629 个事件 (351 ep, 占 42%)
```

### 5.2 残留特征 vs 0.5+ 高跳变对比

| 维度 | 0.5+ (已清理) | **0.2-0.5 残留** | 含义对比 |
|---|---|---|---|
| 类型 | 77% persistent step | **73% spike** | 残留主要是孤立噪声 |
| 同时跳变 joints | 平均 6.4 (85% ≥5) | **平均 2.8 (40% ≤2)** | 残留更"局部" |
| 双臂同步 | 94% | **46%** | 残留多为单臂事件 |
| 位置末尾 10% | 16% | **40%** ⚠️ | 残留大量在末尾 |
| 位置中间 | 62% | 44% | 中间事件减少 |
| 主跳关节 | 肩 1, 9 (大幅度) | **肘 2, 腕 3** (小幅度) | 残留是腕/肘小关节 |

### 5.3 物理解释

#### Pattern A: 末尾"软" reset (40%)
末尾 10-15% 帧的 0.2-0.4 rad 跳变 — 操作员开始收回手臂、不连续动作。

#### Pattern B: 腕/肘小关节噪声 (30%)
joint 2 (肘) 22% + joint 3 (腕) 20% — piper 腕部编码器精度低, CAN bus 偶发丢包仅影响特定 motor。

#### Pattern C: 小型 persistent step (26%)
0.3-0.5 rad 多关节同时变化, 之后稳定 — 操作员小幅重定位 (不是大幅 pause)。

---

## 6. 修复策略 (已实施)

### 6.1 清理逻辑 (`~/tmp/kai0_diagnostic_scripts/clean_vis_base.py`)

```python
对每个 episode, 检测所有 max|a[t+1]-a[t]| > 0.5 rad 跳变, 分类:

Pass 1: 主跳变 (>0.5 rad)
  - persistent_step (中段): 整个 ep DROP
  - tail (末尾 15%): 截尾保留前段
  - spike (单帧): 线性插值 action[t+1] = (action[t]+action[t+2])/2

Pass 2: 末尾低阈值扫描 (>0.2 rad in 末尾 15%)
  - 找到第一个残留 → 截尾
```

### 6.2 清理结果

```
入: 895 ep / 1,063,168 frames
出: 837 ep / 963,177 frames (-6.5% ep, -9.4% frames)

分类:
- 607 keep (无任何 jump)
- 215 truncate (末尾被裁切)
- 12 interp (单帧 spike)
- 3 interp+truncate (双修)
- 58 drop (中段 persistent step)
```

### 6.3 清理效果对比

| 清理方案 | ep | max_jump p99 | vs kai0_base (0.19) |
|---|---:|---:|---:|
| **无清理** | 895 | **1.054** | **5.4×** |
| thresh 0.5 (tail 3 帧) | 827 | 0.580 | 3.0× |
| thresh 0.3 | 792 | 0.556 | 2.9× |
| **thresh 0.5 + tail 15%** (采用) | **837** | **0.442** | **2.3×** |
| (理想) kai0_base | — | 0.195 | 1.0× |

**最终采用 X1 方案**: 既保留了更多有效数据 (837 ep), 又显著降低 p99 (1.05 → 0.44)。

### 6.4 三个 smoothness 指标全面改善

| Metric | 无清理 | **v2 清理后** | kai0_base |
|---|---:|---:|---:|
| overall diff mean | 0.0061 | **0.0059** ≈ kai0 | 0.0058 |
| overall diff p99 | 0.0674 | **0.0650** | 0.0523 |
| max jump mean | 0.2752 | **0.1987** | 0.1268 |
| max jump p99 | 1.0541 | **0.4420** | 0.1945 |

**mean diff 已经几乎等同 kai0_base** (0.0059 vs 0.0058) — 整体平滑度恢复, 仅留少量边界 outlier。

---

## 7. 当前训练实验

### 7.1 数据集
- **A_new_smooth_800** (基于 vis_base_clean_v2 重 build)
- 路径: `/data/shared/tim/data/Task_A/A_new_smooth_800/{base,val}`
- Train: **811 ep / 929,942 frames**
- Val: **26 ep / 33,235 frames** (按日期 stratified split, 26 个跨 10 个 v2 日期)
- 视频用 symlink 复用原始 mp4

### 7.2 训练配置 `pi05_flatten_fold_a_new_smooth_800_new_norm`
```python
model            = Pi0Config(pi05=True)
init             = mixed_1_clean (/home/tim/local_ckpts/Task_A_init/mixed_1_clean/params)
batch_size       = 128
fsdp_devices     = 8
lr_schedule      = CosineDecay(warmup=1000, peak=1.5e-5, decay=50000, end=1.5e-6)
ema_decay        = 0.9999
num_train_steps  = 50_000
save_interval    = 2_000
keep_period      = 2_000
inline_eval_every= 2  (每 4000 步)
norm_stats       = 重新计算 (vis_base_clean_v2 数据分布)
```

### 7.3 启动状态 (持续更新)
- **服务器**: uc03 (8× A100-80GB)
- **状态**: 训练运行中
- **当前进度** (检查时刻): step **17,600 / 50,000 (35%)**, elapsed 11h15min
- **实测速率**: ~1.9-2.5s/step (加 `--num-workers 64` 后)
- **预计 ETA**: 剩余 ~17h (总计 ~28h)
- **Log**: `/data/shared/tim/logs/train_task_a_new_smooth_800_new_norm.log`

### 7.4 训练曾遇到的失败 (已修复)
1. **缺 `episodes_stats.jsonl`** → LeRobot fallback 到 HF hub → repo_id 格式校验失败
2. **`count` 字段维度错** (用 `[count]*14` 应该是 `[count]`) → LeRobot stats 校验失败
3. **`--num-workers` 默认 32 慢** → 加 `--num-workers 64` 速率从 4-5s/step → 1.9s/step

修复后启动成功并稳定运行。

---

## 8. 未解决的问题 / 后续待办

### 8.1 视觉域差 (D435i vs D405) 仍未根本解决
当前训练使用 **纯 D405 数据 (vis_clean)** + mixed_1 init。
- 优点: 视觉适配真机部署的 D405
- 风险: 数据规模仅 811 ep, 复杂场景覆盖可能不足

**后续候选**:
- 加 D435i→D405 图像增强 (brightness ↓12%, sharpness blur)
- 收集更多 D405 复杂场景数据
- 跑对比训练 (A: 纯 vis_clean; B: vis_clean × 3 + kai0_base × 1)

### 8.2 真机测试评估 protocol
建议分级测试 (每级 10 个 ep):
- Level 1: 平铺简单衣物 (baseline)
- Level 2: 部分皱缩衣物
- Level 3: 堆叠衣物
- Level 4: 不同颜色/纹理衣物

记录: 抓衣角成功率、动作平滑度、有无 cloth loop。

### 8.3 dagger 数据是否应过滤
- kai0_dagger 比 base 抖 62% (mean diff 0.0094 vs 0.0058)
- 历史 SOTA 训练用 kai0_advantage (类比 dagger) → val 好但真机抖
- 候选: 对 dagger 也做 thresh 0.5 清理, 看 smoothness 改善

### 8.4 是否需要 cluster training
单节点 8 GPU = 55h 太长。如果要快, 应用 uc 集群 24 GPU (FSDP=[1,24]) 缩到 ~18h。

---

## 9. 相关代码 / 文件清单

> **重要**: 诊断 / 清理 / 构建脚本**不进入项目仓库**, 仅作为本次诊断的一次性工具放在 temp 目录。本文档仅记录路径供溯源, 后续若需重跑请从 temp 复制。

### 9.1 诊断脚本 (一次性工具, temp 目录)
- 本地: `/home/tim/tmp/kai0_diagnostic_scripts/diagnose_cloth_loop.py` — cloth loop 闭环 rollout 诊断
- 本地: `/home/tim/tmp/kai0_diagnostic_scripts/analyze_dataset_diff.py` — 数据集 smoothness / image / state 对比
- 本地: `/home/tim/tmp/kai0_diagnostic_scripts/investigate_vis_jumps.py` — 高 jump 帧详细分析

uc01 / uc03 上对应路径: `/tmp/kai0_diagnostic_scripts/<同名>.py`

### 9.2 清理 + 构建脚本 (一次性工具, temp 目录)
- 本地: `/home/tim/tmp/kai0_diagnostic_scripts/clean_vis_base.py` — vis_base 清理 (thresh 0.5 + tail 15%)
- 本地: `/home/tim/tmp/kai0_diagnostic_scripts/build_task_a_new_smooth_800.py` — 合并为单 LeRobot dataset
- uc01 / uc03: `/tmp/kai0_diagnostic_scripts/<同名>.py`

### 9.3 训练 (持续, 进入项目)
- 配置: `kai0/src/openpi/training/config.py` 中 `pi05_flatten_fold_a_new_smooth_800_new_norm`
- 启动命令: `python scripts/train.py pi05_flatten_fold_a_new_smooth_800_new_norm --exp-name task_a_new_smooth_800_new_norm --num-workers 64 --overwrite`
- Log: uc03 上 `/data/shared/tim/logs/train_task_a_new_smooth_800_new_norm.log`

### 9.4 输出产物 (远程 uc 服务器, 数据/结果)
- `/data/shared/tim/diagnostic_cloth_loop_v1/` — cloth loop 诊断结果 (summary.json + per-ep)
- `/data/shared/tim/dataset_diff_analysis/` — 原始数据集对比 (清理前 baseline)
- `/data/shared/tim/dataset_diff_v2/` — 清理后对比 (smoothness 验证)
- `/data/shared/tim/vis_jump_inspect/` — 高 jump 帧分析 + thumbnails
- `/data/shared/tim/data/Task_A/vis_base_clean_v2/` — 清理后 per-date LeRobot dataset (837 ep)
- `/data/shared/tim/data/Task_A/A_new_smooth_800/` — 训练用合并 dataset (811 train + 26 val)

### 9.5 脚本迁移记录 (2026-05-15)
原本同步到 `/home/tim/workspace/deepdive_kai0/scripts/` 与 `/data/shared/tim/workspace/deepdive_kai0/scripts/`, 后将所有 5 个一次性脚本移至各机器的 temp 目录, 删除空的 `scripts/` 子目录, 保持项目根目录整洁。
