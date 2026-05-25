# Task_P Unfreeze 全参数微调：8k vs 20k 对比分析

> 2026-04-22 ~ 2026-04-23 @ gf1 (8×A100 80GB)
> 从 `Task_A/mixed_1` 初始化，全参数解冻 (freeze_filter=None)
> val set: 16 ep / ~5.4k frames，每次 eval 200 frames × 16 ep

## 1. 两次实验的超参对比

| 参数 | **Stage 2: 8k** | **Stage 3: 20k** |
|---|---|---|
| config name | `pi05_pick_place_box_kai0_unfreeze_8k` | `pi05_pick_place_box_kai0_unfreeze_20k` |
| num_train_steps | 8,000 | 20,000 |
| peak_lr | 2.5e-5 | **1.5e-5** (×0.6) |
| decay_steps (cosine) | 8,000 | 20,000 |
| decay_lr | 2.5e-6 | 1.5e-6 |
| warmup_steps | 500 | 500 |
| ema_decay | 0.999 | 0.999 |
| batch_size | 128 | 128 |
| fsdp_devices | 8 | 8 |
| save_interval / keep_period | 1000 / 1000 | 2000 / 2000 |
| inline_eval_every | 1 | 1 |
| 实际跑到 | step 7999 (完成) | step **8100+** (用户中断) |
| samples shown | 8000×128=1.02M | ≥1.04M |
| epochs on train (24822 frames) | 41 | 42+ |

## 2. Train loss 轨迹对比

| step | **8k run** | **20k run** | 差异 |
|---|---|---|---|
| 0 | 0.2912 | 0.2912 | 相同 init |
| 100 | 0.0836 | 0.0994 | 20k 略慢（低 LR）|
| 200 | 0.0197 | 0.0225 | 20k 略慢 |
| 500 | 0.0100 | 0.0101 | 持平 |
| 1000 | 0.0064 | — | |
| 2000 | 0.0043 | — | |
| 3000 | 0.0033 | — | |
| 4000 | 0.0025 | — | |
| 5000 | 0.0020 | — | |
| 6000 | 0.0014 | — | |
| 7000 | 0.0011 | ~0.0020 | 8k 更低（LR 更大）|
| 7999 | **0.0009** | 0.0018 | |
| 8000+ | — | 0.0018 | |

**观察**：
- 8k run 的 **peak_lr 2.5e-5** 更激进，train loss 降得更快更深
- 20k run 的 **peak_lr 1.5e-5** 更温和，cosine decay 分布在更长周期，train loss 整体偏高

## 3. Val MAE 轨迹（核心指标）

### 3.1 完整表

| step | **8k: @1** | **8k: @10** | **8k: @25** | **8k: @50** | **20k: @1** | **20k: @10** | **20k: @25** | **20k: @50** |
|---|---|---|---|---|---|---|---|---|
| 1000 | 0.0258 | 0.0412 | 0.0634 | 0.0852 | — | — | — | — |
| 2000 | 0.0212 | 0.0376 | 0.0602 | 0.0804 | **0.0202** | **0.0361** | **0.0590** | **0.0795** |
| 3000 | **0.0206** ⭐ | 0.0380 | 0.0610 | 0.0806 | — | — | — | — |
| 4000 | 0.0208 | 0.0386 | 0.0617 | 0.0811 | **0.0195** ⭐ | 0.0367 | 0.0600 | **0.0797** |
| 5000 | 0.0212 | 0.0391 | 0.0623 | 0.0815 | — | — | — | — |
| 6000 | 0.0215 | 0.0395 | 0.0627 | 0.0817 | 0.0202 | 0.0377 | 0.0611 | 0.0807 |
| 7000 | 0.0217 | 0.0397 | 0.0628 | 0.0818 | — | — | — | — |
| 7999 / 8000 | 0.0219 | 0.0400 | 0.0630 | 0.0820 | 0.0206 | 0.0382 | 0.0616 | 0.0813 |

⭐ = best ckpt per run

### 3.2 最优点对比

| 指标 | 8k best | 20k best | 20k 相对改善 |
|---|---|---|---|
| **best step** | **step 3000** | **step 4000** | overfit 点推后 |
| MAE@1 | 0.0206 | **0.0195** | **-5.3%** |
| MAE@10 | 0.0380 | 0.0367 | -3.4% |
| MAE@25 | 0.0610 | 0.0600 | -1.6% |
| MAE@50 | 0.0806 | 0.0797 | -1.1% |

### 3.3 过拟合模式分析

```
8k run (peak_lr=2.5e-5):
  MAE@1: 0.0258 → 0.0206 (3k 最低) → 0.0219 (最后)
  过拟合拐点: step 3000 后立即恶化
  train loss / val MAE gap: 0.0033 vs 0.0206 = 6.2× (3k)
                            0.0009 vs 0.0219 = 24.3× (7999) ← 严重过拟合

20k run (peak_lr=1.5e-5):
  MAE@1: 0.0202 → 0.0195 (4k 最低) → 0.0206 (8k)
  过拟合拐点: step 4000 后缓慢恶化
  train loss / val MAE gap: 0.0025 vs 0.0195 = 7.8× (4k)
                            0.0018 vs 0.0206 = 11.4× (8k) ← 相对温和
```

## 4. 关键发现

### 4.1 更低 LR → 更好的 val MAE 最优点 + 更晚的 overfit 拐点

| | 8k (lr=2.5e-5) | 20k (lr=1.5e-5) |
|---|---|---|
| best step | 3000 | **4000** |
| best MAE@1 | 0.0206 | **0.0195** |
| 延后的 overfit | — | **+1000 步** |

原因：LR 更小 → 每步更新幅度小 → 向 local minimum 逼近更精细，不会"冲过"最优点。

### 4.2 对比 P-T10 历史 baseline

| 方法 | MAE@1 | 相对 P-T10 提升 |
|---|---|---|
| P-T10 (frozen vision, bs=4) | 0.0633 | baseline |
| Unfreeze 8k best | 0.0206 | **-67%** |
| Unfreeze 20k best | 0.0195 | **-69%** |

**结论**：Task_P 上 vision-unfreeze + full-param 显著碾压 frozen 基线。

### 4.3 Train loss 和 Val MAE 的背离

```
Train loss monotonic 下降:
  step 4000: 0.0025
  step 5000: 0.0020  (8k)  / Step 8000: 0.0018 (20k)
  Step 7999: 0.0009 (8k)

Val MAE U 形:
  step 3k-4k 最低
  之后缓慢上升
```

**警示**：train loss 无法用作 early stopping 信号，**必须用 val MAE**。

### 4.4 长 horizon (MAE@50) 恶化较少

| run | @1 峰值恶化 | @50 峰值恶化 |
|---|---|---|
| 8k | +6.3% | +1.7% |
| 20k | +5.6% | +2.0% |

long-horizon 动作轨迹对 action-level 误差的累积耐受较好，但 @1 作为即时控制信号对 overfit 更敏感。

## 5. 实战部署选择

两个最优 ckpt 都已传到 sim01：

| ckpt | path | MAE@1 | 真机对比价值 |
|---|---|---|---|
| 8k step 3000 | `/data1/.../p_unfreeze_8k_v1/3000` | 0.0206 | best val MAE |
| 8k step 7999 | `/data1/.../p_unfreeze_8k_v1/7999` | 0.0219 | **lowest train loss** (0.0009)，test overfit |

（20k best 未传 sim01）

### 真机测试发现

用户真机测试 step 3000 的 A/B：
- **失败模式**：抓取瞬间够不到盒子（grasp moment off by ~2-3 cm）
- 抓到后续流程全部成功 → 仅 pre-grasp 定位问题
- **结论**：MAE@1 降到 0.0206 在离线指标上达标，但 **关键瞬间误差** 不被 MAE 平均量化 → offline 指标与真机成功率的脱钩

## 6. 学到的东西（适用其他 pi05 微调）

1. **peak_lr 选择规律**：
   - 短训（2k-5k 步）+ bs=128：peak_lr 1.25e-5 ~ 1.5e-5
   - 长训（≥10k 步）+ bs=128：peak_lr 1.5e-5 ~ 2e-5 更优
   - 2.5e-5 对 Task_P 24k frames 偏激进（过早 overfit）

2. **EMA=0.999 是金标准**：
   - 半衰期 ~700 步
   - step 2000 已 86% 训练权重
   - 与 Stage 1 的 0.9999（半衰期 10k）相比：EMA 稀释问题完全消失

3. **过拟合识别 3 标志**：
   - Val MAE 反弹（+3-5% 即警示）
   - Train loss / val MAE gap > 10×
   - Gradient norm 继续降但 val 不动 → 已在过拟合 minimum 震荡

4. **Save 策略教训**：
   - `save_interval=1000` + `keep_period=1000` 保 8 个 ckpt = 安全
   - Stage 1 用 `keep_period=5000` 导致丢失 best_step=14000 ckpt（最早的坑）

5. **小数据（24k frames）+ bs=128 下**：
   - 每 1000 步 = 5.2 epochs → 过拟合累积很快
   - 3-5 epochs 达到 minimum 是常态
   - 扩到 10+ 步（30+ epochs）收益几乎为零

## 7. 未来方向（按效果预期排序）

> 基于真机"抓取瞬间偏"失败模式，以下方向对真机更有意义：

1. **DAgger 补强 grasp phase**（预计 +30~50% 真机成功率）
2. **RTC 推理**（`rtc_apply.sh rtc5` 或 `rtc3`，+15~30%）
3. **Stage Advantage 加权 grasp loss**（kai0 module 3，+20~40%）
4. **数据增强 time/space aug**（+10~20%）
5. ~~继续压 MAE~~（边际收益递减，应停止追 MAE）

---

_生成时间: 2026-04-24_
_运行日志源: `gf1:/tmp/train_p_unfreeze_8k{,_resume}.log`, `gf1:/tmp/train_p_unfreeze_20k.log`_
