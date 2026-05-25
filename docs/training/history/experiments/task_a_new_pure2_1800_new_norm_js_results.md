# Task A new_pure2_1800_new_norm_js 训练结果 (js 集群 16-GPU HSDP)

> **结论先行**: js03+js04 (16 GPU HSDP) 上 `task_a_new_pure2_1800_new_norm_js` 从 step 0 跑到 49999, final **MAE@1 = 0.0090**。**未能超过 pure2_1800_6000 (0.0085)** — 1800 ep `-new` 数据 + mixed_1_clean init 不如 7900 ep 大杂烩 + pi05_base init。同时比 `task_a_new_pure_200_new_norm` (200 ep + mixed_1_clean, MAE@1=0.0065) 差 **+38.5%**, 进一步确证 **数据质量 + 精选 > 数据规模** 在 mixed_1_clean init 上仍成立。

## 1. 实验配置

| 参数 | 值 |
|---|---|
| Config name | `pi05_flatten_fold_a_new_pure2_1800_js` |
| Model | pi05 (`Pi0Config(pi05=True)`) |
| **Init** | `/mnt/data/tim/kai0_ckpts/Task_A/mixed_1_clean/params` (MA-merged Task_A warmed-up) |
| **Dataset** | `/mnt/data/tim/data/Task_A/self_built/A_new_pure2_1800` (1800 ep, JuiceFS:visincept) |
| - 来源 | `-new` 日期限定 + hflip mirror 增强, 仅高质量子集 (实际 1790 ep) |
| Val | `/mnt/data/tim/data/Task_A/self_built/A_new_pure2_1800_val` |
| Prompt | "Flatten and fold the cloth." |
| `use_delta_joint_actions` | False |
| LR schedule | Cosine, warmup=1k, peak_lr=1.5e-5, decay_steps=50k, decay_lr=1.5e-6 |
| EMA decay | 0.9999 |
| Steps | 50,000 |
| Batch | **80** (16 GPU × 5/GPU, 2-host HSDP) |
| FSDP | fsdp_devices=8 (per host), HSDP `[2,8]` (2 host × 8 GPU) |
| Save | every 2,000 step, keep_period=2,000, max_to_keep=1 |
| inline_eval | every 2 saves (= 每 4k 步), 200 frames |
| Seed | 42 |
| Servers | **js03 + js04** (各 A800-SXM4-80GB × 8, 共 16 GPU) |
| 网络 | JuiceFS:visincept (POSIX 一致, orbax 跨主机 ckpt) |
| WandB | offline |
| 训练时长 | 19h10m (2026-05-14 08:09 → 2026-05-15 03:22 CST, 含 final eval 24min) |

### 1.1 关键差异 (vs 老 SOTA `pure2_1800_6000`)

| 维度 | 本实验 pure2_1800_js | pure2_1800_6000 (老 SOTA) |
|---|---|---|
| **数据规模** | **1800 ep** (仅 `-new` 限定) | 7900 ep (1790 mirror + 3055 base + 3055 advantage) |
| **Init** | **mixed_1_clean** (Task_A 微调) | pi05_base (原始) |
| 集群 | js03+04 (16 GPU, JuiceFS) | uc01+02+03 (24 GPU, NFS) |
| Mesh | HSDP `[2,8]` | FSDP `[1,24]` |
| Batch | 80 | 120 |
| Final MAE@1 | 0.0090 | **0.0085** |
| Gap | **+5.9% (worse)** | — |

### 1.2 关键差异 (vs NEW SOTA `task_a_new_pure_200_new_norm`)

| 维度 | 本实验 pure2_1800_js | NEW SOTA pure_200 |
|---|---|---|
| **数据规模** | 1800 ep | **200 ep** (1/9 of this) |
| Init | mixed_1_clean | mixed_1_clean (相同!) |
| Batch | 80 | 120 |
| Final MAE@1 | 0.0090 | **0.0065** |
| Gap | **+38.5% (worse)** | — |

## 2. 完整 inline-eval MAE@{1,10,25,50} 曲线

| step | MAE@1 | @10 | @25 | @50 | Δ@1 vs prev | eval (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 4000  | 0.0130 | 0.0339 | 0.0663 | 0.1123 | (start, mixed_1_clean 起点) | 1598 |
| 8000  | 0.0119 | 0.0285 | 0.0515 | 0.0820 | -8.5% | 1415 |
| 12000 | 0.0109 | 0.0257 | 0.0447 | 0.0681 | -8.4% | 1424 |
| 16000 | 0.0103 | 0.0240 | 0.0401 | 0.0590 | -5.5% | 1449 |
| 20000 | 0.0100 | 0.0227 | 0.0367 | 0.0522 | -2.9% | 1434 |
| 24000 | 0.0097 | 0.0216 | 0.0338 | 0.0471 | -3.0% | 1428 |
| 28000 | 0.0096 | 0.0206 | 0.0316 | 0.0434 | -1.0% | 1467 |
| 32000 | 0.0094 | 0.0198 | 0.0298 | 0.0405 | -2.1% | 1430 |
| 36000 | 0.0093 | 0.0192 | 0.0283 | 0.0383 | -1.1% | 1442 |
| 40000 | 0.0092 | 0.0186 | 0.0270 | 0.0363 | -1.1% | 1459 |
| 44000 | 0.0091 | 0.0181 | 0.0260 | 0.0348 | -1.1% | 1418 |
| 48000 | 0.0090 | 0.0177 | 0.0251 | 0.0336 | -1.1% | 1428 |
| **49999** | **0.0090** | **0.0175** | **0.0247** | **0.0328** | 0.0% | 1397 |

**Best**: step 49999, MAE@1 = **0.0090**

## 3. 训练动力学

- mixed_1_clean 起点 (step 4k) MAE@1=0.0130 (vs pi05_base 起点 step 4k=0.0534 在 `pure2_1800_6000`) — Task_A warmed-up init 起点低 4.1x
- 但收敛后 final MAE@1=0.0090 vs pure2_1800_6000 final 0.0085 — **mixed_1_clean 上限低 5.9%**, 印证"warmed-up init 上限不如干净起点"现象
- 全程单调下降 (无反弹), step 28k 之后进入慢速 plateau (~每 4k step -1.0%)
- @50 持续改善到 49999 (0.1123 → 0.0328, **-71%**)
- eval 时长 1397-1598s, 比 uc 集群同任务 (~600s) 慢 2-3x — JuiceFS read I/O 是瓶颈, NFS uc 集群在 eval 上更快
- 全程 0 NaN, param_norm 稳定

## 4. 关键洞察

1. **mixed_1_clean init 在 1800 ep 上没法超过 pi05_base init 在 7900 ep 上**: 同样的 50k 步, 数据量 +4.4x + 干净 init 反而 final MAE 低 5.9%。说明:
   - 当数据规模足够大 (7900 ep) 时, **干净起点 (pi05_base)** 比 warmed-up init 更优
   - 但 `pure_200` (200 ep, mixed_1_clean init, MAE 0.0065) 反过来超过 `pure2_1800_6000`, 说明数据**规模到 7900 时, 干净起点重要; 数据小到 200 时, warmed-up init 重要**
2. **数据规模并非线性增益**: 200 ep → 1800 ep (9x), 但 MAE 反而恶化 +38.5% (从 0.0065 到 0.0090, 同 mixed_1_clean init)。说明 200 ep 已经触及 Task_A `-new` 数据的"信息上限", 加更多 same-distribution 数据反而稀释信号。可能的解释:
   - 200 ep 是精选的最有代表性子集, 1800 ep 加入了边缘 case
   - mixed_1_clean init 已经熟悉 Task_A 分布, 200 ep 精选数据足以细化, 1800 ep 反而引入噪声
3. **HSDP vs FSDP 集群**: HSDP `[2,8]` 在 js 双机集群成功, 没有 uc 集群 24 GPU 上 HSDP `[3,8]` 105 分钟 SPMD partitioner 死锁问题 — 可能与 host 数 (2 vs 3) 相关。
4. **eval 时长瓶颈**: 1400+ s/eval (vs uc 集群 600s) 让总训练时长涨到 19h, 比 uc 集群同步数训练 (~13h) 慢 47%。

## 5. 最佳 ckpt 位置

```
js03:/mnt/data/tim/checkpoints/pi05_flatten_fold_a_new_pure2_1800_js/task_a_new_pure2_1800_new_norm_js/49999/
```

⚠️ JuiceFS:visincept 共享路径 — js01-js04 任一节点都能直接读。

**完整 ckpt 列表** (每 2k 步): 2000, 4000, 6000, ..., 48000, 49999  
**附属文件**: 同目录 `norm_stats.json`

## 6. 与 SOTA 对比矩阵

| 排名 | 实验 | Data | Init | Batch | Cluster | Final MAE@1 |
|---|---|---|---|---|---|---|
| 🥇 | `task_a_new_pure_200_new_norm` | 200 ep `-new` | mixed_1_clean | 120 | js02 (8 GPU) | **0.0065** ⭐ |
| 🥈 | `task_a_new_pure2_1800_6000_new_norm` | 7900 ep mix | pi05_base | 120 | uc 24 GPU | 0.0085 |
| 🥉 | **`task_a_new_pure2_1800_new_norm_js`** (本实验) | **1800 ep `-new`** | **mixed_1_clean** | 80 | **js 16 GPU** | **0.0090** |
| 4 | `mix_b6000_p1200_init_mixed_1` | 7200 ep mix | mixed_1 | 120 | uc01 (8 GPU) | 0.0108 |

## 7. 后续计划

- 不建议把 49999 ckpt 上 sim01 (老 SOTA pure2_1800_6000 已优于本实验)
- 若想进一步推 SOTA: 优先扩展 pure_200 (尝试 pure_400 / pure_800 同 init 看是否能压过 0.0065)
- mixed_1_clean init 配合 1800 ep 的组合不太可能超过 pure_200 0.0065, 建议放弃这条线
