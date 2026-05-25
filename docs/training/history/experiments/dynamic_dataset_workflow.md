# 动态数据集训练流程（Task_A mixed）

> 无源码改动的方案：外部 watcher 监测数据变化，触发 kill → rebuild → `--resume` 循环

## 背景 / 需求

训练 Task_A `pi05_flatten_fold_mixed_visrobot01` 时希望：

- visrobot01 持续采集 → 新 episode 不断进入 `/vePFS/visrobot01/KAI0/Task_A/2026-MM-DD/{base,dagger}/`
- 每次 save_interval 边界做 inline_eval 后，**自动检测数据增长**
- 若有新 episode：重新 build mixed 数据集（按规则从 old base/dagger 各抽 N 配平）
- 重新计算 norm_stats，以 `--resume` 接续之前的 ckpt 继续训练

**挑战**：openpi 训练循环在启动时就固定了 dataloader。中途替换需要改源码（侵入大且危险）。

## 方案 A（采纳）：外部 watcher + `--resume`

利用 openpi `--resume` 原生能力：
- 从最新 ckpt 加载 `train_state`（step 计数、optimizer、EMA 权重、LR schedule 位置）
- 按 config 重建 dataloader，读取当前数据目录的新内容
- 训练继续，仿佛未中断

**唯一代价**：每次 rebuild 有一次 "kill + restore + XLA recompile" overhead（约 30-60s）。

### 为什么 `--resume` 透明切换数据安全

关键事实 1：**norm_stats 是在启动时从数据目录读的**（见 `policy_config.py:64`）：
```python
norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
# data_config.asset_id 实际是 repo_id（绝对路径）
# → 读的是 $DATA_ROOT/Task_A_mixed_gf1/base/norm_stats.json
```
每次 resume 重新读当前 norm_stats，自动对齐新数据分布。

关键事实 2：**episodes.jsonl 在 LeRobotDataset 初始化时重扫**。resume 时 dataloader 重建即触发，自动吸收新的 episode 列表。

关键事实 3：**EMA / optimizer state 存于 `train_state/`**，由 orbax 完整恢复，不受数据变化影响。

### 副作用 / 注意

1. `--resume` 加载 **params/** 为 EMA 权重（openpi 特殊规则）。继续训练时 `train_state` 内还有 live params，实际训练节奏不变。

2. dataloader `num_workers` fork 子进程在 kill 时偶有残留，新训练启动前确保清理（脚本内用 `pkill -9` 兜底）。

3. norm_stats 变化会让 EMA/live params 的动作输出稍有偏移——不是 bug，是正确行为（新数据需要新归一化）。

## 组件

| 文件 | 用途 |
|---|---|
| `train_scripts/data/build_task_a_mixed.py` | 构建 3 源混合数据集（visrobot01 + existing base + dagger），加 `--val-size` 切 val 分层抽样 |
| `train_scripts/data/generate_episodes_stats.py` | v2.1 required per-episode stats |
| `kai0/scripts/compute_norm_states_fast.py` | 全局 norm_stats（state/action mean/std/q01/q99）|
| `train_scripts/launch/run_taska_mixed_gf1.sh` | 初次启动（用 `--overwrite`）|
| **`/tmp/dynamic_dataset_train.sh`** | **本方案核心 watcher** |

## 工作流

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 0: 首次启动 (人工)                                         │
│    ① build_task_a_mixed.py --val-size 21                        │
│    ② generate_episodes_stats.py base + val                      │
│    ③ compute_norm_states_fast.py                                │
│    ④ nohup bash run_taska_mixed_gf1.sh &                        │
│       (用 --overwrite 创建新实验目录)                              │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: watcher 启动 (人工)                                     │
│    nohup /tmp/dynamic_dataset_train.sh > /tmp/dyn_train.log &   │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: 常态循环 (自动)                                         │
│  每 30s poll /tmp/train_taska_mixed.log 看 inline-eval 新行      │
│  有新 eval → 比对 visrobot01 完整 ep 数 vs manifest 中记录          │
│                                                                   │
│  if (new_eps >= MIN_NEW_EPS=3) and (since_last_rebuild >= 15min)│
│    → rebuild + resume 循环:                                      │
│      step 1/5: pkill train.py                                    │
│      step 2/5: build_task_a_mixed.py --force  (rebuild)          │
│      step 3/5: generate_episodes_stats.py base + val             │
│      step 4/5: compute_norm_states_fast.py                       │
│      step 5/5: nohup python train.py <config> --resume &         │
│                                                                   │
│  循环继续...                                                       │
└─────────────────────────────────────────────────────────────────┘
```

## 触发条件（可调）

脚本顶部：

| 变量 | 默认 | 含义 |
|---|---|---|
| `POLL_SEC` | 30 | 检查 log 的间隔 |
| `MIN_NEW_EPS` | 3 | visrobot01 至少新增 N 个完整 ep 才触发 rebuild（避免每 1 ep 抖动）|
| `MIN_REBUILD_INTERVAL` | 900 | 两次 rebuild 间 ≥15 min（避免短时间内连续重启）|
| `VAL_SIZE` | 21 | val 集大小（每源 7 ep，分层抽）|

## 使用步骤

### 1. 首次启动（人工操作）

```bash
# 在 sim01 或 gf1：
ssh -p 11111 tim@14.103.44.161

# 第 1 次手动 build + 启动
cd /vePFS/tim/workspace/deepdive_kai0
PYTHON=/home/tim/workspace/deepdive_kai0/kai0/.venv/bin/python

# 构建数据
$PYTHON train_scripts/data/build_task_a_mixed.py --val-size 21 --force
$PYTHON train_scripts/data/generate_episodes_stats.py \
    /vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A_mixed_gf1/base
$PYTHON train_scripts/data/generate_episodes_stats.py \
    /vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A_mixed_gf1/val
source setup_env.sh; export KAI0_DATA_ROOT OPENPI_DATA_HOME PYTORCH_CKPT_BASE
cd kai0; $PYTHON scripts/compute_norm_states_fast.py --config-name pi05_flatten_fold_mixed_visrobot01; cd ..

# 启动训练（前台，log 到 /tmp/train_taska_mixed.log）
nohup bash train_scripts/launch/run_taska_mixed_gf1.sh > /tmp/train_taska_mixed.log 2>&1 &
```

### 2. 启动 watcher（人工，一次性）

```bash
# 把脚本 scp 到 gf1 的 /tmp/
# 运行:
nohup /tmp/dynamic_dataset_train.sh > /tmp/dyn_train.log 2>&1 &
```

之后全自动：

- watcher 每 30s 检查 visrobot01 新数据
- 数据增长 ≥3 ep 且 ≥15 min 间隔 → 自动 rebuild + resume
- 训练 log 持续写入 `/tmp/train_taska_mixed.log`（累积所有 resume 轮次）
- Watcher log `/tmp/dyn_train.log` 记录所有周期

### 3. 停止 watcher（不停训练）

```bash
pkill -f dynamic_dataset_train.sh
```

### 4. 停止训练

```bash
pkill -f 'train.py pi05_flatten_fold_mixed_visrobot01'
```

## 监控示例日志（预期）

```
[08:00:03] dynamic dataset watcher started
[08:00:03] poll=30s  min-rebuild-interval=900s  min-new-eps=3
[08:00:03] config=pi05_flatten_fold_mixed_visrobot01 exp=mixed_visrobot01_v1
[08:15:27] eval@step=500  visrobot01 complete=80  in-dataset=80  new=0
[08:30:41] eval@step=1000 visrobot01 complete=80  in-dataset=80  new=0
...
[10:45:12] eval@step=5000 visrobot01 complete=95  in-dataset=80  new=15
[10:45:12] *** trigger: +15 new eps available, last rebuild 0s ago ***
[10:45:12] >>> rebuild+resume cycle start
[10:45:12] step 1/5: killing training process...
[10:45:20] training killed
[10:45:20] step 2/5: rebuilding mixed dataset...   # ~2 min
[10:47:18] step 3/5: regenerating episodes_stats...
[10:47:45] step 4/5: recomputing norm_stats...     # ~5s
[10:47:50] step 5/5: launching --resume...
[10:47:50] <<< rebuild+resume cycle complete
[10:48:20] eval@step=5000 visrobot01 complete=95  in-dataset=95  new=0   # next eval
```

## 成本估算（每次 rebuild）

| 步骤 | 耗时 |
|---|---|
| pkill + 清理 | 5-15s |
| build_task_a_mixed.py | 1-3 min（取决于数据量）|
| generate_episodes_stats.py × 2 | 30-60s |
| compute_norm_states_fast.py | 5s |
| JAX/XLA restore + recompile | 30-60s |
| **总计** | **2-5 min** |

Assuming rebuild every 1000-2000 steps (every eval if data 持续增长)，每小时 loss 几分钟训练时间，可接受。

## 与全量 restart 的区别

| 方案 | LR schedule | EMA | optimizer state | 数据 |
|---|---|---|---|---|
| **本方案 `--resume`** | 继续 | 保留 | 保留 | 新数据 |
| 全量 `--overwrite` | 从头 warmup | 重置为 init | 重置 | 新数据 |

保留 optimizer 历史（Adam 的 m/v 矩阵）+ EMA 累积 = 训练不会"倒退"，只是数据换壳继续。

## 可能的故障模式

1. **build 失败**（数据格式错）→ watcher 日志记录 `[ERROR] build failed`，**训练保持停止**直到人工介入。不会偷偷启动错的训练。

2. **norm_stats 计算失败** → 同上，停止。

3. **--resume 找不到 ckpt** → 第一次训练尚未 save 就数据更新，resume 失败。脚本内 `--resume` 要求至少有 1 个 ckpt。若这种场景 → 改成 `--overwrite` 新启（需改 script）。

4. **watcher 自己崩了** → 训练继续不受影响，只是不再自动 resume。人工 `nohup ./dynamic_dataset_train.sh` 即可续上。

5. **两个 watcher 并发** → 若不小心起了 2 个 watcher，会抢 rebuild。脚本没加 lockfile 保护；用户应避免。

## 未来扩展

- **扩展到其他任务**：Task_P/E 同理，修改顶部路径变量即可复用。
- **智能触发**（更精细）：除了 ep 数量增长，还可监测 `best_val_mae` 不再下降时主动 rebuild（防止过拟合当前数据）。
- **并发 val**：rebuild 时不切出 val，用固定 held-out + rotating rebuild 的 train，保证 MAE 跨 rebuild 可比。

---

_方案决策：_ 源码改动 = 高风险（改 train loop 数据循环需动 JAX jit edge case）。外部编排 = 低风险（只重启 process，拿 --resume 的原生能力）。推荐 A 方案作为生产级实现。

_文档: 2026-04-24_
