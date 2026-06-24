# 数据集裁剪方法 & PTS 归零 — 方法与注意事项

> **建立**: 2026-06-22（由 Exp-C 真机抓取回归根因调查沉淀）
> **适用**: 所有对 lerobot v2.1 数据集做"裁帧"的处理(v3 前裁/尾裁、idle-downsample、tail-cap 等)
> **铁律**: **任何裁掉视频头部帧的操作,必须把视频 PTS 归零**。否则 lerobot 按时间戳解码会**静默取错帧** → 训练时视觉↔动作错位 → **真机失败,而 offline MAE 完全测不出**。
> **关联**: [[reference_v3_trim_video_pts_bug]] · [`dagger_validity_and_finetune_comparison.md` §8.10](../../training/future_plans/plans/dagger_validity_and_finetune_comparison.md) · 脚本 `train_scripts/kai/data/build_no_release.py` / `reset_video_pts.py`

---

## 1. 裁剪方法（v3 = 前裁 + 尾裁）

裁剪由 `train_scripts/kai/data/build_no_release.py` 完成,输出 `kai0/data/Task_A/vis_base/v3/<date>-v3/`。

| 步骤 | 做什么 | 机制 |
|---|---|---|
| **① 前裁(motion_onset, `--mode no_release`)** | 裁掉每条 episode 开头"投放等待/静止头" | `onset` = 首个"12 臂维 mean\|Δaction\| 连续 WIN 帧 > THR"的帧;`cut = max(0, onset − MARGIN)`;丢 parquet 行 `[0:cut]` + 视频裁掉前 `cut` 帧 |
| **② 尾裁(tail-cap, Step 3, 2026-06-16)** | 裁掉末端"任务完成后的长静止尾",保留 N 帧 terminal-settle | 就地裁 v3 末端长尾,保留少量收尾帧 |
| **校验(必须)** | `assert video_frames == parquet_rows` | 裁后逐 episode 视频帧数 == parquet 行数 |

视频裁帧用 `trim_video_pyav()`(re-encode,crf18 + veryfast,近无损)。

### 1.1 在线录制器：采集即标准对齐（不再需要离线裁剪/归零）

> **2026-06-22 起**:`web/data_manager/backend/app/dataset_writer.py::EpisodeWriter`(teleop / dagger / autonomy 三条采集链路**唯一**写盘器)已把上面整套裁剪 + 对齐**搬到录制时在线完成**,采集出的 V3 episode **直接符合标准、可直接训练**,**无需再事后跑 `build_no_release.py` / `reset_video_pts.py`**。

| 机制 | 实现 |
|---|---|
| **前裁** `KAI0_FRONT_TRIM=1` | 滚动缓冲 + onset 检测,保留 15 帧 lead-in。被丢的头帧**从不写入**视频/parquet |
| **尾裁** `KAI0_TAIL_TRIM=1`(默认跟随 FRONT_TRIM) | `_stage_tick` 扣住末尾连续 idle(arm `mean\|Δ\|>3e-3` **且** grip `max\|Δ\|>0.02` 都静止才算 idle → 末尾松手/放置永不被裁),finalize 只留 `TAIL_CAP=15` 帧收尾;中段 idle 不抽稀。逐帧等价 `build_no_release.tail_cap_keep_indices` |
| **PTS 零基(天然满足铁律)** | `frame.pts = frame_index`(只数保留帧)→ 首帧 pts=0,**头帧从不写入故无需事后归零** |
| **parquet timestamp** | `_write_parquet` 写 `timestamp = frame_index / fps`(**非** wall-clock `ts−t0`)。若沿用 wall-clock,前裁后 `timestamp[0]≈cut/fps≠0` → 与零基 PTS 反向错位(§2 同类 skew 的 parquet 轴版本) |
| **逐 ep 自检** `KAI0_VALIDATE_TRIM=1` | finalize 重解码 3 路 mp4,断言首帧 `pts==0` 且 `video帧数==parquet行数`(§4)。默认关(重解码有开销),验证/抽查时打开 |

⚠️ `autonomy_recorder`(诊断录制)默认 **不开** trim,保持原始采样。

**采集帧率(`KAI0_ASYNC_WRITER=1`,默认开)**:30Hz 采集线程只做「取帧 + 入队」,后台 writer 线程做 encode + 深度压缩,慢 tick(NVENC/IO 停顿、与 API server 抢 GIL)不再丢帧——修复实测 27fps 欠采样(标 30fps 实 27fps → 动作快 1.11×)。输出与同步路径**逐字节一致**。`KAI0_ASYNC_WRITER=0` 回退。⚠️ 欠采样不破坏帧↔动作对齐,但会让动作显得偏快,故仍需保证采集稳在 30Hz。

---

## 2. 🔴 头号注意事项：裁完必须 PTS 归零

### 2.1 机制（为什么会错位）
- 裁掉前 `cut` 帧后,**保留帧若沿用原始 PTS**,视频首帧的 PTS ≈ `cut/fps`(例:cut=34 帧 → 1.13s);而 parquet 的 `timestamp` 列是从 0 起。
- lerobot 训练/eval **按时间戳解码视频**(`delta_timestamps`),且 kai/vis 路径 `tolerance_s=30.0`(为容忍抖动时间戳放得很松,`data_loader.py:191-195`)。
- 于是请求 parquet 第 `t` 行(`query_ts = t/fps`)时,解码器在"偏移了 cut/fps 的视频时间轴"上找最近帧 → **返回第 `t − cut` 帧**,30s 超大 tolerance 把偏移整个吞掉 → **不报错、静默取错帧**。
- 结果:训练时 `action[t]` 被配上**早 cut 帧的图像** → 策略学成"画面比动作滞后" → 真机视觉实时对齐时,伸向目标**过去的位置** → 抓不到。

### 2.2 为什么 offline MAE 测不出（盲区）
- train 和 val **用同一套错位解码** → 模型自洽拟合"滞后映射" → **val MAE 照样低**(甚至看不出异常)。只有真机(视觉实时对齐)才暴露。
- ⚠️ **VLA 数据裁剪后,offline MAE 不能作为"对齐正确"的判据** —— 必须显式验 PTS(见 §4)。

### 2.3 正确做法（裁剪即归零）
`trim_video_pyav` 已内置修复:写出每帧时 `new.pts = None` → encoder 自动按 0 起顺序分配 PTS。
**任何新写的裁帧脚本都必须这么做**(或裁完跑 `reset_video_pts.py` 补救)。

---

## 3. build 数据集的其它注意事项

- **视频用 copy,不要 symlink**:TOS 重构/重处理会**原地覆盖** v3 源视频(裁尾/重命名),旧 symlink 会指向"已变的源" → 与 parquet 错位(2026-06 实测 4/6 ep mismatch)。`build_v3early_dagger.py` / `build_task_ah1_split.py` 已改 copy 让数据集自包含。
- **逐 episode 校验 `parquet_rows == video_frames`**(裁剪 + build 后都验)。
- **norm_stats 重算**:裁剪改变了帧分布,build 后必须重算 norm。
- **kai0-native vs lerobot schema**:v1 是 `episode_id`,lerobot 是 `episode_index`;loader 需双 schema 兼容 + stale-manifest skip。

---

## 4. 验证 checklist（裁剪/build 后必跑）

```python
# (1) 首帧 PTS 必须 = 0
import av; c=av.open(mp4); print(next(c.decode(video=0)).pts)   # 必须 0

# (2) 解码对齐自测：按"时间戳"取第 IDX 帧，应等于按"帧序号"取的第 IDX 帧
#     (PTS 未归零时会差 cut 帧、像素差 ≫ 0)
```
- 首帧 `pts != 0` → **有 bug,禁止用于训练**。
- 全盘扫描一行命令(扫 v3 + self_built 各数据集抽 1 视频报非零 PTS)见本次调查脚本。

### 4.1 现成验证工具

- **离线 / 任意已存 episode**:`train_scripts/kai/data/validate_episode_pts.py`(逐 ep 实现上面 (1)(2) + `timestamp==frame_index/fps`,`--deep` 再做"按时间戳解码 vs 按帧号解码"的像素对齐自测):
  ```bash
  kai0/.venv/bin/python train_scripts/kai/data/validate_episode_pts.py \
      <数据集 leaf 目录, 如 …/Task_A/base/v3/2026-06-22-v3> --deep
  # 全 OK → exit 0;任一 ep 首帧 pts≠0 / 帧数≠行数 / timestamp 偏移 → 打印定位行 + exit 1
  ```
- **在线录制即时自检**:采集时设 `KAI0_VALIDATE_TRIM=1`,`EpisodeWriter.finalize` 每条 ep 重解码 3 路 mp4 断言 `pts0==0 & 帧数==行数`,失败直接抛错(见 §1.1)。验证一两条后可关掉(重解码有开销)。

---

## 5. 存量修复：`reset_video_pts.py`

对**已建好**的视频(裁剪时没归零的)做无损 PTS 归零:

```bash
kai0/.venv/bin/python train_scripts/kai/data/reset_video_pts.py <某数据集>/videos --workers 8
```
- **packet 级 remux,NO re-encode**(快、无画质损失):demux → 每包 `pts/dts -= first` → mux 复制 codec。
- 跟随 symlink 到真实源、原子 temp+rename、自带"首帧 ≈0"校验(BAD_PTS 会报)。
- ⚠️ 源视频若 root-owned(Volc job 写的)不可改 → 在 **tim-owned 的源/中间视频**上 reset,数据集 copy 后即带正确 PTS。

---

## 6. 2026-06 事故复盘 & 受影响数据集（已全部修复）

- **起因**:2026-06-15 TOS 重构**原地重裁** v3 源视频,走了**旧裁剪路径(未归零 PTS)** → 在所有重裁的 v3 上**重新引入** bug(`trim_video_pyav` 早已修,但重构没用它)。
- **暴露**:`pi05_flatten_fold_v3early_dagger`(Exp-C)真机抓衣角成功率骤降,offline MAE 却"不变" → 排查到 PTS 错位(详见 §8.10)。
- **修复**(2026-06-22,`reset_video_pts.py`,首帧 PTS 全部 → 0,解码对齐恢复 pixel-diff=0):
  - `vis_base/v3`:**04-23~05-10(11)** + **05-18~05-28 嫌疑窗(8)** 共 19 个日期。
  - `vis_dagger/v3`:8 个 dagger 日期。
  - self_built:`A_v3early_dagger`(重建)、`A_0522_0526_no_release`(直接 reset)。
  - 已清:`Task_AH1`(06-15-v3 本就 PTS=0,06-13 修复后处理 → 天生干净,真机正常)、`vis_awbc_merged_*`(重构前从干净源 copy)。
- **验证修好**:Exp-C 重训 `v3early_dagger_ptsfix`(robot-task 50k)→ offline @50 0.0387→**0.0274**(长 horizon −29%)+ **真机抓取恢复**。
- ⚠️ **副产物洞察**:"5-16~5-27 真机嫌疑窗"的部分嫌疑,很可能就是这些日期的 PTS 错位本身(现已修)。

---

## 7. 一句话总结
**裁视频头 = 必归零 PTS**(`new.pts=None` 或 `reset_video_pts.py`)+ **build 用 copy** + **验 `pts==0` 和 `frame==parquet`**。漏了 → 真机静默失败、offline MAE 骗你。
