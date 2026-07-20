# RTC/EMA 速度消融 — 采样 episode 登记表

> 配套实验设计: [`rtc_ema_speed_ablation.md`](rtc_ema_speed_ablation.md)
> 每采一段 A/B 数据就登记一行, 并在该 episode 的 `meta/episodes.jsonl` 里打 `experiment` 标记 (见文末检索法)。

## 登记表

| 组 | ckpt变体 | RTC | EMA(α) | 数据集路径 | ep | 帧数/时长 | ckpt | 备注 |
|---|---|---|---|---|---|---|---|---|
| A3 全开 | v0 | on | on(0.5) | `Task_A/autonomy/v2/2026-07-19-v2` | **1** | 5073 / 187.8s | `pi05_v4_awbc_chunk001_dagger_crave_step49999` | RTC+EMA 基线 |
| A0 全关 | v0 | off | off(1.0) | `Task_A/autonomy/v2/2026-07-19-v2` | **3** | 3506 / 120.1s | `pi05_v4_awbc_chunk001_dagger_crave_step49999` | `enable_rtc:=false publish_smooth_alpha:=1.0` |
| A1 仅RTC | v0 | on | off(1.0) | `Task_A/autonomy/v2/2026-07-19-v2` | **4** | 2578 / 92.6s | `pi05_v4_awbc_chunk001_dagger_crave_step49999` | `publish_smooth_alpha:=1.0` (RTC 默认开) |
| A2 仅EMA | v0 | off | on(0.5) | `Task_A/autonomy/v2/2026-07-19-v2` | **5** | 4555 / 164.0s | `pi05_v4_awbc_chunk001_dagger_crave_step49999` | `enable_rtc:=false` (EMA 默认 0.5) |
| **B3 全开** | **v1** | on | on(0.7) | `Task_A/autonomy/v2/2026-07-19-v2` | **8** | 2475 / 82.8s | `pi05_v4_awbc_chunk001_dagger_crave_step49999` | V1 默认; ⚠prompt 标 mixed_1 (V1 WS cosmetic, 实际=crave v1) |
| B0 全关 | v1 | off | off(1.0) | `Task_A/autonomy/v2/2026-07-19-v2` | **9** | 1850 / 61.7s | `pi05_v4_awbc_chunk001_dagger_crave_step49999` | `--no-rtc publish_smooth_alpha:=1.0` |
| B1 仅RTC | v1 | on | off(1.0) | `Task_A/autonomy/v2/2026-07-19-v2` | **10** | 5445 / 181.8s | `pi05_v4_awbc_chunk001_dagger_crave_step49999` | `publish_smooth_alpha:=1.0` (RTC 默认开) |
| B2 仅EMA | v1 | off | on(0.7) | | | | | ⬜ `--no-rtc` |

## 检索法 (跨所有日期找已标记 episode)

```bash
# 列出所有打了 speed_ablation 标记的 episode (路径 / ep / 组 / RTC / EMA)
/data1/miniconda3/bin/python - <<'PY'
import json, glob
for mp in glob.glob("/data1/DATA_IMP/KAI0/Task_A/*/*/*/meta/episodes.jsonl"):
    for l in open(mp):
        d=json.loads(l); e=d.get("experiment")
        if e and e.get("study")=="rtc_ema_speed_ablation":
            print(f"{mp.rsplit('/meta',1)[0]}  ep={d['episode_id']:>3}  "
                  f"group={e['group']:10s} rtc={e['rtc']} ema={e['ema']} "
                  f"len={d['length']} ckpt={e.get('ckpt')}")
PY
```

标记写在 episode 的 `episodes.jsonl` 记录里: `scene_tags` 含 `speed_ablation`, 另有结构化 `experiment` 字段
(`study/group/ckpt_variant/rtc/ema/ema_alpha/ckpt/purpose`)。采新段后照 episode 1 的格式补标 + 在此登记。
