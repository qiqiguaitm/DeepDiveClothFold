# Idle(静止/投放)数据裁剪对训练的影响 — 汇总 + 分步规划

> **核心目的(本系列的真正主线,之前文档未点明)**: 验证**裁掉 episode 里的 idle(静止)帧能否让模型真机表现更好**。idle 帧分两类:① **前端**"投放等待"长静止段(机械臂不动、操作员往台上放衣服);② **中段**操作里的停顿/犹豫/反复。假设:idle 帧被 BC 忠实模仿 → 真机走停 / 犹豫 / cloth loop / 拉取松手。
> **分步走**: **Step 1 前端投放裁剪**(= 之前的 v2→v3 / no_release,已做)→ **Step 2 中段 idle 裁剪**(未来)→ Step 3 节奏归一(可选)。
> **状态**: Step 1 ✅ 真机初步成立(裁前端真机明显改善);Step 2 📋 规划。
> **建立**: 2026-06-07(**完整合并自** `v2v3_data_window_scaling_experiments.md`(2026-06-08 该文件已删,明细见 §5)+ [`data_root_cause_probe_experiments.md`](data_root_cause_probe_experiments.md) H1,围绕 idle 主题重组)。
> ⚠️ **方法学铁律**: **真机为终判,offline MAE 系统性反指** —— idle 多的慢/停顿轨迹逐帧 teacher-forcing MAE 反而低,真机却灾难。MAE 仅用于确认训练健康 + 选 ckpt。

---

## 0. 为什么怀疑 idle 数据(已坐实的签名)

后期数据 offline MAE 更低却真机 fail;早期 smooth_800 offline 略差却真机 work。两者最大的数据侧差异之一就是 **idle/静止帧占比 + episode 长度**:

| 段 | 真机 | ep 中位长 | 静止帧 %(\|Δ\|<2e-3) | 投放 onset 中位 |
|---|---|---:|---:|---:|
| smooth 4-25~5-09(work) | ✅ | 1091 | **32.7%** | 短 |
| 后期 5-18~5-27(fail) | ❌ | 1600+ | **37~40%** | ~127 帧 |

→ 后期 ep **长 50%、静止帧多 5~7pp、开头投放等待长**。假设这些 idle 被学进策略 → 真机犹豫/走停。**本系列就是逐步裁掉 idle 验证之。**

---

## 1. idle 数据 + 裁剪机制(代码)

- **检测 motion-onset**(`build_no_release.py`):12D `|Δaction|` 均值持续 > `thr=3e-3`(rad/帧)达 `win=10` 帧的首帧 = 真运动起点;`margin=15` 帧。
- **前端裁**:`cut = max(0, onset - margin)`,删 parquet 行 `[0:cut]` + 同步裁 3 路 mp4(`assert video_frames == parquet_rows`)。
  - `--mode no_release`:对指定 2 天做前裁(单实验对照)。
  - `--per-date`(v3):对 `vis_base/v2/<date>` 每个 ep 前裁 → 输出 `vis_base/v3/<date>-v3`。
- **版本含义**: **v2 = 未裁原始;v3 = 前端投放已裁**。(只裁前端,中段 idle 仍在 → Step 2。)

---

## 1.5 ⭐ 互联网深度调研结论:是否需要删除"所有"idle 数据(2026-06-07)

> deep-research(101 agents / 19 源 / 23 条 3-票核验 claim)。**Bottom line:删掉"所有"idle 帧 —— 既非必要、也不是领域默认、且有风险;证据支持"靶向中间路线"而非一刀切全删。**

1. **idling/卡顿的真因是 single-step / history-conditioned BC 的 copycat 过拟合**,不是 idle 帧本身有毒 —— 单步策略学会"重复上一动作"(训练 near-zero 误差却真机不动)。[Diffusion Policy `2303.04137`:BC-RNN/IBC "get stuck when idle actions not removed";Causal Confusion `1905.11979`;Copycat `2010.14876`;ACT RSS19]
2. **chunked 策略(ACT / Diffusion Policy / pi0 / pi0.5 / X-VLA)天生吸收 chunk 内 pause** → 对保留的 idle 鲁棒(DP:horizon>1 "compensate for idle portions",h=8 最优)。**我们 pi05/XVLA 都是 chunked → 删中段 pause 的边际收益很可能很小。** 但**超过 chunk 长度的"前端长 idle"chunking 盖不住 → 仍需裁**(正好解释前端裁有效)。
3. **主流数据集不做"全删低速帧"**:DROID 只删 operator-gated idle(投放/setup 等待),非速度阈值全删;openpi pi0-DROID 只过滤"整块都 idle 的 chunk"。"删 idle 是常见标准做法"被**证伪(0-3 vote)**。
4. **全删的风险**:任务需要的 pause(叠衣 settle / regrasp 稳定)被删 → 策略学不会"该等时等";train/deploy 分布漂移;丢失恢复力。[DP 明确"因任务需要故意不删 idle"]
5. **领域推荐的是"删的替代方案"**:action chunking(已有)+ **upweight changepoint 关键帧**(Keyframe-Focused IL `2106.06452`)+ **语义/熵感知选择性下采样**(ESPADA `2512.07371` ~2×、DemoSpeedup `2506.05064` ~3×,压缩 casual 段、保留高精度段;**ESPADA 含真机叠衣**)。
6. **关键空白 = 本实验的价值**:目前**无任何"在 chunked VLA 上直接 ablate 删中段 pause vs 靠 chunking 吸收"的工作** → 本系列 v3.1 实验正好填空白。研究**预测:全删 ≈ 或略差于 前端裁(v3)**。

> **据此定 Step 2(按最优解)**:**主线 = v3.2 选择性**(前端裁 + 中段长 pause 设上限/下采样、保留短功能性 settle);**不做 v3.1 全删**(非必要/有风险/chunked 边际小),仅留作可选极端对照。两个实验都用 v3.2,各自对 v3(前端裁)baseline。详见 §3。
>
> **Sources**: DP `2303.04137` · ACT `roboticsproceedings.org/rss19/p016` · PIP `2508.15669` · Causal Confusion `1905.11979` · Copycat `2010.14876` · Keyframe-Focused `2106.06452` · ESPADA `2512.07371` · DemoSpeedup `2506.05064` · DROID `droid_policy_learning` · openpi droid README。

---

## 2. Step 1 — 前端投放裁剪:已完成实验汇总

| 实验 | 数据 | 裁剪 | offline | 真机 | 结论 |
|---|---|---|---|---|---|
| **no_release probe**(data_root_cause Exp-1)| `A_0522_0526`(后期 fail 2 天 200ep)| no_release(前裁)vs raw(未裁)| best step20k MAE@1 **0.0160**(与 raw 持平,offline 看不出)| ✅ **no-release 明显改善**(用户 2026-06-02)| 🟢 **H1 投放污染初步成立**:前裁真机更流畅 |
| v2/v3 window 系列 | v2/5-18(Exp-A 未裁)· v3 窗口/全量/去脏(Exp-B/C/D 前裁)| 混合(见下注)| 各 horizon MAE | ⏳ 真机待做 | 量数据量/时窗/去脏,**顺带都在 v3 前裁基础上** |

> ⚠️ **诚实标注(之前的混淆)**: v2/v3 window 系列(Exp-A~D)其实**混了多个变量**(trim v2/v3 × 窗口 1日/多日/全量 × 去脏 Exp-D),**没把"idle 裁剪"作为单变量隔离**。真正干净的"裁 vs 不裁"单变量对照是 **no_release probe(Exp-1 no_release vs raw,同 2 天同量)** → 这才是 Step 1 的关键证据。Exp-A~D 的窗口/去脏结论与 idle 主题正交(Exp-D `t-20260607104053-jgbgw` 仍在跑,完整明细见 §5)。

**Step 1 结论**: **前端投放裁剪(no_release / v2→v3)真机明显改善**(H1 初步成立)。offline MAE 看不出(反指)。→ **idle 数据确实伤真机,裁了有用;下一步裁中段。**

---

## 3. ⭐ Step 2 — 两个实验(按调研最优解:v3.2 选择性 idle 处理)

> **决策(按最优解,2026-06-07)**: 调研(§1.5)证据明确 —— **不做"全删 idle(v3.1)"**(非必要/有风险/pi05 chunked 边际小),改做**最优解 v3.2 = 前端投放裁 + 中段长 pause 设上限/下采样、保留短功能性 settle**(ESPADA/DemoSpeedup 式选择性下采样)。两个实验都用 v3.2。
> ~~v3.1 全删~~ 仅留作**可选极端对照**(若 v3.2 证明"裁中段有用"再考虑跑极端验上限),不作主线。

### 3.1 v3.2 数据构建 — 选择性 idle 处理(待实现)
- **定义**: v3.2 = v3(前端投放已裁)+ **对中段静止段**:
  - **保留短 settle**(段长 ≤ `KEEP_LEN`,如 ≤0.5s/15 帧)—— 功能性稳定/regrasp,**不动**(避免删掉"该等时等"的能力)。
  - **长 pause 下采样/设上限**(段长 > `KEEP_LEN`):保留前后边界 + 中间每 `k` 帧抽 1(或截断到 `MAX_PAUSE`),压缩但不清零 → 既减 idle 又不破坏"存在 pause"这一语义。
  - (进阶,可选)**熵/速度感知**(DemoSpeedup 式):低速 casual 段压得多、抓取/折叠高精度过渡段全率保留。
- ⚠️ **轨迹连续性**(调研强调):任何删/抽帧后必须 **重排 `frame_index`/`index` + 重建 chunk + 保留段边界过渡帧**,否则 action chunk 跨缝跳变。
- 实现:扩 `build_no_release.py` 加 `--mode idle_downsample`(参数 `KEEP_LEN/MAX_PAUSE/k`),输出 `vis_base/v3.2/<date>`。**先 2a 小规模可视化验证再批量。**

### 3.2 两个实验(主线;同 init/参数/pipeline/部署,单变量=数据)
| 实验 | 数据 | idle 处理 | 对照 baseline(已有 v3 ckpt 复用) |
|---|---|---|---|
| **Exp-1** ⭐ | **≤2026-05-10(早期 work 段)** | **v3.2 选择性** | smooth800 / ≤5-10 的 **v3(仅前端裁)** |
| **Exp-2** ⭐ | **全量 v3(排 5-16)** | **v3.2 选择性** | Exp-C/Exp-D 的 **v3(仅前端裁)** |

- **统一配置**(对齐 work 锚 smooth_800):pi05 / init `mixed_1_clean` / 50k / bs128 / 16卡 / norm 各自重算 / `vis_v2_merged_val` inline-eval。
- Exp-1 选 ≤5-10:这是真机已 work 的"干净早期段"(= smooth_800 来源),在它上加 v3.2 看能否在已 work 基础上更流畅/更快;Exp-2 全量看 v3.2 在大数据上的效果。

### 3.3 判据(真机为终判)
| Exp-N(v3.2)真机 vs 其 v3 baseline | 结论 |
|---|---|
| **更流畅/成功率↑/执行更快** | ✅ v3.2 选择性 idle 处理是增量 → 采纳为新默认(v3.2) |
| **≈ baseline** | 前端裁已吃掉主要收益,中段处理无额外增量(与"chunked 吸收 pause"一致)→ 维持 v3 |
| **更差** | 中段处理伤了(删了功能性 settle / 轨迹断裂)→ 回 v3,调 KEEP_LEN 更保守 |
> offline MAE 仅看训练健康(idle 轨迹 MAE 反指,§1.5/铁律)。

### 3.4 分步执行
- **2a** ✅ `build_no_release.py` 加 `--per-date-v32`(idle_downsample:保留运动帧+短settle≤15帧,长pause保边界+每k帧抽1;重排 frame_index/index + 帧选择性视频重编码,assert 帧数对齐)。单 ep 验证:854→804,seam max|Δa|≈原始(chunk 不跳变),视频==parquet ✅。commit `e8afea5`。
- **2b** ✅ build 19 个 v3 日期 → v3.2(压缩率 4–19%,晚期>早期,符合假设);merge `A_v32_le0510`(≤5-10,985ep)+ `A_v32_all`(全量排5-16,1940ep)+ norm 各自重算(gf3 本地建,免传输)。
- **2c** ✅ config `pi05_flatten_fold_v32_{le0510,all}`(init mixed_1_clean,50k/bs128/fsdp8)+ **cnbj 8卡**(非16:H20 配额;global batch 128 不变)。commit `301d1ea`。Exp-1 `t-20260607234501-5stck`、Exp-2 `t-20260607234509-qfq8w`。
- **2d** ⏳ 对齐各自 v3 baseline(复用现成 ckpt)+ **真机对比** → 落 §3.3 判据 + 回填(终判)。

### 3.5 offline 结果(inline-eval `vis_v2_merged_val`)
**Exp-1 `v32_le0510`(≤5-10 v3.2,985ep)— 训完 50k(2026-06-09):**

| step | MAE@1 | @10 | @25 | @50 |
|---|---|---|---|---|
| 8000 | 0.0088 | 0.0223 | 0.0465 | 0.0845 |
| 16000 | 0.0079 | 0.0191 | 0.0369 | 0.0640 |
| 24000 | 0.0074 | 0.0180 | 0.0344 | 0.0593 |
| 32000 | 0.0071 | 0.0176 | 0.0335 | 0.0577 |
| 40000 | 0.0069 | 0.0174 | 0.0331 | 0.0571 |
| **48000** | **0.0069** | **0.0173** | **0.0329** | **0.0569** |

- 单调收敛、末端平台(40k≈48k @1=0.0069)。最佳已保存 ckpt = `49999`(48000 最低但未存;keep_period=10000)。
- **最佳 ckpt**:`/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_v32_le0510/v32_le0510_cnbj/49999`
- **Exp-2 `v32_all`(全量1940ep v3.2)** cnbj 训练中,训完回填。
- ⚠️ **offline 仅确认收敛**:idle 裁剪是否有用是**真机判据**(v3.2 vs 各自 v3 前端裁 baseline);idle 多的慢轨迹逐帧 MAE 反指(§铁律),offline 看不出。

---

## 4. 关联文档
- **前端裁根因线**: `data_root_cause_probe_experiments.md`(H1=投放污染=本系列 Step 1;H2 慢节奏 / H3 gripper 漂移 / H4 wrist OOD 是**其它**真机失败根因,不属 idle 主题,各自单查)。
- **裁剪脚本**: `train_scripts/kai/data/build_no_release.py`(`--mode no_release` 前裁 / `--per-date` v3 / 待加中段裁模式)。
- **work 锚点**: `task_a_new_smooth_800_new_norm_results.md`。

---

## 5. 附录 — v2/v3 窗口/数量实验明细(2026-06-08 完整并入,原 `v2v3_data_window_scaling_experiments.md` 已删)

> 原 v2/v3 系列(2026-06-03 建)的初衷是查**数据时窗/数量**对真机的影响,**顺带都跑在 v3 前端裁基础上**(故与本 idle 主题相关但非单变量,见 §2 诚实标注)。下为完整明细 + Exp-D 任务跟踪。
> ⚠️ 真机为终判,offline MAE 反指;每个 ckpt 出来必须真机测,MAE 仅确认收敛 + 选 best。

### 5.1 数据清单(2026-06-03 实测,源在 uc `vis_base/`)
- **Exp-A 源** `vis_base/v2/2026-05-18-v2`:**201 ep, 25G**。⚠️ **v2 版本**(未裁,与 B/C 的 v3 是不同 pipeline)。
- **Exp-B/C 源** `vis_base/v3/<date>`(v3 = 前端投放已裁):

| date | ep | | date | ep |
|---|--:|---|---|--:|
| 2026-04-23-v3 | 21 | | 2026-05-10-v3 | 95 |
| 2026-04-24-v3 | 187 | | 2026-05-16-v3 | 16 ⛔排除(残缺3.3M) |
| 2026-04-25-v3 | 96 | | **2026-05-18-v3** | **201** |
| 2026-04-28-v3 | 152 | | **2026-05-19-v3** | **100** |
| 2026-04-29-v3 | 100 | | **2026-05-20-v3** | **100** |
| 2026-04-30-v3 | 83 | | **2026-05-21-v3** | **100** |
| 2026-05-06-v3 | 100 | | **2026-05-22-v3** | **100** |
| 2026-05-07-v3 | 20 | | **2026-05-26-v3** | **100** |
| 2026-05-08-v3 | 101 | | **2026-05-27-v3** | **105** |
| 2026-05-09-v3 | 30 | | **2026-05-28-v3** | **149** |

- **Exp-B(5-18~5-28,加粗 8 日)= 955 ep**(5-23/24/25 无采集)。
- **Exp-C(全 v3 排 5-16)= 1940 ep**(4-23~5-10 共 985 + 5-18~5-28 共 955)。

### 5.2 三个实验规格(单变量名义=数据量/时窗,均在 v3 前裁上)
| | **Exp-A** | **Exp-B** | **Exp-C** |
|---|---|---|---|
| 数据 | v2/2026-05-18 单日 | v3 5-18~5-28 窗口 | v3 全量(排 5-16) |
| ep 数 | 201 | 955 | 1940 |
| config | `pi05_flatten_fold_v2_0518_201` | `pi05_flatten_fold_v3_0518_0528` | `pi05_flatten_fold_v3_all_no0516` |
| 集群 | cnsh(robot-task A100) | cnbj(Robot-North-H20) | cnbj |
| 卡数 | 16(2节点) | 16 | 16 |
| 顺序 | 独立 | 先跑 | **Exp-B 完成后顺序跑** |

**统一训练配置(对齐 work 锚 smooth_800)**: pi05 / JAX `scripts/train.py` / init `mixed_1_clean` / prompt `"Flatten and fold the cloth."` / absolute joint / LR cosine warmup1k peak1.5e-5→1.5e-6 / EMA 0.9999 / bs128 / fsdp16 / **50k step** / norm 各自重算 / inline-eval `vis_v2_merged_val`。

### 5.3 执行链(每实验)
1. **build**(合并日期 → lerobot v2.1 单集,episode_index 重排):`build_no_release.py --mode raw`(不再裁,v3 已前裁)。产物落各集群 vePFS `self_built/<数据名>/`。
2. `compute_norm_states_fast.py --config-name <config>`(数据所在机)。
3. 注册 config 到 `config.py` + commit/push(gf3/cnbj、gf0/cnsh 由 cron/pull 同步)。
4. init `mixed_1_clean/params` 在目标集群 vePFS 就位(22G 校验)。
5. 提交 16卡 YAML(cnsh/cnbj 对应 queue+image+SubPath)。
6. 验证 log:`Generating train split` + `Step N: loss` + 熬过首次 ckpt save。

### 5.4 注意事项
- **v2 vs v3 混用**(已定 Exp-A 保 v2):Exp-A 是 v2-单日独立基线,**不与 B/C 构成严格单日 vs 窗口同版本对比**;干净窗口对比在 **Exp-B vs Exp-C(同 v3)**。
- **早期数据增益方向**:Exp-C vs Exp-B 的差(加 4-23~5-10 的 985ep)→ 真机判"早期数据帮忙/稀释"。
- **5-16 排除**:仅 16ep/3.3M 残缺,无争议。
- **steps 全 50k**:Exp-A 单日 201ep 在 50k 大概率过拟 → inline-eval 选 best ckpt 取中段(参考 no_release ~20k 触底)。

### 5.5 ⭐ Exp-D — 排除嫌疑窗 5-19~5-27(2026-06-07,已提交)
> **假说(用户)**: 之前 v3 训练(Exp-C/B)真机有问题,怀疑混入 **5-19~5-27 脏数据**。本实验剔除这 6 天,其余同 Exp-C,真机对比验证。

- **数据集 `A_v3_excl_0519_0527`** = ≤5-18(排5-16)+ 5-28 = **1335 ep**(13 天)。选入:4-23(21) 4-24(187) 4-25(96) 4-28(152) 4-29(100) 4-30(83) 5-06(100) 5-07(20) 5-08(101) 5-09(30) 5-10(95) 5-18(201) 5-28(149)。排除嫌疑窗 5-19/20/21/22/26/27(605 ep)+ 5-16。= Exp-C(1940)− 605 = 1335。
- **config** `pi05_flatten_fold_v3_excl_0519_0527` · init `mixed_1_clean` · 50k · bs128 · 16卡 · norm 重算 · inline-eval `vis_v2_merged_val`。
- **提交(cnbj 16卡, 2026-06-07)**: YAML `train_scripts/kai/volc/v3_excl_0519_0527_cnbj_16gpu.yaml`(2-host×8 H20)。**task `t-20260607104053-jgbgw`**(cn-beijing / Robot-North-H20)。
- ⚠️ **数据 build 折进 entrypoint**(node-0 in-pod `build_no_release.py --merge-src v3 --merge-dates ...` 合并 13 天 + 重算 norm,sentinel barrier;build 失败则 preflight 安全退出)。原因:cnbj `/vePFS-North-E/vis_robot` 为 root:root 700,本地 tim 无权预建。
- 监控:`volc_job_status.py` / cnbj `logs/v3_excl_0519_0527_*.log`(root)。**真机为终判**:出 ckpt 后真机对比 Exp-C(含嫌疑窗)→ Exp-D 明显改善则假说成立。

> **决策记录(2026-06-03)**: 数据 master 在 vePFS(vis_base 软链,uc 回退无影响);Exp-A 保 v2/5-18;steps 全 50k;Exp-C 触发=手动(Exp-B 完成后)。
