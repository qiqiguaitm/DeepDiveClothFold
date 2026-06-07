# v2/v3 数据量×时窗 扩展实验 (Data Window Scaling)

> 📦 **2026-06-07 已并入** [`idle_data_trimming_experiments.md`](idle_data_trimming_experiments.md):本系列的**真正主线是"idle(投放/静止)数据裁剪对训练的影响"**(v2=未裁 vs v3=前端投放已裁),已围绕该核心目的重组到新文档。**本文档留存窗口/数量细节 + Exp-D(去脏 5-19~5-27, task `t-20260607104053-jgbgw`)任务跟踪**;idle 主题的汇总与"中段裁剪"分步规划见新文档。
>
> **目的**: 用 3 个 pi05 cloth-fold 训练实验,考察**数据时窗/数量**对真机表现的影响 —— 单日 vs 多日窗口 vs 全量。与 [`data_root_cause_probe_experiments.md`](data_root_cause_probe_experiments.md) 互补(后者查"数据质量/裁剪",本系列查"数据数量/时窗")。
> **状态**: 📝 规划已定稿 (决策见 §7;待 build 数据集 + 注册 config + 提交)
> **建立**: 2026-06-03
>
> ⚠️ **方法学铁律**(沿用本项目一贯结论): **真机为终判, offline MAE 系统性反指**。下面每个实验出 ckpt 后**必须真机测**,MAE 仅用于确认收敛 + 选 best ckpt。

---

## 0. 研究问题

后期数据(5-18 起)offline SOTA 但真机犯病(走停/犹豫/松手)。本系列换一个轴:**喂多少天、喂哪个时窗,真机怎么变**?
- **Exp-A 单日 5-18**: 最小数据(201 ep),看单日能否 work / 过拟。
- **Exp-B 5-18~5-28 窗口**: 后期 8 天(955 ep),"近期多日"。
- **Exp-C 全 v3(排 5-16)**: 全量 1940 ep,"全历史"。
- **Exp-B→C 顺序**: 先窗口后全量,对比"加早期数据(4-23~5-10)是帮忙还是稀释"。
- **⭐ Exp-D (2026-06-07 新增)**: 排除嫌疑窗 **5-19~5-27**(= Exp-C 去掉这 6 天)。用户怀疑之前 v3 训练有问题源于**混入 5-19~5-27 脏数据** → 本实验隔离验证。详见 §8。

---

## 1. 数据清单(2026-06-03 实测,源在 uc `vis_base/`)

### Exp-A 源: `vis_base/v2/2026-05-18-v2`
- **201 ep, 25G**。⚠️ **v2 版本**(与 B/C 的 v3 是不同处理 pipeline,见 §5 注意)。

### Exp-B / Exp-C 源: `vis_base/v3/<date>`
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

- **Exp-B (5-18~5-28, 加粗 8 日)** = **955 ep** (注:5-23/24/25 无采集)。
- **Exp-C (全 v3 排 5-16)** = **1940 ep** (4-23~5-10 共 985 + 5-18~5-28 共 955)。

---

## 2. 三个实验规格

| | **Exp-A** | **Exp-B** | **Exp-C** |
|---|---|---|---|
| 数据 | v2/2026-05-18-v2 单日 | v3 5-18~5-28 窗口 | v3 全量(排 5-16) |
| ep 数 | 201 | 955 | 1940 |
| 建议 config 名 | `pi05_flatten_fold_v2_0518_201` | `pi05_flatten_fold_v3_0518_0528` | `pi05_flatten_fold_v3_all_no0516` |
| 集群 | **cnsh** (robot-task, A100) | **cnbj** (Robot-North-H20) | **cnbj** (排队) |
| 卡数 | 16 (2节点) | 16 (2节点) | 16 (2节点) |
| 顺序 | 独立 | 先跑 | **Exp-B 完成后顺序跑** |

### 统一训练配置(单变量:只改数据)
对齐 work 锚点 smooth_800 / A_0522_0526 系列:
| 项 | 值 |
|---|---|
| Model | pi05 (`Pi0Config(pi05=True)`) |
| 框架 | JAX/Flax NNX (`scripts/train.py`) |
| Init | `mixed_1_clean`(与既往 work 锚点一致) |
| Prompt | `"Flatten and fold the cloth."` |
| use_delta_joint_actions | False (absolute) |
| LR | Cosine, warmup=1k, peak=1.5e-5, decay→1.5e-6 |
| EMA | 0.9999 |
| batch_size / fsdp_devices | 128 / 16 |
| **Steps** | **全部 50k**(用户定) |
| norm_stats | **各自重算**(`compute_norm_states_fast.py`),不复用 |
| inline_eval_val_root | `vis_v2_merged_val`(与既往 cross-val 一致,便于横比) |

---

## 3. 每实验执行链(逐个)

1. **build 数据集**(合并选定日期 → lerobot v2.1 单集, episode_index 重排):
   - 参考 `train_scripts/kai/data/build_no_release.py`(`--mode raw` 不裁)/ `build_task_a_new_100.py`。
   - Exp-A: 仅 `v2/2026-05-18-v2`。Exp-B: 合并 v3 的 8 个日期。Exp-C: 合并 v3 除 5-16 外全部。
   - 产物落各集群 vePFS 的 `self_built/<config数据名>/`。
2. **compute_norm_states_fast.py --config-name <config>**(数据所在机)。
3. **注册 config** 到 `kai0/src/openpi/training/config.py` + `git commit && push`(gf3/cnbj、gf0/cnsh 由 cron/pull 同步)。
4. **init ckpt** `mixed_1_clean/params` 在目标集群 vePFS 就位(size 校验 22G)。
5. **提交 16 卡 YAML**(cnsh / cnbj 对应 queue + image + SubPath,详见 [`../../deployment/training_ops/submission/`](../../deployment/training_ops/submission/) + 共性坑 `training_pitfalls_common.md`)。
6. **验证**: log 出 `Generating train split` + `Step N: loss` + 熬过第一次 ckpt save。

> **Exp-C 顺序触发(手动)**: Exp-B 在 cnbj 完成(50k + final save)后,**手动**提交 Exp-C(同 cnbj 16 卡)。

---

## 4. 数据定位(已澄清:master 在 vePFS,uc 回退无影响)

✅ **`vis_base` 是 gf0/vePFS 的软链接,raw 实际落盘在 vePFS** —— 数据 master 不在 uc,uc 回退**不影响**本系列。uc 上看到的 323G 只是经软链接访问 vePFS 的视图。

| 项 | 状态 / 动作 |
|---|---|
| Exp-A 源 (cnsh) | ✅ `vis_base/v2/2026-05-18-v2` 在 gf0/cnsh vePFS,**直接 build** |
| Exp-B/C 源 (cnbj) | ⚠️ v3 需在 **cnbj vePFS**(vePFS-North-E)。cnsh 有 → 若 cnbj 没有,按既往做法 **TOS 跨区同步**(参考 A_0423_0527 迁 cnbj 流程);build 在 gf3 本地做 |
| build 位置 | 在数据落地的 vePFS 机器本地 build(cnsh→gf0、cnbj→gf3),避免跨集群读 |

---

## 5. 注意事项 / 待决

1. **v2 vs v3 版本混用(已定:Exp-A 保持 v2)**: Exp-A 用 v2/5-18、B/C 用 v3 → 跨版本。因此 Exp-A **不与 B/C 构成严格"单日 vs 窗口"同版本对比**;Exp-A 是 v2-单日的独立基线,窗口效应的干净对比在 **Exp-B vs Exp-C(同 v3)** 之间。
2. **早期数据增益方向**: Exp-C vs Exp-B 的差(加 4-23~5-10 的 985ep)是本系列核心对比 → 真机判定"早期数据帮忙/稀释"。
3. **5-16 排除**: 仅 16ep/3.3M,残缺,排除无争议。
4. **steps(已定:全部 50k)**: ⚠️ Exp-A 单日 201ep 在 50k 大概率过拟 → inline-eval 选 best ckpt 时取中段(参考 no_release ~20k 触底,见 [`../../history/experiments/data_root_cause_probe_results.md`](../../history/experiments/data_root_cause_probe_results.md))。

---

## 6. 关联 XVLA 8 卡 volc 任务
X-VLA 的 volc 8 卡训练规划写在 [`xvla_track_x_curriculum.md`](xvla_track_x_curriculum.md)(本 pi05 系列之外的独立轨)。

---

## 7. 决策记录(2026-06-03 已定)
- ✅ **数据**: vis_base 软链接→vePFS,master 在 vePFS,uc 回退无影响(§4)。
- ✅ **Exp-A 版本**: 保持 v2/5-18,不改 v3(§5.1)。
- ✅ **steps**: 全部 50k。
- ✅ **Exp-C 触发**: 手动(Exp-B 完成后)。
- ✅ **XVLA**: 细化提交,规划落 `xvla_track_x_curriculum.md`。
- ⏳ 仍需: cnbj vePFS 确认/同步 v3(§4);config 注册 + 数据 build。

---

## 8. ⭐ Exp-D — 排除嫌疑窗 5-19~5-27(2026-06-07,已提交)

> **假说(用户)**: 之前 v3 训练(Exp-C 全量 / Exp-B 窗口)真机有问题,怀疑是**混入了 2026-05-19-v3 ~ 2026-05-27-v3 的脏数据**。本实验把这 6 天剔除,其余同 Exp-C,真机对比验证。

### 数据集 `A_v3_excl_0519_0527` = ≤5-18(排5-16)+ 5-28 = **1335 ep**(13 天)
选入(实测 ep):4-23(21) 4-24(187) 4-25(96) 4-28(152) 4-29(100) 4-30(83) 5-06(100) 5-07(20) 5-08(101) 5-09(30) 5-10(95) **5-18(201)** **5-28(149)**。
排除:**5-19/5-20/5-21/5-22/5-26/5-27(嫌疑窗 605 ep)** + 5-16(残缺)。
> = Exp-C(1940)− 嫌疑窗(605)= 1335。即"全历史去掉嫌疑 6 天"。

### 配置(同 Exp-B/C 单变量,只改数据)
- config `pi05_flatten_fold_v3_excl_0519_0527`(config.py)· init `mixed_1_clean` · 50k · batch128 · 16卡 · norm 重算 · inline_eval `vis_v2_merged_val`。

### 提交(cnbj 16卡, 2026-06-07)
- YAML `train_scripts/kai/volc/v3_excl_0519_0527_cnbj_16gpu.yaml`(2-host × 8 H20)。
- **task_id `t-20260607104053-jgbgw`**(cn-beijing / Robot-North-H20)。
- ⚠️ **数据 build 折进 entrypoint**(node-0 in-pod 用 `build_no_release.py --merge-src v3 --merge-dates ...` 合并 13 天 + 重算 norm,sentinel barrier;build 失败则 preflight 安全退出,不浪费训练)。原因:cnbj `/vePFS-North-E/vis_robot` 为 root:root 700,本地 tim 无权预建,故在 pod 内(root)建。
- 监控:`volc_job_status.py` / cnbj `logs/v3_excl_0519_0527_*.log`(root)。
- **真机为终判**:出 ckpt 后真机对比 Exp-C(含嫌疑窗)→ 若 Exp-D 真机明显改善则假说成立。
