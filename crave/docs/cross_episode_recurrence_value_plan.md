# 跨 Episode 重复度挖掘 → 自动 Milestone / Value(探索记录索引)

> 📌 **本文档 = 探索记录的索引存根**(18 次迭代、所有否决的死路、56 图、文献调研)。详细叙述已收口进干净文档,这里只保留**结论速览 + 迭代索引(保 §-锚点)+ 文献 + 工件清单**,供溯源。
> **日常实现/方法/效果**:
> - 离散主线方法 → [METHOD](cross_episode_recurrence_value_METHOD.md)(V2.4 九步配方 + 四场景 + 否决死路 + 结论)
> - 连续 value 形态 → [CONTINUOUS](cross_episode_recurrence_value_CONTINUOUS.md)(端到端 TCC + DP 时序证据,原 §4.6)
> - 跨数据集泛化 → [GENERALIZATION](cross_episode_recurrence_value_GENERALIZATION.md)
> - 定位/场景/roadmap → [positioning_and_roadmap](CRAVE_positioning_and_roadmap.md)
> - 下游 A/B 落地 → [AB_plan](awbc_milestone_value_AB_plan.md)

> **核心假说(用户,2026-06-11)**:同任务多条 episode 中**反复出现的图像/状态 = 任务必经过程(milestone)**;低重复度图像 = 非必要操作甚至 error/negative。据此从跨 episode 结构挖 value,替代/增强 AWBC 的逐帧进度回归。
> **结果**:假说前半成立并落地(→ V2.4);后半实证否定(低重复 = 稀有 item,非错误,图6/16)。
> **上游**:AWBC pipeline([awbc_implementation_plan.md](../../docs/deployment/strategy/awbc_implementation_plan.md))。图像目录 `docs/visualization/`(图 1-55,GitHub 直接渲染)。

---

## 0. 结论速览(表1 — 核心结论与证据索引)

**图21 一图总览**:value 质量递进(同 kai0_advantage 50 ep / 同协议)`summary_tau_progression.png`。
**示例视频(git 内)**:[milestone_ep_s800_660_final_v4gated_sync.mp4](visualization/milestone_ep_s800_660_final_v4gated_sync.mp4)(终版配方 + 门控,held-out ep660)。

| # | 结论 | 关键数据 | 证据 |
|---|---|---|---|
| 1 | ✅ 假说前半成立:重复状态 = 必经 milestone | 覆盖率峰 82-92%,跨 3055 ep 稳定 | 图1/2/10/11 |
| 2 | ✅ 自动 milestone 可替代 DSM 手标 | 同 30 ep:median \|Δt\|=3.7% 时长,80%≤0.10 | 图12,§2.3 |
| 3 | ✅ 零训练 V_milestone **反超**监督 value | τ 0.812→0.865(臂掩膜)→0.875(⊕proprio)→**0.922**(500ep/k96/M20,held-out) vs 监督 0.896 | 图21,§2.10/2.11 |
| 4 | ✅ value 状态触发且泛化零衰减 | held-out 50ep τ=0.868;真机 3 轮 rollout 旁证 | §2.1(d),图7/8/22 |
| 5 | ❌ 假说后半("稀有=negative")否定 | 低覆盖段 = 稀有衣物类型,三数据集一致 | 图6/16 |
| 6 | ❌→✅ TCC 失败→修复→端到端追平主线 | v2(−0.31)→v3 frozen(0.75)→端到端 0.842(Pearson 反超),MAE 0.137→0.107≈主线 0.105 | §2.4.1/2.4.4 |
| 7 | ⭐ 顶簇偏置=机械臂占画面 → 臂掩膜修复 τ+0.05 | 92% 臂伪簇 → 82-90% 真布料簇 | §2.6,图17-19 |
| 8 | ⭐ coverage 低估根因 = 外观分裂;跨衣物泛化是梯度的 | 同阶段兄弟簇 sim 0.95;{c13,c18}合并 80→94% | §2.8,图28 |
| 9 | ✅ proprio 入聚类 = 最高性价比升级 | coverage 54-90%→82-100%;腕相机判不入 | §2.10,图30 |
| 10 | ✅ 挖掘规模有效但须 N/k/M 同扩 | N×10→k×2、M=k/5 | §2.11 |
| 11 | ⭐ 前段误对应根因=首尾混淆+低置信单帧 → 置信门控修复 | M16 首入 0.15→0.91;门控=驻留≥2帧∨margin≤0.8 | 图33 |
| 12 | ❌ K==M(全簇皆 milestone)否定;**τ 已饱和** | 线性时间 τ=1.000;K=M 高 τ = 计时器假象 | §2.12 |
| 13 | ⚠️ 随机泛化:对应稳,命中率受稀有外观限 → item 分组必做 | 稀有 item V 封顶 0.45-0.55 | 图34 |
| 14 | ⭐ V2 规划+落地:milestone 进度校准 + 相对(循环)milestone | P_k 校准 MAE 0.199→0.128(−36%) | §4.4,图35/37 |

---

## 1. 迭代索引(保留 §-锚点,供 METHOD/CONTINUOUS 交叉引用溯源)

> 每行 = 一次迭代的一句话结论。详细诊断/图在原始 git 历史的本文件长版(2026-06-16 前);最终配方见 [METHOD](cross_episode_recurrence_value_METHOD.md)。

**§1 文献调研与失败模式预注册** — deep-research(104 agents/22 源/25 claims 三票核验,21 confirmed/4 killed):假说前半 ✅25 年先例(McGovern&Barto 2001 bottleneck),后半 ⚠️文献明示脆弱。四教训:first-visit / 双阈值 recurrence / milestone=局部峰 / 离散图需 embedding 层。

**§2 实验记录**
- §2.1 V0 探针:(a)覆盖率结构真实存在;(b)GT 验证零训练 value vs `stage_progress_gt`;(c)低覆盖段=稀有 item 非 error;(d)鉴别实验 value 状态触发非时间驱动;(e)vis_dagger 探针。
- §2.2 全量挖掘 — milestone 跨规模稳定(集群 8×A100 提特征 + 本地挖掘)。
- §2.3 自动 milestone vs ViVa-DSM 手标 — 同 episode 直接对比,可替代。
- §2.4 TCC 复现(XIRL 官方代码):冻结特征版裁决不如聚类;**§2.4.1** v3 两根因修复跑通(τ −0.31→0.75);**§2.4.2** 数据处理管线 + 消融;**§2.4.3** 互补应用(App① 锚位消歧/App② 失败定位/App④ OOD 门控);**§2.4.4** 端到端微调捅破 frozen 上限,τ 0.718→0.842 追平主线。
- §2.5 逐 episode 视频与 value 鲁棒性配方(4 版迭代)。
- §2.6 覆盖率偏置 → 臂掩膜修复:τ +0.05。
- §2.7 聚类-审计可视化:覆盖率计算透明化。
- §2.8 coverage 系统性低估 = 外观分裂。
- §2.9 k 自适应停止准则(coverage==100% 失守即停)实测评估。
- §2.10 聚类纳入 action/proprio + 腕部视角实测(proprio 入、腕相机不入)。
- §2.11 挖掘规模扩展:500ep 有效,但 k/M 必须同步扩。
- §2.12 K==M(全簇皆 milestone)否定;τ 指标已饱和警示。

**§3 阶段结论** → V1 配方收口(armmask⊕proprio + N500/k96/M20 + 门控 + 时间分桶)。

**§4 方案演进**
- §4.1-4.3 V1 终态 pipeline + 下一决策点(AWBC 对照训练)。
- §4.4 V2 规划(milestone 进度校准 + 相对/循环 milestone):**4.4.1** 调研;**4.4.2** 预实验;**4.4.3** 方案设计;**4.4.4** E1-E3 快验;**4.4.5** V2 标签 + 可视化;**4.4.6** 循环 milestone 表示(多模式 V + 前向对齐 + 退步回落);**4.4.7** 两线合流 + advantage 层结构发现;**4.4.8→4.4.9** rollout 失败真因 = 读出逻辑非 domain gap(纠错);**4.4.10** F2 退步回落两发两中;**4.4.11** 多模式别名簇 + task-agnostic 连续性 DP;**4.4.12** action/proprio 消融(正向);**4.4.13** 端点锚 >> min-max 归一化;**4.4.14** 鲁棒性 V2.3 三路集成 + 硬边界;**4.4.15** 连续化评估(双锚插值 vs DP vs AE);**4.4.16** 四方法对画面核对;**4.4.17** 真根因=left-truncation 偏差 + 进度均匀选法;**4.4.18** V2.4 完整验证(前段误判通病解决);**4.4.19** 段内 value 细化文献+实测否决(τ 0.841→0.805 无增益)。
- §4.5 V2 完整实现配方表(收口,= [METHOD](cross_episode_recurrence_value_METHOD.md) §2)。
- §4.6 段间连续化(端到端 TCC 连续插值)→ 已抽出独立文档 [CONTINUOUS](cross_episode_recurrence_value_CONTINUOUS.md)(图47-55,DP 时序证据读出 + 跨数据集泛化 + 推导)。

**否决的死路(实证排除,勿重试)**:完整表见 [METHOD §4](cross_episode_recurrence_value_METHOD.md#4-否决的死路实证排除勿重试详见探索文档)。要点:段内 value 细化(§4.4.15/19)、因果硬约束(§4.4.16)、task-specific 规则(§4.4.14)、min-max 归一化(§4.4.13)、K==M(§2.12)、top-K coverage 选 milestone(§4.4.16/17)、coverage 减分母(§4.4.17)、纯几何距离插值(CONTINUOUS §1)。

---

## 2. 基础设施与执行记录

**集群任务**(均 cnsh;pod venv = `xvla/X-VLA-env/.venv`):8×A100 全量特征提取 14 分钟(`t-20260611230152-x5k2d`,坑:缺失视频弄死 shard,已加 skip);臂掩膜全量重提 ~50 分钟(`t-20260612100427-kzz5l`,806+3055)。
**经验定论**:pod venv 一律用 `xvla/X-VLA-env/.venv`(vePFS 自包含,已补 matplotlib/sklearn);开发机队列禁 Flexible → Preset `ml.pni2.7xlarge`;DINOv2 权重缓存 `HF_HUB_CACHE=/vePFS/tim/workspce/hf_cache/hub_default`;⚠️ gf0 本地 GPU 驱动 2026-06-11 消失 → 本地仅 CPU。

---

## 3. 参考文献(经 3 票核验)

### 第一梯队(必读)
| # | 文献 | 链接 | 重点 |
|---|---|---|---|
| 1 | Dwibedi et al., **TCC**, CVPR 2019 | [1904.07846](https://arxiv.org/abs/1904.07846) | cycle-consistency=逐帧共性分数;Fig.7 异常检测("稀有=negative"唯一定性先例) |
| 2 | Zakka et al., **XIRL**, CoRL 2021 | [2106.03911](https://arxiv.org/abs/2106.03911) | TCC 跨 episode 对齐 → value=到 goal 负距离;[代码](https://github.com/google-research/google-research/tree/master/xirl) |
| 3 | McGovern & Barto, ICML 2001 | [PDF](https://mcgovern-fagg.org/amy_html/old/pubs/mcgovern_barto_isairs2001.pdf) | 假说原始形式化;first-visit;§6 软负警告 |

### 第二梯队(实现前)
| # | 文献 | 链接 | 重点 |
|---|---|---|---|
| 4 | **GraphIRL**, CoRL 2022 | [2207.14299](https://arxiv.org/abs/2207.14299) | 先抽象外观再对齐(臂掩膜理论依据) |
| 5 | Şimşek et al., **L-Cut**, 2004 | [PDF](http://all.cs.umass.edu/pubs/2004/simsek_wb_TECH04.pdf) | Binomial 双阈值 recurrence 判定 |
| 6 | Şimşek & Barto, NeurIPS 2008 | [PDF](https://proceedings.neurips.cc/paper/2008/file/934815ad542a4a7c5e8a2dfa04fea9f5-Paper.pdf) | milestone=局部极大(betweenness) |

### 第三梯队(背景)
HRL Survey 2025 [2506.14045](https://arxiv.org/abs/2506.14045) · VIP/LIV [2210.00030](https://arxiv.org/abs/2210.00030)/[2306.00958](https://arxiv.org/abs/2306.00958) · AWE/Keyframe-IL [2307.14326](https://arxiv.org/abs/2307.14326)/[2106.06452](https://arxiv.org/abs/2106.06452) · LAV/GTCC [2103.17260](https://arxiv.org/abs/2103.17260)。

### V2 调研补充(支撑 §4.4 校准/循环 milestone)
**进度估计**:SARM [2509.25358](https://arxiv.org/abs/2509.25358)(T恤折叠,stage 进度=跨 demo 平均时间占比)· GVL [2411.04549](https://arxiv.org/abs/2411.04549)(逐帧完成度,shuffle 防计时器)· de Boer ICCVW2023 [2308.05533](https://arxiv.org/abs/2308.05533)(t/T 回归退化数帧)· TimeRewarder [2509.26627](https://arxiv.org/abs/2509.26627)(pairwise 进度差分>绝对 t/T)· ROVER [2508.01943](https://arxiv.org/abs/2508.01943)(步骤占比式进度)。
**重复动作/循环 milestone**:RepNet [2006.15418](https://arxiv.org/abs/2006.15418)(时间自相似矩阵周期)· GTCC CVPR2024(多模态 cycle-back + 可学 drop)· Drop-DTW [2108.11996](https://arxiv.org/abs/2108.11996)(drop 率=循环判据)· 外科 phase/gesture 两级 [Springer](https://link.springer.com/article/10.1007/s11548-024-03101-6)。
**规范时间轴/子目标值标定**:CTW/GCTW · soft-DTW barycenter [1703.01541](https://arxiv.org/abs/1703.01541) · StepFormer [2304.13265](https://arxiv.org/abs/2304.13265) · Okudo & Yamada 子目标 shaping [2104.06411](https://arxiv.org/abs/2104.06411)。

**开放问题(可发表贡献点)**:① 首达时间鲁棒分位数定位挖掘子目标;② 重访状态→相对进度信号(瓶颈挖掘一脉只做 first-visit 过滤);③ "recurrence→自动 milestone→AWBC 标签"完整链无人发表。

---

## 附录 — 工件清单

**图像**(图1-55):`docs/visualization/`(40+ 张,命名 `<阶段>_<数据集>_<内容>`)。
**示例视频(入 git)**:`milestone_ep_s800_660_final_v4gated_sync.mp4`。其余视频不入 git,在 `temp/`(rollout 四 value 同步 / 稀有 item 退化 / milestone-coverage 对账 / 三方 value 同步 / 跨数据集叠加 / TCC 对齐演示 / F2 退步回落等)。
**脚本**(`train_scripts/kai/data/`):探针 `recurrence_v0_probe.py` · GT 验证 `recurrence_v0_gt_validation.py` · 全量挖掘 `recurrence_full_mining.py` · 臂掩膜三件套 `build_arm_prototypes.py`/`extract_masked_features.py`/`armmask_compare.py` · 视频五代 `make_milestone_ep_video{,_v2..v5}.py` · TCC `tcc_v3_armmask.py`/`cache_frames_224.py`/`tcc_e2e_finetune.py` · 连续 value `tcc_e2e_dp_readout_ep2047.py`/`generic_continuous_generalize.py`/`render_3way_ep2047.py` · 统一库 `crave_value.py`。
**特征/挖掘缓存**(`temp/`):`tcc_{smooth800,kai0,dagger_*}{,_armmask}/feat_cache` · `full_mining_*/mining.npz` · `armmask/arm_prototypes.npz`。
**外部代码**:`/vePFS/tim/workspace/recurrence_research/google-research/{xirl,tcc}`(XIRL `one_hot` device bug 已 patch)。
