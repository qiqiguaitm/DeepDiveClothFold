# CRAVE 工作集 · STATUS / TODO

> 单页事实源:已收口 / 已否决 / 未做可做(带优先级 + 资源判据)。
> 维护规则:每完成一项就勾选并回填**结论 + 留痕路径**;新结论同步进 [METHOD](cross_episode_recurrence_value_METHOD.md) / [positioning](CRAVE_positioning_and_roadmap.md)。
> 最后更新:2026-07-01。

---

## ✅ 已收口(不再做)
- [x] 离散 V2.4 主线(9 步配方 + 四场景 + 跨数据集泛化 XVLA 0.956 / coffee 0.988)→ [METHOD](cross_episode_recurrence_value_METHOD.md)
- [x] 连续 TCC+DP value 形态(advantage 密集 81-96%,跨数据集 corr 0.94-1.00)→ [CONTINUOUS](archive/cross_episode_recurrence_value_CONTINUOUS.md)
- [x] value 计算重构为统一库 `train_scripts/kai/data/crave_value.py`
- [x] **B1 pilot(决定性地基)**:R²(action|milestone)=0.43 vs R²(action|时间)=0.22(2×)→ milestone=动作相关技能相位。脚本 `crave_milestone_action_pilot.py`
- [x] A 臂三档落地 + 本地 sanity(MAE@1 0.0086,介于 SFT 0.0089 与 C 0.0079)+ 正式收敛训练已提交集群(job `t-20260617102228-7zpvl`)
- [x] **Viterbi 替代方案探索 (2026-07-01)**: 在 ep2302 上系统比较了 raw argmin → distance correction → cumulative vote → transition probability → cooldown。核心发现:① 30Hz 原生处理远优于 3Hz(stride=10 引入 aliasing);② 距离校正(α=2.0)+ 非对称累积投票(fwd=1,bwd=50f≈1.7s)可达 mono=0.997,end=0.95,6 bounces——接近 Viterbi 的 22 jumps/0 bounces 但更简单且完全在线;③ coverage-gating/EMA/完全对称投票/cooldown 均被实验否定。结论:30Hz + DC + mild backward vote 是可部署的 Viterbi-free 方案。留痕:脚本 `crave/experiments/viterbi_vs_ema_ep2302.py`,`argmin_jump_analysis.py`,`em_hmm_vs_kmeans.py`;数据 `temp/coverage_gated_jump/`;消融图 `temp/coverage_gated_jump/ablation_30hz.png`。
- [x] **Viterbi-free 最终方案收口 (2026-07-01)**: DC(α=2.0) + sym adaptive vote(wd=10,t=0.3) + smooth_monotone(w=15,boxcar 0.5s)。30Hz 原生,仅 3 参数,完全在线。vs Viterbi target: corr=0.974,末值=0.981,0 尖峰。对标 Viterbi 的 5 个参数/离线/全局 DP。详细文档:[sym_adaptive_vote](sym_adaptive_vote.md)。消融:DC alone 无效(224 jumps),sym vote 削尖峰(5 spikes),boxcar smooth 生成斜坡(+corr)。数据:`temp/coverage_gated_jump/smoothed_final.png`,`ep2302_spike_free.mp4`。
- [x] **后续标准: 30Hz 原生处理** — 不再使用 3Hz stride=10 → 30Hz upsample 的间接路径。所有新 episode 的推理直接在 30Hz 特征上运行(KMeans 模型仍可用 3Hz 训练)。依据:30Hz raw argmin mono=0.96 vs 3Hz 0.78,高频采样本身就是最强的平滑器。

## ❌ 已否决(别再走)
- [x] **B2 弱成败信号**:AE-neg 残差 0.981 vs AE-pos 0.979 不可分 → CRAVE 无细微失败信号,只抓粗失败。脚本 `crave_b2_failure_signal.py`
- [x] 二值化 advantage(38.8% exact-zero 平台帧被强劈)→ 已改三档
- [x] **EM-HMM 统一概率框架**:768D 下塌缩(96→12 状态),所有簇质量指标差于 KMeans K=96。根因:isotropic Gaussian 在高维中的 $-d\log\sigma^2$ 项驱动方差竞争→大簇吃小簇。结论:KMeans 在本场景更优。留痕:[em_hmm_negative_result](em_hmm_negative_result.md) · 脚本 `crave/experiments/em_hmm_vs_kmeans.py` · 数据 `temp/em_hmm_vs_kmeans/`

---

## 🔴 P0 · 决定性(阻塞结论,需集群/sim,非本地)
- [ ] **Tier3 sim01 rollout(A/B/C 三臂)** — AB_plan 唯一决定性判据。等 A 臂集群训完。`本地无 sim,不可本地验`
- [ ] **A 臂集群收敛训练收尾** → 出 ckpt 后离线 MAE 对照 C

## 🟠 P1 · 快赢(低成本,本地可验)— **本轮已跑实**
- [x] **A1-切分** ✅ 20-milestone 骨架,19.3/20 到达,覆盖 0.967,~12 段/ep;raw τ=0.43 印证有序相位需 DP → [结果](visualization/crave_a1a2_results.md)
- [x] **A2-keyframe** ✅ milestone 跨越帧导出(~12/ep,`segments.json`)
- [x] **A2-OOD/残差** ⚠️ 跨任务 AUC xvla0.997/coffee1.000 ✅;域内细粒度 AUC0.545≈随机 ❌(印证 B2 粗失败边界)
- [x] **A2-dedup** ✅ 覆盖率 r=0.65 质量分;段时序指纹 std0.34/56 近重复(修复 reached-set 退化)
- [ ] **A1-VLM 段命名**(~12 次/任务,需 API)→ 接 AWBC `prompt_from_task`【唯一剩余本地外项】
- [ ] **B 臂(V2.4→AE 蒸馏)** 写伪 GT → Stage1 训 AE(50k,需 GPU,集群)→ 打 advantage → AWBC `本地仅 sanity`

## 🟡 P2 · 规模化(中成本,单机 GPU,搁置)
- [ ] **B1 全量** pilot→全量,出 action-aware 在线 advantage labeler
- [ ] **C1 蒸馏分布式离散 value 头**(RECAP 式 201-bin CE,非 scalar+MSE)`因 causal-DP 已 0.94 非必要`

## 🔵 P3 · 高天花板(高成本,真机,依赖 P0)
- [ ] **C2 CRAVE 冷启 V + 真机 RL 微调** — 唯一能碰"超越示教"的路径,补结果信号洞

## 🟢 持续性
- [ ] **D1 增量挖矿 + 漂移监控 + 域自适应** — 制度化消灭手动挖矿域错误

---

## 本轮自动化执行记录(2026-06-17~)
> 每完成一项回填:命令 / 输入数据 / 结论 / 留痕路径(脚本 + 输出图表)。

### A 组(立即可做)本地零成本验证 ✅ 已跑
- **脚本**:`train_scripts/kai/data/crave_a1a2_validate.py`(挖掘逐字复刻 `smooth800_v24_full.py`,与 mv_value 同模型 20 milestone)+ `crave_a2_dedup_fingerprint.py`
- **命令**:`kai0/.venv/bin/python train_scripts/kai/data/crave_a1a2_validate.py --mine-n 700 --max-ood 200`(全 CPU,~数分钟)
- **输入**:`A_smooth800_dagger_all` 全 1117ep 缓存特征 + `generalization_value_eval/{xvla,coffee}` + `mv_value_full/corr.json`
- **留痕**:`temp/crave_a1a2/{summary.json, segments.json, ood_residuals.npz, dedup_fingerprint.json}` + `docs/visualization/crave_a1_*.png, crave_a2_*.png` + 结果文档 `crave_a1a2_results.md`
- **结论**:
  | 项 | 结果 | 数字 |
  |---|---|---|
  | A1 切分 | ✅ | 19.3/20 骨架,覆盖 0.967,raw τ=0.43 |
  | A2 keyframe | ✅ | ~12 去重段/ep |
  | A2 OOD | ⚠️ 跨任务✅/域内❌ | AUC xvla 0.997 · coffee 1.000;域内 0.545≈随机 |
  | A2 dedup/质量 | ✅ | 覆盖 r=0.65;指纹 std0.34,56 近重复 |
- **独立复现**:A2 域内 OOD≈随机,与 [B2 否决](CRAVE_positioning_and_roadmap.md#b-组--核心研究补最大软肋无-action无结果信号)同向 → 再证 CRAVE「只抓粗失败/脱轨,抓不到 on-manifold 细微差异」。

### 下一步本地可做(尚未做)
- A1 的 VLM 段命名(需 API)→ 真正接上 AWBC `prompt_from_task`
- D 组段时序指纹的检索/增量挖矿扩展(纯 CPU)
