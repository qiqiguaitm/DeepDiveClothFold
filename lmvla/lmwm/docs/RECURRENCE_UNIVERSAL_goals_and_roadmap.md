# Recurrence-as-Universal-VLA-Signal · 需求对齐 + 路线图 + 探索日志

> 立档 2026-07-14。本文件 = **① 对齐后的需求(不可漂移的北极星)+ ② 验证路线图 + ③ 持续追加的探索/实验日志**。
> 每做一步实验,结果与结论 append 到 §4;方案变更必须回看 §1 是否仍对齐。
> 关联:CRAVE `../../crave/docs/HISTORY.md`(旧 milestone 方案淘汰史)· LMWM `HISTORY.md` · 失败案 `RESULTS_lmwm_vs_lawam_libero10_2026-07-15.md`。

---

## 1. 核心目的与需求(用户 2026-07-14 亲述,对齐版)

### 1.1 一句话目的
把「**同任务、跨 episode 反复出现的状态**」这一**原始信息**,做成一个 **普适 · 不强加离散 milestone · 贯穿整个 VLA 框架** 的信号并充分利用——**不是再造一个针对特定任务调参的 milestone 标签器**。

> **关键反转**:recurrence 是"金矿";milestone 只是过去挖它的一种、且已被证明**不普适**的方式(视觉聚类在 LIBERO 低视觉方差任务上塌成 M=1)。要的是把这块信息**普遍榨干**、让 VLA **训练/推理都受益**的方法。

### 1.2 硬约束(缺一不可)
| # | 约束 | 依据 |
|---|---|---|
| C1 | **普适 > 每任务调参**:一套固定超参跨 kai0 / LIBERO-40 / robotwin2.0,**不按任务或数据集开关/调 K** | 用户反复否决 coverage 开关、per-task K |
| C2 | **尺度无关自适应**:参数固定,**输出**(边界数/结构)随任务复杂度**自涌现**(简单少、多步多),非固定 K、非塌 1 | 用户"随任务复杂度自适应涌现" |
| C3 | **不绑死 milestone**:recurrence 当**连续场**;milestone/进度/子目标只是**可选派生读法**,先验证、好留坏弃 | 用户"不一定代指 milestone" + "可以验证一下不好就抛弃好就保留" |
| C4 | **框架级价值**:能进 VLA 的 **训练(无标注 advantage/加权)· 条件(子目标)· 部署(OOD/脱稿监控)**,不只是离线标签 | 用户"对 VLA 整个框架大有裨益" |
| C5 | **评估诚实**:proxy 只作诊断;**随机对照 · 多 seed · 最终下游 SR** 才是判据 | 用户全程把关 + LMWM PITFALLS C6 |

### 1.3 交付双目标(用户:两者都要)
- **研究贡献**:普适性是卖点 → 评估重心含 LIBERO-40 通用性 + 机制消融。
- **落地**:真正把 VLA 训得更好 → 评估重心含**下游 SR**(kai0 真机 / LIBERO eval / robotwin)。

### 1.4 数据集(用户:都要普适;先小验证定论,再大数据复核)
- **小验证场**:LIBERO-40(特征已抽 1693ep)、kai0(crave bank)、task6(聚类塌缩的诊断任务)。
- **大数据复核**:robotwin2.0(27500ep,**特征未抽** → 待抽)+ LIBERO 全量 + kai0 全量。

---

## 2. 当前工作假设(手段,可被证伪/替换)

**逐帧 cross-ep 复现密度场 `r(o)`** 作为普适原语:
```
r(o_t) = 1/(N_ep-1) · Σ_{j≠ep(t)} exp( -dmin(o_t, E_j)² / 2σ² )
  dmin(o_t,E_j) = o_t 到 episode j 最近帧距离(冻结 DINOv3-base pooled 特征, L2)
  σ = 所有跨-ep dmin 的中位(尺度无关带宽)
```
- **无聚类 · 无阈值 · 无 K · 无锚** → 天生普适(同一个 kNN + σ 中位跑所有数据集)。
- **三个正交读法**(对应 C4 三处用法):
  - **幅值 r**:在不在"任务共识流形"上 → 部署 OOD/脱稿监控(用途 C)+ 训练加权(用途 A)。
  - **低谷**(-r 显著极小):跨-ep 分歧点 = 普适子任务边界(用途 B 的 milestone 派生)。
  - **高脊**:canonical 关键态 = 子目标锚(用途 B)。

---

## 3. 验证路线图(对齐用户 4 点回答)

| 阶段 | 内容 | 判据 | 状态 |
|---|---|---|---|
| **V0** | r 场普适非退化(task0/6/kai0) | 聚类塌处 r 仍连续(std>0) | ✅ 见 §4.1 |
| **V1** | r-低谷分割器全 40+kai0,单一全局阈值 | 边界数随复杂度涌现 + 跨-ep 稳定(>随机) | 🔄 跑中 §4.2 |
| **V2** | milestone 派生读法 keep/drop | 好留坏弃(C3) | 待做 |
| **V3a** | 用途 A:r 加权 / 无标注 advantage(训练) | r vs 动作一致性 | ✅ premise 通过 §4.5(ρ+0.21,100%正;kai0/LIBERO强/robotwin弱) |
| **V3b** | 用途 B:高脊/低谷作子目标条件 | 逐个验证(C4) | 待做(下一个) |
| **V3c** | 用途 C:部署 r 监控 | 低 r 命中 OOD | ✅ 跨任务 AUROC 0.999 §4.3;同任务失败帧待失败rollout |
| **V4** | 普适复核:kai0 + robotwin2.0 特征 | 同一超参跨本体成立 | ✅ 特征抽取+同步 §4.4;robotwin V0/V1 复核待(72任务≥10ep) |
| **V5** | (a)结构接法下游重训 + SR | 研究+落地双目标 | ✅ 内在(前向gain 2.1× §4.7);**sim-SR §4.9: M''=93.8% > M=92.2%, 但 <B=96.4%** |
| **V6** | 置信门控:high-r→r-脊 / low-r→t+7,+ hint-dropout | 修 M'' 别名任务 −12(task8)不牺牲 task6 | 待做(下一个,根因见 §4.9);history gate ✅ 未撞旧方案 |

**评估纪律**:每个 proxy 配随机/基线对照;宣称"普适"= 同一超参在所有场成立;最终以下游 SR 收口。

---

## 4. 探索 / 实验日志(持续追加)

### 4.1 V0 · r 场普适非退化 ✅(2026-07-14)
脚本 `recurrence_field.py` · 图 `assets/recurrence_field.png`。
| 任务 | 聚类给的 | r std | r range | 结论 |
|---|---|---|---|---|
| LIBERO task0(清晰) | 可分 | 0.057 | [0.25,0.74] | 平缓梯度 |
| **LIBERO task6(弥散,BGMM 塌成 M=1)** | **M=1** | **0.100** | **[0.21,0.80]** | **聚类塌处 r 最丰富** |
| kai0(视觉多变) | 可分 | 0.097 | [0.17,0.82] | 随折叠升高 |

**关键发现**:
1. 同数据聚类塌成 1 团(std=0),而 r 场 std=0.10、range 0.2–0.8 连续;热图竖带 = **跨-ep 一致**(真信号非噪声)。
2. **task6 意外**(修正预判):低 r 谷不在"未训练尾"(t>0.4),而在 **t≈0.4–0.5 的子任务交界/分歧点**(放 mug→转拿 chocolate,各 demo 走法最不一致);尾段 r 回升(结局一致=canonical)。→ **r 低谷 = 天然子任务边界**。
3. `low(r<0.3)=0%`:三者全是成功 demo → 全在流形上;OOD 报警(用途 C)须用**失败 rollout** 才测得出,此处不夸大。

### 4.2 V1 · r-低谷分割器(全 40 LIBERO + kai0,单一全局阈值)✅(2026-07-14)
脚本 `rvalley_segmenter.py` · 图 `assets/rvalley_segmenter.png`。边界 = 跨-ep 中位 r(t) 在 −r 上的显著低谷(find_peaks,prominence=全局阈值)。

| 判据 | 结果 | 判定 |
|---|---|---|
| **边界数涌现**(非全 0/全 22) | thr0.03: 分布 **[17×0, 23×1, 1×2]** med=1;thr0.02: range[0,**4**] | ✅ 逐任务涌现:短单动作→0、pick-place→1、多物→2 |
| **随复杂度自适应** | corr(边界, #"and")=**0.22**、corr(边界, ep长)=**0.29**(弱) | ⚠️ 分布上成立,但**数值相关弱**——#"and" 是烂 proxy("pick up X and place"=1 转换却有 and);需更干净复杂度标签(夹爪开合次数/人工子任务数)才能严格证 |
| **跨-ep 稳定**(真>随机) | 真 recall 中位 **0.67** vs 随机 **0.29**,**真>随机=100% 任务** | ✅ **强**:每个任务的 r 谷都比随机边界更跨-ep 一致,散点全在对角线上方 |
| **单一全局阈值鲁棒** | thr 0.02/0.03/0.05 → recall 0.71/0.67/0.63,粒度单调可调 | ✅ 一个全局旋钮控粒度,recall 不崩 |
| **kai0 普适** | 同一全局阈值 boundaries=1,recall 0.80 vs 0.56 | ✅ 同超参跨本体成立 |

**V1 结论**:**r-低谷是一个普适(一套全局阈值跨 40 LIBERO + kai0)、跨-ep 强稳定(vs 随机 100% 胜)、边界数逐任务涌现的分割器**——满足 C1/C2。**诚实保留**:①"随复杂度自适应"仅分布层面成立,数值相关弱(proxy 差,待更好标签);②thr0.03 下 17/41 任务 0 边界(短单动作任务,本就=1 段,合理但意味着该分割器主要在多步任务上产边界)。→ **milestone 派生读法(用途 B)有普适可用的边界来源;是否真帮下游留给 V3b/V5 的 SR 裁。**

### 4.3 V3c-proxy · 用途C(部署 OOD/脱稿监控):r 的流形判别力 ✅(2026-07-14)
脚本 `recurrence_ood_monitor.py` · 图 `assets/recurrence_ood.png`。跨任务注入:参考任务 80%ep 建流形(σ内部标定),in-task=留出20%ep帧(应高r),off-task=别任务帧(应低r),测 AUROC。

| 指标 | 结果 |
|---|---|
| **AUROC(in vs off)** | 中位 **0.999**,最低 **0.972**,**>0.9 on 100% 任务**(12 个) |
| r 幅值分离 | in-task r≈**0.59–0.62** vs off-task r≈**0.01–0.11**(gap≈**+0.5**) |

**结论**:**"低 r = 脱离 demo 流形"作为部署监控成立,且跨 12 任务近乎完美、普适**(同一 kNN/σ 机制,零 per-task 参、零失败数据)。→ C4 的"部署"这条用途**premise 验证通过**。
**诚实保留**:off-task 帧来自**别的 LIBERO 任务**(场景/物体不同 = 视觉差异大)= **相对"易" 的跨任务 OOD**;真正难的是**同任务内的细微失败帧**(subtle OOD),那需要**失败 rollout**(部署侧数据,当前无)才能测——留 V3c-full。但机制与普适性已强证。

### 4.4 V4 · robotwin2.0 抽特征(跨本体普适复核准备)🔄(2026-07-14)
脚本 `robotwin_dinov3base_extract.py`(cam_high,frame_cache_jpeg256 → 池化 [N,768],**与 LIBERO 逐比特同 `encode_grid` 路径**,双卡 --shard 0/1)。
- **关键验证(冒烟)**:robotwin 池化 mean=0.002/std=0.302/norm=8.36 **≈ LIBERO** mean=0.004/std=0.300/norm=8.31 → **同编码空间、跨本体可比**(跨本体普适 claim 的前提成立)。
- **覆盖**:frame_cache_jpeg256 只缓存 **5000 ep(0–4999,~1.1M 帧)**,已覆盖 ~全部 task 类型(300ep→270 task_index);其余 22500 仅 av1 视频(需 pyav,慢)→ **先抽 5000(充分大数据),22500 av1 余量作可选后续**。
- 输出 `lmwm/data/robotwin_dinov3base/ep{e}.npz`(key=pooled)→ rsync 到 gsy North-E `/vePFS-North-E/vis_robot/workspace/deepdive_kai0/lmvla/lmwm/data/robotwin_dinov3base`。
- **✅ 完成(2026-07-14)**:5000 ep / **1.09M 帧** / 1.5GB / 双卡各 ~1900s(250 fr/s);格式 pooled[N,768] 尺度对齐 LIBERO(norm 8.36)。→ 同步 North-E ✅(tar-over-ssh,REMOTE_NPZ=5000)。
- **✅ 跨本体复核(`robotwin_revalidate.py` · 图 `assets/robotwin_revalidate.png`,72 任务≥10ep,同 LIBERO/kai0 超参)**:V0 r_std 中位 **0.061**、100%>0.02 非退化;V1 边界涌现 [30×0,4×1,38×2];**V1 跨-ep 稳定 真 recall 0.73 vs 随机 0.30、100% 真>随机**。→ **同一套超参跨 3 本体(kai0/LIBERO/robotwin-aloha双臂)全成立 = C1 普适确认。**

### 4.5 V3a-proxy · 用途A(训练:r 加权/无标注 advantage)机制验证 ✅(2026-07-14,无需训练)
脚本 `recurrence_action_consistency.py` · 图 `assets/recurrence_action_consistency.png`。测 `corr(r, −跨demo动作离散度)`:高 r 帧的近邻(同状态他demo)动作是否更一致。

| 任务 | ρ(r, 动作一致) | 动作离散 低r→高r |
|---|---|---|
| kai0 | **+0.544** | 3.16→1.90 |
| LIBERO-task0 | **+0.478** | 3.10→1.38 |
| LIBERO-task6 | +0.274 | 2.07→1.01 |
| robotwin t2851/2834/2838 | +0.085/+0.112/+0.147 | 弱降 |
| **中位 ρ +0.210 · >0 on 100% 任务** | | |

**结论**:**"高 r = 跨-demo 动作更一致 = BC target 更可靠"方向普适成立(100% 正)**→ "按 r 加权 BC / r 作无标注可靠度权重"这条训练用途 **premise 通过**。**诚实保留**:效应量**数据集依赖**——kai0/LIBERO 强(ρ 0.27–0.54,离散度降 ~2×),**robotwin 双臂弱**(ρ 0.09–0.15,14 维双臂动作本就更散,且每任务仅 ~20ep 复现弱)。→ r 是有效的无标注可靠度信号,但"加权是否真提升 SR"因本体而异,留 **V5 下游 A/B** 裁。

### 4.6 V3b-proxy · 用途B(子目标条件)机制验证 ✅(2026-07-14,无需训练)· ⭐关键修正
脚本 `recurrence_subgoal_consistency.py` · 图 `assets/recurrence_subgoal_consistency.png`。三种 world-model 目标的跨-demo 特征一致性(近邻子目标离散度,越低越可学):**脊(下一段 canonical 高-r 收敛点)/ 谷(下一 r-边界)/ 固定(t+τ)**。

| 任务 | 脊 ridge | 谷 valley | 固定 fixed | ridge vs fixed |
|---|---|---|---|---|
| kai0 | 0.067 | 0.150 | 0.096 | **+30% 胜** |
| LIBERO-task0 | 0.026 | 0.030 | 0.024 | −7% 负 |
| LIBERO-task6 | 0.036 | 0.049 | 0.033 | −10% 负 |
| robotwin t2851/2834/2838 | ~0.117 | ~0.17 | ~0.145 | **+17~24% 胜** |
| **脊胜固定 67% · 谷胜固定 0%** | | | | |

**结论(3 条,含对 −4.2pt 的机理解释)**:
1. **谷(边界)是普适最差目标(0% 胜)**——边界=跨-demo 分歧点。**坐实 −4.2pt 机理**:旧 milestone+1 把目标定在转变/边界点 → 本就比 t+7 固定目标难学 → world-model 学不好 → SR 掉。
2. **脊(canonical 高-r 收敛点)远好于谷**,且在**长程任务胜固定**(kai0 +30%、robotwin +17~24%),**短 LIBERO 略负**(−7/−10%)。
3. → **Use B 保留(好留坏弃),但只用脊目标、绝不用边界**;收益集中长程。
**⭐ 对 V5(a) 的强制修正**:world-model 目标 = **下一 r-脊(canonical 态)**,NOT r-谷边界;末段仍锚末帧。用边界=重蹈 milestone+1 覆辙。

### 4.7 V5(a) · 结构接法下游重训(内在对比,本机 2 卡)✅(2026-07-14)
脚本 `p1_libero_rvalley_pairs.py`(建对)+ `p1_train_lmwm_heldout.py`(训+留出评)。同模型/步数/协议,只换 pairs;episode 级 15% held-out 评 `recon_cos−persist`。

| pairs | held-out recon_cos | persist(不动基线) | **GAIN** | 覆盖 |
|---|---|---|---|---|
| milestone(边界目标,旧) | 0.9688 | **0.9315** | **+0.0373** | 95395对(丢末段) |
| **r-脊(canonical+末段锚,新)** | 0.9697 | **0.8906** | **+0.0792(2.1×)** | 137154对=100% |

**结论**:两者 recon 几乎一样,但 **milestone persist 高达 0.93 = 目标离当前太近 → world-model 近乎复制当前帧(gain 仅 +0.037)**,正是 −4.2pt 机理;**r-脊目标更远(persist 0.89)却预测得一样准 → 前向建模增量 2.1×**,真在预测下一 canonical 态。→ **结构接法让 world-model 前向预测翻倍**,内在支持优于旧 milestone。
**诚实保留**:内在 recon 跨不同目标有混淆(GAIN 已部分归一 persist);**最终判据 = sim-SR**(需 lawam 部署 infra / gsy,和 gsy 上 milestone-vs-baseline 同框架对比)。

**✅ gsy 北京队列已提交并 RUNNING(2026-07-16)· 8 卡版**:`t-20260716105522-vsnqm`(lmwm-rvalley-recurridge-**8h20**,Robot-North-H20,`ml.hpcpni3ln.45xlarge` 8×H20 整节点,**12500步/save2500**;CUDA_VISIBLE_DEVICES=0..7 用满 8 卡,墙钟 ~2× 快,样本量 = 4卡×25000 → 与 milestone 可比)。(旧 4 卡 `t-...7xj7k` 已 stop。)
- 三件套已在 North-E:`lmvla/lmwm/data/libero_rvalley/{pairs.npz,target_compact.npz(2.12GB)}` + `checkpoints/lmwm_libero_rvalley/lmwm.pt`;yaml=`train_scripts/kai/volc/lmwm_rvalley_recurridge_4h20.yaml`。
- **volc 坑(已记忆)**:Robot-North-H20 **不支持 FlexibleResourceClaim**,必须整节点 flavor:4卡=`ml.pni3ln.17xlarge` / 8卡=`ml.hpcpni3ln.45xlarge`;`train_lawam_distributed.sh` 用 `nvidia-smi --list-gpus` 数卡(不看 CUDA_VISIBLE_DEVICES)→ 要用满 8 卡须 CUDA_VISIBLE_DEVICES=0..7。
- 训完 → ckpt 在 North-E → 接 lawam LIBERO sim eval,和 Arm M(milestone)/ Arm B(baseline)同框架比 SR = V5 终判。监控:`mlp job get t-20260716105522-vsnqm`。
- **✅ 训练完成(2026-07-16 08:27 UTC,12500步 5:18h)**:loss 收敛佳——total 0.2255→0.0102(22×)、distill(r-脊目标)0.725→**0.0023**(300×)、**val_loss_distill 0.604→0.0020**(无过拟合)。ckpt(job `t-...47pfw`,venv 版):`.../results/Checkpoints/libero/20260716_030632+lmwm_rvalley_recurridge_volc/checkpoints/steps_{2500..12500}_pytorch_model.pt` + `final_model/`。
- **剩:V5 终判 = LIBERO sim eval**(Arm M''=steps_12500 vs Arm M=milestone vs Arm B=baseline 的 SR),需 lawam policy server + libero env(RESULTS 里那套本机 A100 infra)。

### 4.8 视觉别名诊断 · image-only vs image⊕proprio ✅(2026-07-16)
脚本 `recurrence_aliasing_diag.py` · 图 `assets/recurrence_aliasing_diag.png`。别名=图像近、真实时间远(近邻时间 std)。
| 任务 | img time_spread | +proprio | 别名率>0.15 | corr(别名,r粗糙) |
|---|---|---|---|---|
| kai0 | 0.079 | 0.071 | 13% | +0.09 |
| LIBERO-task0 | 0.059 | 0.029(-52%) | 10% | +0.07 |
| LIBERO-task6 | 0.049 | 0.035(-29%) | 0% | +0.06 |
| robotwin x2 | 0.11~0.12 | 0.05(-56%) | 32~36% | ~+0.03 |

**结论**:①加 proprio 确实分开别名帧(中位 -52%);②别名是本体属性,robotwin 双臂最重(36%)、task6 反而最低(全局指标抓不到 mid≈end 局部窄别名);③别名几乎不腐蚀 r 场(corr≈0,r 对多 episode 取平均)。
**判定**:保持 image-only 普适默认(r 对别名鲁棒 + img-only 世界模型已 +2.1× gain + proprio 增益历史证据小);proprio 列 robotwin opt-in(per-task z-score+l2 能量1:1 是同一普适规则,不破坏普适)。诚实保留:粗糙度指标抓不到别名的平滑偏置。

### 4.9 V5 终判 · LIBERO sim-SR 三臂 + 根因 + V6 设计 ✅/🔄(2026-07-16)
North-E 干净 eval(修 3 个环境坑:LIBERO config 绝对路径 / assets 缺 119 PNG 已同步 / eval server 走 PYTHONPATH)。Arm M'' = steps_12500(8卡=4卡25000同样本量),50 trials×10 task,seed 0。权威 `summary.json` **469/500 = 93.8%**。

**逐任务三臂**(M''本次North-E;M/B 来自本机 `data/lmwm_vs_lawam_libero10_20k.json` step20000,**口径未完全对齐**):

| task | 任务 | M'' r-脊 | B base | M ms | M''−B | M''−M |
|---|---|---|---|---|---|---|
| 0 | soup+tomato | 96 | 98 | 98 | −2 | −2 |
| 1 | cheese+butter | 100 | 100 | 96 | 0 | +4 |
| 2 | 灶台+moka壶 | 94 | 100 | 100 | **−6** | −6 |
| 3 | 黑碗入抽屉+关 | 100 | 96 | 94 | **+4**✅ | +6 |
| 4 | 双杯左右 | 98 | 98 | 100 | 0 | −2 |
| 5 | 书入caddy | 100 | 100 | 98 | 0 | +2 |
| **6** | **白杯+巧克力布丁** | 80 | 84 | 68 | −4 | **+12**🔥 |
| 7 | soup+cheese | 100 | 100 | 100 | 0 | 0 |
| **8** | **两个moka壶入灶** | 88 | 100 | 86 | **−12**⚠️ | +2 |
| **9** | **黄白杯入微波+关** | 82 | 88 | 82 | **−6** | 0 |
| **聚合** | | **93.8** | 96.4 | 92.2 | **−2.6** | **+1.6** |

**两条读数**:①**r-脊价值成立**:+1.6pt vs milestone 几乎全来自 task6(+12,milestone 最烂的弥散任务)→ 兑现「脊>边界」。②**vs baseline 差 −2.6 全在别名任务**:task8(两个相同moka壶)−12、task2/9(moka壶/微波炉遮挡终态)−6。只在 task3 超 baseline。

**根因(设计层,非表面)**——B/M'' 同架构(LaWAM+WM头),只差 **目标(t+7 vs r-脊)+ swap_teacher**,故 task8 −12 精确隔离出目标选择的代价:
- **B 的 t+7 = 局部相对自监督目标**:预测「顺当前轨迹 7 步后到哪」,是局部动力学外推,**不需要在流形上全局定位**;target 是自己对真未来的编码,无外部标签可指错;train/inference gap 极小。→ **天生绕开别名**。
- **M'' 的 r-脊 = 全局语义目标**:要回答「我在 canonical 结构的哪个收敛点」——**全局定位问题**。而定位发生在 **冻结 mean-pooled DINOv3 空间**,该空间为普适(C1)选成 **实例不变 / 空间池化 / 无 proprio**,恰好对「两个相同物体的哪一个、门开/关的遮挡态」**多对一塌缩**。信息在编码器就丢了,下游 r-脊分段/target 检索/生成器都继承这个塌缩 → 注入**系统性指错**的 hint;swap_teacher 又把策略训得无条件服从 → 翻车。
- **关键澄清**:§4.8 已证别名**几乎不腐蚀 r 标量场**(corr≈0);病不在 r 值,在 **target 的分配/检索**(把 pot-A相位 与 pot-B相位 检索成同一脊)。故 §5 里「proprio probe 0.96→0.97」的否决测的是 **r-场层面**,与此处 **target 塌缩** 是不同 locus,不能据此断定 proprio 对 task8 无用——但也不急着回退 proprio。
- **两种别名要分开**:(a)**重复物体**(task8)信息在像素里(空间位置),被 pooling 丢;(b)**遮挡终态**(task2/9)信息**根本不在当前单帧**(门挡住),任何前馈感知都救不了,需时序/proprio 记忆。二者殊途同归到「错脊目标」但根因不同。
- **一句话**:baseline 不是更强,是**从不下「在有损池化空间里全局定位」这个会爆的赌**;r-脊下这个赌,在 task6 赢、在 task8 爆。

**V6 设计(修法,不改 r 信号本身,减小坏 hint 的伤害)**:
1. **hint-dropout**:训练时以概率 p 把 milestone latent 换成可学习 null embedding → 策略学会 hint 缺席时回退 base 能力(保住 baseline 底座),坏 hint 不再致命。p∈{0.15,0.25} 各一版(p 过大稀释 task6 真有用的 hint)。
2. **r 置信门控目标插值(更本质)**:用 recurrence 第三读法(r 幅值=on-manifold/OOD,AUROC 0.999)做门:`a=a_base+g(r)·(a_hint−a_base)`,**high-r→r-脊(吃 task6 红利)/ low-r→退化 t+7(回 B 的抗别名安全区)**。等于用 recurrence 自己的置信信号把 B 的抗别名性和 M'' 的长程收益缝起来;hint-dropout 是其 g 随机置 0 的粗糙特例。
- **预测**:task8/2/9 回升向 baseline,task6 基本不掉,聚合有望 93.8%→96%+。
- **history gate ✅**(2026-07-16):CRAVE HISTORY §2 + 本 §5 对 dropout/guidance/门控/插值 零命中,未撞旧方案。

**任务队列(V6 + 补齐)**:
- [ ] **T1 同口径重eval**:用修好的 North-E harness 把 Arm B / Arm M 重跑到 step12500(同环境同step),坐实三臂 per-task 表(现成 armB/armM yaml + config/assets 修复,可两节点并行 ~1.7h)。
- [ ] **T2 hint-dropout 训练**:训练 forward 加概率 p null-embedding 遮挡 milestone latent;出 p=0.15 / p=0.25 两版 ckpt。
- [ ] **T3 r-门控目标插值**:实现 `a_base + g(r)·(a_hint−a_base)`(g 随 r 单调);一版 ckpt。
- [ ] **T4 三选优 eval**:T2/T3 三版 + M'' 基线在 North-E 同口径 eval,看 task8/2/9 是否回升、task6 是否保住。

---

## 5. 已排除(勿回头,详见各 HISTORY)
- per-mode coverage≥0.5:kai0 专属,LIBERO 塌 M=2(消融)。
- BGMM 视觉聚类:LIBERO 低视觉方差 → 塌成 1 component,γ 四数量级无效(2026-07-14 实测);时间注入只在窄 w 甜点脆弱工作 → **视觉聚类不普适**。
- 固定 K / K=0.55√N:非自适应 / 按长度错轴。
- 双锚 Viterbi:kai0 专用,LIBERO 吸尾(last_frac 0.30>0.08)。
- proprio:probe 0.96→0.97 几乎不加分。

---

## 6. Positioning & 相关工作(paper 用 · 2026-07-16 查新后)

### 6.1 一句话定位
把示范集当作**冻结视觉-语义(DINOv3)空间里的非参数知识库**,逐帧定义 **Recurrence Density `r(o)` = 检索增强的、按"来源多样性"加权的 kNN 密度**(有多少条**不同 demo** 各自独立检索到 o 附近的态,median-heuristic 带宽)。这个**连续场**(非离散 milestone 聚类)是**一个普适、多用途的机器人信号**:幅值=在不在流形上(部署 OOD),脊=canonical 收敛态(世界模型子目标/目标),谷=跨-demo 分歧点(子任务边界);训练时挖库、部署时**蒸馏成参数化 latent world model(零检索)**。一套超参跨 3 本体(kai0/LIBERO/robotwin)。

### 6.2 相关工作五轴 + 直接对标(含 2025–26)
| 轴 | 最近对标 | 与本工作差 |
|---|---|---|
| **复现/瓶颈→子目标** ⭐祖先 | McGovern & Barto 2001《Diverse Density 子目标发现》(MIL, Maron&Lozano-Pérez 1998) | 他们把 diverse density **阈值化成离散 subgoal/option(tabular RL)**;本工作保留**连续场**、多用途、接 VLA/世界模型 |
| **部署 OOD/失败检测** ⭐最近 | **FAIL-Detect 2025**(流式密度建"成功轨迹分布",偏离=失败)、kNN-OOD(Sun 2022)、Mahalanobis(Lee 2018) | 他们是**学习式密度 + 单一用途(只做失败检测)**;本工作是**非参数检索密度 + 同一原语还做分割/子目标/加权** |
| **检索增强 VLA** | Retrieval-VLA 2026、MAP-VLA、Retrieve-Don't-Retrain | 他们**检索轨迹内容直接喂策略**(in-context 适配);本工作用**检索统计量(密度/来源数)当信号**,不喂内容 |
| **latent 世界模型 × VLA** | DreamVLA、WorldVLA、AHEAD(预测未来 patch token) | 他们预测**固定 horizon 未来 token**;本工作的 novelty 在**预测什么目标**——r-脊(canonical 复现态)胜过固定 horizon/边界(见 §4.6/4.7) |
| **示范学 progress/value** | TCC(Dwibedi 2019)、VIP/LIV、GVL(VLM 价值学习 2024)、RECAP | 同目标;r-脊→双锚 Viterbi progress 更普适+可与上述任一互补 |

### 6.3 新在哪(delta,可辩护)
1. **统一**:一个原语(cross-ep 检索密度)同时兑现 OOD 监控 + 子任务分割 + 子目标/世界模型目标 + BC 加权——上面每条轴单独都有先例,**但没人用同一连续场打通全部**。
2. **连续场 + 来源多样性**:区别于 ①离散 diverse-density subgoal(McGovern-Barto)、②学习式 flow 密度(FAIL-Detect)、③内容检索(Retrieval-VLA);"数不同 demo 非帧"→ dwelling 鲁棒。
3. **机理发现(最硬的新点)**:**r-脊(canonical 收敛点)是比 milestone 边界/固定 horizon 更好的世界模型目标**——量化解释了 milestone+1 的 −4.2pt,内在前向 gain 2.1×(§4.6/4.7)。这是可复现、可证伪的新洞察。
4. **跨本体普适**:一套超参跨 dual-Piper/Panda/ALOHA(§4.4),多数 WM-VLA/OOD 工作单本体。

### 6.4 审稿人会引的 + 反驳
- "这不就是 diverse-density(2001)+ kNN-OOD 套到 VLA" → 反:①连续场非离散、②统一多用途非单点、③ridge>boundary 的机理发现是新的且解释了真实负结果。
- "FAIL-Detect 已做密度 OOD" → 反:它单一用途 + 学习式密度;本工作同一原语多用途 + 非参数 + 蒸馏进世界模型。
- "DreamVLA 已做 WM×VLA" → 反:贡献不在"接世界模型",在"**世界模型该预测 r-脊而非固定 horizon**"这个目标选择 + 其普适 recurrence 来源。

### 6.5 查新结论
- **Novelty ≈ 6.5/10 · PROCEED WITH CAUTION**。原语(密度)与每个单独用途都有先例;**新意在统一 + 连续 recurrence-density 框架 + ridge-target 机理发现 + 跨本体普适**。
- **写法建议**:标题/摘要主打**统一的连续复现场**和 **ridge>boundary 的世界模型目标发现**,而非任何单一用途;显式区分 FAIL-Detect(OOD)/McGovern-Barto(subgoal)/DreamVLA(WM),把它们放进对比而非回避。
- **待补**:sim-SR 三臂(Arm M'' vs M vs B)是把"内在 gain 2.1×"变成落地 SR 的关键证据(V5,gsy 训练中)。
- 查新工具:WebSearch(Codex 交叉模型本会话不可用,以 web 检索+自评替代)。
