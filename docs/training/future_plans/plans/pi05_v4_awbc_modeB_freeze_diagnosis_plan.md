# Mode B(回折过渡冻结)根因逐步排查 — 诊断 plan

> **建立**: 2026-07-06
> **目的**: 真机上 pi05 v4 AWBC 策略在**回折过渡**处出现 **20–35s 冻结(Mode B)**;`pi05_v4_awbc_dct/49999`(续训前)与 `pi05_v4_awbc_dct_freshft/29999`(续训后)**都冻结**。用**单变量逐级消元**定位根因:**第一步去掉 DCT**,同数据同参重训;不解决再逐级往下。
> **状态**: ✅ **根因确诊 + 修法验证 (2026-07-14)**。dagger 迟疑起手边界段(06-16 变点)是冻结根因,双向起爆点前裁(launchtrim)是修法。真机验证:裁后不再冻结。详见 §9 + §10。
>
> ### 📊 2×2 证据(DCT × freshdagger 微调)
> | ckpt | DCT | 微调 | Mode B 冻结 |
> |---|---|---|---|
> | `pi05_v4_awbc/49999` | 无 | 无 | ✅ **不冻** |
> | `pi05_v4_awbc_dct/49999` | 有 | 无 | ❌ 冻 |
> | `pi05_v4_awbc_dct_freshft/29999` | 有 | 有 | ❌ 冻 |
> | `pi05_v4_awbc_freshft_nodct/29999` | 无 | 有 | ❌ **仍冻** |
> → 冻结 = **(DCT=有) OR (freshdagger微调=有)**;两个独立充分诱因。
> **上游分析**: 本 session 已排除 —— 不是 freshdagger 微调(续训前也冻)、不是欠训(50k 也冻)、不是 AE value 塌缩(value std 0.227 健康)、不是数据学来的(训练数据 0 个 ≥5s 冻结段)。→ **20–35s 冻结是部署时涌现**,真因在两个 ckpt 共有的 **DCT / AWBC-always-positive / flow-matching 多峰塌缩 / proprio 捷径**。
> ⚠️ **铁律**: 真机为终判(offline MAE 对冻结盲);判据 = 真机回折过渡是否还冻 >5s。

---

## 0. 问题定义 + 候选根因(消元顺序依据)

**Mode B**: 回折过渡(衣物折成长条、双臂在两端)处,`|action−state|≈0.005`、臂 std 0.015(≈0.8°)、夹爪闲置张开、0.05Hz 微颤,持续 20–35s → 模型输出"保持当前姿态"。

**机制假说(自强化环)**: 回折起爆是**多峰决策边界** → 无主峰时 flow-matching 采样落到最低能量的 **hold 峰(动作≈state)** → hold→下一帧同态输入→再 hold,自强化。三个放大器(两 ckpt 都有):

| # | 候选放大器 | 证据强度 | 排查级 |
|---|---|---|---|
| 1 | **DCT loss** 压掉果断高频起爆动作 | ✅ 坐实为**诱因之一**(E0)| — |
| 1b | **freshdagger 微调**(506×22s 纯纠错短片段 → 策略漂向纠错分布、丢完整任务平滑覆盖)| ✅ **坐实为独立诱因**(E1:无 DCT 微调后仍冻)| **E4→主线 / 部署兜底** |
| 2 | **AWBC always-positive hold 先验** | 🟡 positive 类 36% 低速、adv↔速 r=−0.11(弱但一致)| E2 |
| 3 | **proprio 捷径**(echo state 自洽不动点)| 🟡 pi05 吃 state;vision-blind 前科 [[project_xvla_action_repr_d5]] | E3 |
| 4 | **回折起爆覆盖稀/OOD** | 🟡 dagger 短纠错片段不覆盖完整起爆;base 部分完整 | E4 |
| 5 | **flow-matching 多峰塌缩(架构固有)** | 🟢 底层机制,前四个都放大它 | E5 |

→ **按"最可能 + 最省"排序:E1(DCT)→ E2(AWBC先验)→ E3(proprio)→ E4(覆盖)→ E5(架构/部署)**。任一级真机不冻即定位成功、停止。

---

## 1. 逐级排查阶梯(每级单变量)

### E0 —— ✅ 已完成:非 DCT 双胞胎真机预检 → **DCT 坐实为根因**
真机测**已存在**的非 DCT 双胞胎 `pi05_v4_awbc/49999`(同 `A_v4_base_dagger` 数据、同 AWBC、同 pi05_base init、同 50k、**唯一差=无 DCT**):**实测无长冻结**。
- **结论**: `pi05_v4_awbc`(无DCT,不冻)vs `pi05_v4_awbc_dct`(有DCT,冻)= 单变量自然实验 → **DCT loss 是 Mode B 根因**。E1 转为确认 + 产出可部署修复版。

### E1 —— ⭐ Step 1:去 DCT 重训(确认 + 产出可部署修复版)
**唯一变量 = 关 `use_dct_loss`**,数据/参数与 freshft 完全相同。目标 = 拿到**同时修好夹爪(fresh dagger)+ 冻结(无 DCT)**的可部署模型。
- config `pi05_v4_awbc_freshft_nodct` = 克隆 `pi05_v4_awbc_dct_freshft`,`model=Pi0Config(pi05=True)`(去掉 `use_dct_loss=True`),其余逐字段不变(data=`A_v4_freshdagger_ft`,LR warmup500/peak1e-5/decay30k,30k,batch128,fsdp8)。
- ⚠️ **init 洁净度**:freshft 的 init 是 `pi05_v4_awbc_dct/49999/params`(DCT 血统)。两选:
  - **E1a(字面同参)**: init 仍用 `pi05_v4_awbc_dct/49999/params`,30k 续训覆盖 DCT 行为。
  - **E1b(洁净隔离,推荐)**: init 换非 DCT 的 `pi05_v4_awbc/49999/params` → 全链路零 DCT。
- **⚠️ 必做:去 DCT 是否丢平滑收益?** DCT 当初为压 Mode A 抖动引入。E1 真机需同时看**冻结(应消失)+ 抖动/Mode A(是否回来)**:
  - 非 DCT 若**不抖** → DCT 纯有害,直接弃,E1 即终态。
  - 非 DCT 若**抖动回来** → 需 **DCT-lite**(降 `dct_high_freq_weight`/`dct_loss_weight`,只压真高频毛刺、不压果断起爆),或用**部署侧动作平滑**替代 DCT。→ 见 E1'。
- **⚠️ E1 实测结果(2026-07 真机)**: `pi05_v4_awbc_freshft_nodct/29999` **仍有长冻结** → **去 DCT 不足以修复**。结合 2×2 → freshdagger 微调是**第二个独立诱因**。分支转 **E1.5 + E4/部署兜底**。

### 📊 数据集诊断(2026-07,已定位差异)—— 不冻源 vs 冻结源
| | `A_v4_base_dagger`(不冻)| `A_v4_freshdagger_ft`(冻)|
|---|---|---|
| base:dagger 配比 | **60:40(demo主导)** | **48:52(dagger主导)** |
| dagger ep 长 | 1280fr/**43s** | 660fr/**22s** |
| dagger 短片段 <500fr | **7%** | **38%** |
| dagger <1000fr | 31% | **82%** |
| dagger 日期 | 05-29~06-23 | 06-29~07-03 |
| 数据自身 hold率 / ≥5s冻结 | 20.7% / ~0 | 17.1% / 0 |

→ **两处差异共同破坏完整任务时序连贯**:① **新 dagger 采集方式变了**——碎片化短抓拍(22s,38% <17s vs 旧 7%),遥操只在失败瞬间介入几秒;② **配比反转**(demo 主导 → 短纠错碎片主导)。两数据集自身 hold 率都低、都无长冻结 → **冻结不是学来的 hold,而是短碎片主导丢失了"第一折→第二折过渡"的完整轨迹覆盖** → 部署回折点塌缩。`A_v4_base_dagger` 的 dagger 是较完整纠错段(43s)保住连贯 → 不冻。

**✅ 排除"冻结错误引导"(全量扫描 973ep)**:数据里**没有任何 episode 教冻结** —— 全集最长静止段仅 **3.0s**,**0 个 ≥5s 静止**,**0 个长静止被标 positive**;用户标记的 06-18/06-28 是干净的长完整 demo(85s/106s 连续,无 ≥3s 静止)。→ 真机 20-35s 冻结是**部署涌现(模型放大 6-10×)**,非数据学来 → **修法是改数据构成让完整轨迹主导,不是删坏 episode(没有坏的)**。

**base 日期差异(base 确实带完整过程,但被稀释)**:冻结集 base 用"取最新 ~467ep"逻辑 → **丢了整段 4 月(739ep,含最长的 04-30 66s / 05-06 61s 完整 demo)**,只留 05-07~06-04(且含偏短的 05-18 26s / 06-04 **5s**)。base 从 1207→467(−61%),完整 base(05-07~10 的 52-61s ≈243ep)只占总量 ~25%。→ **完整任务信号没消失,但从主导跌成少数派(48% base 中还含短的),被 52% 短 dagger 碎片压过**。⚠️ 教训:**别用"取最新 N ep"选 base(牺牲完整性),应按 episode 完整性/长度选**。

### E1.5 —— 隔离"微调数据构成"这一诱因(证据已足,可跳过直接修)
2×2 已证 freshdagger 微调独立致冻;若要坐实是**"纯纠错短片段"数据构成**(而非"续训"这个动作本身),做一次对照:同 no-DCT init,微调数据换成 **fresh dagger + 完整任务 base demo 混合(含回折起爆完整轨迹)**,其余不变。
- 不冻 → **数据构成**(dagger-only 短片段)是因 → 走 E4 主线。
- 仍冻 → "续训"动作本身有害 → 放弃微调路线,走**部署兜底**(下)。

**⚠️ 澄清(总时长重算)**:冻结集 base 时长占比其实**已更高**(63% vs 不冻集 57%)→ 单纯"base>dagger 时长"非判别因素。真判别 = ① base **数量/多样性**(1207→467 砍半)② dagger **碎片化**(43s→22s)。修法要**加 base 数量 + 把 base 时长占比推到远超 63%**。

### 🍳 路线乙数据配方 `A_v4_fullbase_freshdagger`(已量化,可直接 build)
| 组 | 内容 | ep(净)| 时长 |
|---|---|---|---|
| **base** | **全部 16 个 base v4 日期**,剔 ≥2s静止(22ep)+ 可选剔 06-04(5s残段)| ~1244 | ~793min |
| **dagger** | fresh 06-29~07-03(gripper修复),剔 ≥2s静止(2ep)| 504 | ~185min |
| **配比** | | **2.5:1(ep)** | **4.3:1(时长)= base 81%** |
- 满足三要求:剔 ≥2s静止(24ep)/ base 数量 467→1244 / base 时长 4.3× dagger(远超冻结集 63%)。
- **训练**:init=非DCT `pi05_v4_awbc/49999`,`use_dct_loss=False`,30k;norm 对新集重算;夹爪不裁。
- ⚠️ **gripper 张力**:base 旧语义(action==state)81% 占比 → fresh dagger(gripper-from-master)成少数 → 若微张开回来,叠加部署侧 clamp(idx6/13→state)正交兜底。

### E2 —— 破 AWBC always-positive 的 hold 先验
唯一变量 = **positive 类不含低速帧**。重做 discretize:**velocity-aware** —— 把臂速 <阈值的帧从 positive(task_index=1)剔除/降权,只让"果断进展"帧进 positive。数据/init/DCT 设置沿用 E1 的胜出配置。
- (备选 E2') 直接对照**纯 BC**(去 advantage prompt,default_prompt 固定)版是否还冻,隔离"AWBC 条件"本身。
- **判据**: 不冻 → AWBC-positive-hold 是因。仍冻 → **E3**。

### E3 —— 降 proprio 依赖(破 echo-state 不动点)
唯一变量 = **proprio dropout / 关 state 输入**(vision-blind 修复,见 [[project_xvla_action_repr_d5]] E1 `use_proprio=False` 思路)。沿用前级胜出配置。
- **判据**: 不冻 → proprio 捷径是因。仍冻 → **E4**。

### E4 —— 补回折起爆的果断完整示范(覆盖)
唯一变量 = **数据**。在训练集里**上采样/补入**"长条→第二次回折起爆"的完整果断轨迹(从 base 完整 ep 抽,或新采集专项 demo);其余沿用胜出配置。
- **判据**: 不冻 → 覆盖/OOD 是因。仍冻 → **E5**。

### E5 —— 动作头 / 部署侧(架构固有多峰塌缩)
- **E5a 部署反冻结(0 成本,可先并行上)**: 检测臂动作范数连续 N 帧 <阈值 → 抬高 flow-matching 采样温度 / 注入小扰动,把状态踢出 hold 峰;或脚本"轻推"。
- **E5b**: 提高部署采样温度 / 换动作头 / RTC-chunk 调整,打散多峰塌缩。
- **判据**: 兜底能出冻结即可上线;E5b 是深水区,视前面结果决定投入。

---

## 2. 评估协议(真机为终判 + offline 代理筛选)

- ⚠️ **offline MAE 对冻结盲** → 每级必须真机测**回折过渡**是否冻 >5s。
- **offline 代理(省真机次数)**: 每个候选 ckpt 在一批**回折过渡 val 帧**上前向,量**预测动作范数**;若在这些帧上输出≈0(低范数)→ 高概率真机会冻。先用代理筛掉明显不行的,再上真机。
- **统一记录**: 每级记 { 真机冻结时长/次数、回折过渡通过率、offline 动作范数、成功率 }。

**冻结判据(单级 go/no-go)**:
- ✅ 解决 = 真机回折过渡**无 >5s 冻结** + 成功率不降。
- ⚠️ 缓解 = 冻结时长明显缩短但仍有 → 记录,继续下一级(叠加)。
- ❌ 无效 = 冻结依旧 → 进下一级。

---

## 3. 落地步骤
1. **E0**(免费): 真机测 `pi05_v4_awbc/49999`,填冻结判据。
2. **E1**: 注册 `pi05_v4_awbc_freshft_nodct`(去 DCT,推荐 E1b init 非DCT)→ commit/push → 8 卡 30k 重训 → offline 代理 → 真机。
3. E1 不解决 → **E2**(velocity-aware discretize 重打标 → 重训 → 真机)。
4. E2 不解决 → **E3**(proprio dropout 重训 → 真机)。
5. E3 不解决 → **E4**(补果断起爆示范 → 重训 → 真机)。
6. 全程并行 **E5a 部署反冻结兜底**(不依赖根因,先给安全网)。
7. 每级回填本文档结果 + 更新 master history;定位到即结束。

---

## 4. 风险 / 注意
- **叠加效应**: 若单级只"缓解"非"消除",真因可能是**多因叠加**(DCT+AWBC先验+proprio)。策略:每级在**前级胜出配置**上继续加,而非从头,末尾得到最小充分修复组合。
- **init 血统混淆**(E1): 见 §1 E1a/E1b;归因存疑就跑 E1b。
- **真机成本**: 每级一次真机;用 offline 动作范数代理先筛,减少上机次数。
- **DCT 的两难**: DCT 本是为压抖(Mode A/抖动)引入;若 E1 去 DCT 修好冻结但抖动回来 → 需 DCT 权重折中(降 high-freq 权重而非全关),或用部署侧平滑替代。
- **base 旧语义**: freshft 数据的 base 是 action==state 旧语义(仅夹爪维);与 Mode B(臂冻结)关系小,但换数据的 E4 要留意别把夹爪语义又搅乱。

---

## 5. 决策树(一图流)
```
E0 非DCT旧ckpt(无微调) ──✅不冻──→ DCT 是诱因之一
E1 非DCT freshft(有微调) ──❌仍冻──→ ★微调是【第二个独立诱因】★
   │ → 2×2: 冻 = (DCT) OR (freshdagger微调)
   │
   ├─【路线甲·最快·推荐】部署侧兜底 on 不冻的 pi05_v4_awbc/49999:
   │    夹爪 clamp(idx6/13→state, 修微张开) + 反冻结(低动作范数→提采样温度/轻推)
   │    = 无训练拿到「不冻 + 夹爪修复」 → 先上线
   │
   └─【路线乙·训练修复】E1.5: no-DCT + (fresh dagger + 完整任务demo混合) 重训
        ├─不冻→ ✅数据构成是因, 用此配方(修夹爪且不冻)
        └─仍冻→ 微调动作本身有害 → 弃微调, 用路线甲
(E2 AWBC先验 / E3 proprio 仅在路线乙仍冻时才排查)
```

---

## 8. ⭐ 落地:两个 8 卡对照任务(2026-07 用户定)

> 思路:**保住已证「无冻结」的 `A_v4_base_dagger` + 无 DCT 配方不动**,任务①做基线锚,任务②在其上**只加**新 dagger 验证"能否修夹爪且不重新冻结"。全程 **no DCT**。

### 任务① —— 无 DCT 基线(control)= `pi05_v4_awbc`
- **数据**: `A_v4_base_dagger`(与 `pi05_v4_awbc/49999` **完全相同**,2017ep)。
- **config**: `pi05_v4_awbc`(**已注册**;= `pi05_v4_awbc_dct` 除 `use_dct_loss` 外逐字段相同 → 即"除 DCT 外全同")。init=pi05_base,50k,LR warmup1k/peak1.5e-5,batch128,fsdp8,**无 DCT**。
- **现状**: ✅ **已训 `pi05_v4_awbc_gf3`(exp tszn6,A_v4_base_dagger cnsh-prune 版 2006ep)→ 真机几乎不再冻结**(2026-07-08 用户确认);另 `49999`(2017ep)早已验无冻结。仅夹爪微张开未修。
- **作用**: "无冻结 / 夹爪未修" 的对照锚。**⭐ 关键**:tszn6(①)与 plus_freshdagger(②)**共用同一份 2006ep pruned base_dagger,唯一差 = ② 多 506 fresh dagger** → ①不冻/②冻 = **单变量对照坐实 fresh dagger 是致冻因**(airtight)。

### 任务② —— 在①数据基础上 **额外加新日期 dagger**(treatment)
- **数据**: `A_v4_base_dagger` **全量** + **额外 fresh dagger 06-29~07-03(506ep)** → 新集 `A_v4_base_dagger_plus_freshdagger`(~2523ep)。
  - 关键:**保留①的整套配方不动**(base 1207 + 旧 dagger 810 = 已证不冻的主体,占 ~80%);fresh dagger 仅作 **~20% 补充**加入 gripper-from-master 新语义 → 修夹爪但主体仍不冻。
  - **build**: 合并 raw(base 13 日期 + 旧 dagger 12 日期 + **新 dagger 5 日期**)→ AE `adv_est_v1` 打标 → discretize **top-30%**(阈值在**全集**上算,保持一致)→ labeled。
  - ⚠️ **norm 对新集重算**;夹爪不裁;(可选)剔 ≥2s 静止 ep。
- **config**: `pi05_v4_awbc_plus_freshdagger`(**克隆 `pi05_v4_awbc`**,只换 `repo_id`→新集,**无 DCT**,其余逐字段同:init=pi05_base,50k,LR 同)。
- **8 卡**,cnbj/cnsh 择空闲。
- **判据(真机)**: **无回折冻结**(base+旧dagger 主导 → 应保持)+ **夹爪微张开消失**(fresh dagger 新语义生效)。
  - ①无冻/夹爪坏 → ②应 无冻 + 夹爪好 = 成功。
  - 若②又冻 → 20% fresh dagger 即致冻 → 退部署侧兜底(clamp + 反冻结)。

> 与 §E1.5「🍳 路线乙配方」的区别:E1.5 是"全 base + 仅 fresh dagger(去旧 dagger)";**任务② 更保守 —— 完整保留已证不冻的 `A_v4_base_dagger`,只做加法**。

### 任务③ —— 在 tszn6 基础上**只加 DCT**(隔离 DCT 效果 + 量化平滑收益)
> 目的:① 在与 tszn6 **完全相同的 2006ep pruned 数据**上确认 **DCT-alone 致冻**(补齐 DCT×fresh-dagger 对照矩阵,消除旧 `pi05_v4_awbc_dct/49999` 用 2017ep 的混淆);② **量化 DCT 的 Mode A 平滑收益**——DCT 当初就是为压抖动引入的,需知它换来的平滑是否值得冻结代价。
- **数据/init/LR/步数**:与 tszn6 **逐字段相同**(A_v4_base_dagger 2006ep pruned、pi05_base、warmup1k/peak1.5e-5、50k、bs128、fsdp8)。
- **唯一变量 = `use_dct_loss=True`**(`Pi0Config(pi05=True, use_dct_loss=True)`)。
- **config**:复用 `pi05_v4_awbc_dct`(其 repo_id=A_v4_base_dagger 现即 2006ep pruned 版),**exp_name `pi05_v4_awbc_dct_gf3`** 独立跑,不碰旧 `/49999`。
- **8 卡**,gf3/cnbj 择空闲。
- **判据(真机,与 tszn6 同协议对照)**:
  - **Mode B 冻结**:预期**冻**(DCT 是独立诱因)→ 冻则在 2006ep 上再次确认 DCT 致冻。
  - **⭐ Mode A 抖动/平滑度**:vs tszn6 —— DCT 是否明显更平滑(动作 chunk 高频能量↓)= DCT 的**唯一价值**。
- **判读**:
  - 冻 **且** 明显更平滑 → DCT 有平滑价值、代价是冻结 → 值得做 **DCT-lite**(降 `dct_high_freq_weight`/`dct_loss_weight` 找"不冻且平滑"甜点)。
  - 冻 **且** 不更平滑 → DCT 纯有害 → 彻底弃。
  - 意外**不冻** → DCT 在 2006ep pruned 上不致冻(与旧 2017ep 结论矛盾)→ 复查数据/DCT 实现差异。

**完整对照矩阵(2006ep pruned 数据)**:
| exp | DCT | fresh dagger | 冻结 | 平滑度 |
|---|---|---|---|---|
| ① tszn6 | ✗ | ✗ | ✅不冻 | baseline |
| **③ dct_gf3** | **✓** | ✗ | **预期冻** | **待测(核心)** |
| ② plus_freshdagger | ✗ | ✓ | ❌冻 | — |
| (dct_freshft) | ✓ | ✓ | ❌冻 | — |

---

## 9. 🔴 任务② 结果(2026-07-08)+ 冻结根因收敛

### 完整证据表(唯一不冻 = 无 DCT 且 无 fresh dagger)
| ckpt | DCT | 数据 | Mode B 冻结 |
|---|---|---|---|
| `pi05_v4_awbc/49999` | 无 | A_v4_base_dagger | ✅ **不冻** |
| `pi05_v4_awbc_dct/49999` | 有 | A_v4_base_dagger | ❌ 冻 |
| `pi05_v4_awbc_dct_freshft/29999` | 有 | freshdagger_ft | ❌ 冻 |
| `pi05_v4_awbc_freshft_nodct/29999` | 无 | freshdagger_ft | ❌ 冻 |
| `pi05_v4_awbc_gf3`(tszn6,**任务①**)| 无 | A_v4_base_dagger(2006ep pruned)| ✅ **几乎不冻**(07-08 确认)|
| **`pi05_v4_awbc_plus_freshdagger`(任务②)** | **无** | **①的2006ep + 506 fresh dagger** | ❌ **冻** |
→ **冻结 = (DCT=有) OR (fresh dagger 06-29~07-03 在训练集)**;两个独立充分诱因。**"稀释"被推翻**(fresh dagger 仅 20% 加法、完整保留不冻主体仍冻)。
> ⭐ **airtight 单变量证明**:任务①(tszn6)与任务②**共用完全相同的 2006ep pruned base_dagger,唯一差 = ②多 506 fresh dagger** → ①几乎不冻 / ②冻 → **fresh dagger 因果坐实为致冻因**(排除 base ep 数差、prune 差等一切混淆)。

### fresh dagger(06-29~07-03)vs 旧 dagger(05-29~06-23)差异(实测)
| 指标 | 旧 dagger(不冻)| fresh dagger(致冻)|
|---|---|---|
| clip 长 | **43s** | **22s**(碎片,clip 边界密度 2×)|
| clip 首 15 帧臂速 | 0.0202 | **0.0096**(慢一半 = 迟疑"先评估再动")|
| clip 尾静止收尾 | 80% | 80%(相同,非判别)|
| **AWBC positive 占比** | **22%** | **29%**(fresh 更多帧被标 positive)|
| 低速帧 positive 率 | 24% | 28% |

### 收敛推测(根因)
两个诱因**殊途同归**到同一失败:**回折多峰决策边界上缺乏"果断起爆动作",任何削弱果断性 / 注入迟疑的因素都把策略推入 hold 不动点**。
- **DCT**:频域惩罚直接压掉果断高频起爆动作。
- **fresh dagger**:①碎片化短 clip + **迟疑慢起手**(起手速度仅旧的一半)②在 AWBC **positive 类过表达**(29% vs 22%),而部署永远条件在 positive → 把"在纠错态迟疑慢动"的先验更强注入 deploy 策略。回折过渡 ≈ 纠错起手态 → 触发迟疑 → 冻。
- **为何 20% 就够**:fresh dagger 的影响**集中在 positive 类(deploy 唯一用的)+ 决策点态**,不是均摊 → 小比例也能主导过渡区行为。

### 🎯 关键实践结论
- **唯一不冻配置 = `pi05_v4_awbc`(无 DCT、无 fresh dagger)**,但它有**夹爪微张开**。
- **fresh dagger 是修夹爪的必需数据,但它可靠致冻** → **无法在不引入冻结的前提下用 fresh dagger 训练修夹爪**。
- **→ 转向:`pi05_v4_awbc`(不冻)+ 部署侧夹爪 clamp(idx6/13→state,正交修夹爪,0 训练)** = 同时拿到不冻 + 夹爪修复。这是当前最稳落地。

### 🔬 fresh dagger 致冻机制(clip-start 实证,2026-07-08)
dagger clip 本质 = 遥操员在"机器人卡住/失败"瞬间接管 → **每个 clip 起手帧 = 卡住态**。fresh vs 旧关键差:
| | 旧 dagger | fresh dagger |
|---|---|---|
| 起手→首个果断动作(>0.02)延迟 | 0.27s | **0.53s(2×)** |
| 前 10 帧臂速 | ~0.02 | **~0.008(半)** |
| clip 长 → 起手态密度 | 43s | 22s(**2×**)|
| AWBC positive 占比 | 22% | **29%** |

**链条**:fresh dagger 起手 = "卡住态 + 迟疑 0.5s 再动"(采集风格变/前裁保留了接管-评估段)→ 短 clip 使该模式密度翻倍 → AWBC 把 29% 标 positive、部署永远条件 positive → **回折过渡本身是"低速卡住样决策态",恰匹配 dagger 起手分布** → 部署检索到"卡住→迟疑微动" → 微动留在同态 → 自强化冻结。解释了"20% 就够 / 专在回折 / 旧 dagger 不冻"。DCT 则从另一端(压果断动作)达同样 hold 塌缩。

### 💊 解决办法
- **A 数据外科(推荐,让 fresh dagger 可用)**:① **裁掉每 clip 迟疑起手段**(前裁到臂速首 >0.02 的起爆点,~16帧/clip)→ 只留果断纠错、保留 gripper 新语义 → 有望"既修夹爪又不冻"。② **velocity-aware discretize**(低速帧踢出 positive)。
- **B 训练侧**:down-weight fresh dagger / 压 positive 占比回 ~22%。
- **C 部署侧(正交,最快)**:④ 反冻结看门狗(低动作范数→抬温度/扰动);⑤ **夹爪 clamp(idx6/13→state)→ 根本不需 fresh dagger,直接用 tszn6**。
- **最优组合**:彻底不训 = **tszn6 + C⑤ clamp + C④ 兜底**;训练修 = **A①(裁迟疑)+ A②(velocity-aware)后再加 fresh dagger 重训**(唯一有望训练修夹爪且不冻,并直接验证机制)。

### 下一步诊断(若仍要训练修夹爪)
隔离"fresh dagger 帧 vs 全局重标注/重归一化":加 fresh dagger 但 **discretize 阈值 + norm 冻结为 A_v4_base_dagger 的原值**重训。
- 仍冻 → 是 fresh dagger 帧本身(迟疑/positive 过表达)→ 走 A①+A② 清洗。
- 不冻 → 是全局 re-discretize/re-norm 扰动 → 加数据时固定阈值/norm 即可。

---

## 10. ✅ 根因确诊 + 修法验证 (2026-07-14)

**launchtrim 实验** ([`dagger_launchpoint_trim_freeze_fix_plan.md`](dagger_launchpoint_trim_freeze_fix_plan.md)):

- config: `pi05_v4_awbc_launchtrim` (无 DCT, dagger 双向起爆点前裁 THR=0.02/K=5/M=2)
- 数据: `A_v4_base_dagger_launchtrim` (2510ep: 1200 base 整段 + 1310 dagger 前裁)
- 唯一变量 vs 任务② = dagger 是否裁边界
- 训练: cnbj 8×H20, 50k, loss 0.70→0.003

**真机结果**: ✅ **不再冻结** — 任务② 冻, launchtrim 不冻, 单变量归因成立.

**完整根因链**:
```
06-16 起 dagger clip 变短 + 含 ~0.5s 迟疑低速起手 (v<0.02) + 静止收尾
  → AWBC 把低速帧标 positive
  → 部署 always-positive 放大
  → 回折过渡 (卡住样决策态) 触发"迟疑不动"
  → 剂量效应: 数据够多 (任务②) 则冻, 少量 (tszn6) 不冻
```

**修法**: 双向起爆点前裁 — 前砍迟疑起手 + 后砍静止收尾, 只留果断动作核心.
**参数**: THR=0.02, K=5, M=2, MIN_LEN=30. 前裁 avg 0.33s / 后裁 0.45s, 整体保留 98%.

**两个独立诱因的最终定性**:
| 诱因 | 机制 | 修法 |
|---|---|---|
| DCT | 频域 loss 惩罚高频 → flow-matching 趋保守 → 易塌到 hold | 关 DCT (pi05_v4_awbc_launchtrim 无 DCT) |
| dagger 边界段 | 迟疑低速帧被 AWBC 标 positive → 决策态迟疑不动 | 起爆点前裁 |

两个诱因**独立加性**:关 DCT + 裁边界 = 不冻. 单修任一个不足以在 dagger 污染严重时完全消除冻结.

## 关联
- **修法验证**: [`dagger_launchpoint_trim_freeze_fix_plan.md`](dagger_launchpoint_trim_freeze_fix_plan.md) — 起爆点前裁实验 + 真机结果
- 冻结逐天分析(06-16 变点): 本 plan §9
- 续训 plan(freshft 来源): [`pi05_v4_awbc_dct_freshdagger_finetune_plan.md`](pi05_v4_awbc_dct_freshdagger_finetune_plan.md)
- DCT 来源: [`vlanext_dct_then_soft_connection_plan.md`](vlanext_dct_then_soft_connection_plan.md)
- 非 DCT 双胞胎 ckpt(E0): `kai0/checkpoints/pi05_v4_awbc/pi05_v4_awbc/49999`
- proprio 捷径前科: [[project_xvla_action_repr_d5]]
- launchtrim ckpt: `/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/checkpoints/pi05_v4_awbc_launchtrim/pi05_v4_awbc_launchtrim_cnbj/49999`
- 新格式 dagger 数据 (含 `dagger_frame_class`): `vis_dagger/v4/<date>/data/chunk-001/` (387 ep, TOS 对齐)
