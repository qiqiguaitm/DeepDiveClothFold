# LMWM 试错与踩坑记录(2026-07-03,防重复踩坑)

> 本文沉淀 milestone+1 方向 + gf3 8卡实验里**走过的弯路和最终解**,以后直接查、别再踩。
> 每条:**结论 → 证据 → 怎么做**。

## A. 建模 / 方法层面(哪些方向证伪了)

### A1. latent cosine ≠ 解码保真;别用 L1 选解码器
- **证据**:decode-space loss vs latent loss,latent cos 更低但解码 L1 更好(L8);同数据 grid 预测头,decode-loss 让特征 cos 塌到 0.015 却图像最好(L9)。CRAVE 横评:L1 偏爱模糊会骗人,**再编码 cos** 才是跨解码器统一保真指标。
- **怎么做**:衡量 LMWM 预测质量用 **latent-cos**;要"latent→图"用**检索(0.84)**,合成解码器语义封顶 ~0.47。合成偏软是结构性的。

### A2. 扩散 / 多假设(MHP)subgoal 头**打不过回归**
- **证据**:flow best-of-16(25 ODE步/2×训练)= 0.834 < 回归 0.872;MHP deploy(gate)也输。augin 条件下 next-medoid 条件分布很集中,采样只加噪。
- **怎么做**:milestone+1 latent 用**回归(近条件均值最优)**。多模态生成头这条路**别再投**,除非拿到"未来实际分支"信号(VLA 闭环)。

### A3. 删辅助头会**伤** milestone(多任务=正则)
- **证据**:milestone-only(0.388)< milestone+subgoal 多头(0.393),nll 也更差(H2)。"专注单头"实测更差,单头 NLL/CVaR 暴涨(过拟合)。
- **怎么做**:**保留 subgoal/辅助头当正则**,即使只关心 milestone。

### A4. 更大 backbone / LaWM transformer **不赢 top1**;grid 空间信息**无用**
- **证据**:1024×3 大 trunk 生产过拟合(0.398<0.408);gf3 全量 8成员 LaWM top1 **0.382 < MLP 0.459**;**H7 决定性**:同一 LAMEncoder 喂真实 grid vs 广播 pooled(无空间)结果 **A≈B(差<0.01)** → 空间 token 无增益。
- **怎么做**:milestone+1 用 **pooled MLP** 就够,**别编码 256-token grid**(130GB/8卡全白费)。LaWM 的 top5/subgoal 小优势来自**架构/参数非空间**,不值 ~100× 成本。

### A5. CVaR-CE、λ 都是**配置相关**,别硬套
- **证据**:CVaR-CE 在 pooled MLP 上净赚(均值+方差,H4),搬到 LaWM 却**过度压平 top1**(0.382,nll_std 0.68 异常低)。λ:pooled 上最优≥0.5(H5),但 LaWM 上 λ0.5 反伤(fused<raw)。
- **怎么做**:换模型/数据后 **λ 必扫、CVaR 权重必验**,别复用旧值。

### A6. 对比必须**单变量、同数据**
- **证据**:子集"LaWM-grid vs MLP-pooled"(LaWM 赢)≠ 全量"LaWM vs MLP-augin"(MLP 赢);CVaR+λ+输入多变量一起变,结论被污染。
- **怎么做**:一次只动一个变量,同 split 同 recipe 对比;绝对值受数据规模/recipe 影响大,看**相对**。

### A7. top1 有硬天花板,别只盯它
- **证据**:任务 ~13 分支固有歧义,top1 封顶;真正可拉 = **top-k / NLL / subgoal cos**(也正是 VLA 用法)。

## B. gf3 / 基础设施层面(工程坑)

### B1. kai0 的 transformers(4.53.2)**没有 dinov3**
- **现象**:`AutoModel/AutoImageProcessor.from_pretrained(dinov3)` 报 "Unrecognized ... dinov3"。
- **解**:用**纯 torch 门控 DINOv3-H 编码器**(`lmwm/scripts/dinov3h_gated.py`,复用 openpi `dinov3_vit_standalone` 的 blocks + SwiGLU gated MLP),直接载 safetensors。**本地实测 cos 0.99987 对齐 crave** 再上机。不要升级共享 env 的 transformers。

### B2. `lmwm/vendor/` 被 **gitignore** → git 不同步
- **现象**:gf3 pull 后没有 vendor/LaWAM,`ModuleNotFoundError: latent_action_model...`。
- **解**:vendor 代码要用 **rsync 手动传**(138K,便宜);别指望 git。

### B3. 跨区传输 ~5.7 MB/s → **131GB 别传**
- **证据**:cnsh↔cn-beijing 实测 5.7MB/s,131GB grids ≈ 6.4 小时,不可行。
- **解**:**在数据所在机就地编码**(gf3 有 kai0_base 视频);只传小文件(权重 3.2G + pairs 1.9G ≈ 18min)。

### B4. `pkill -f <脚本名>` 会**自杀当前 shell**
- **现象**:命令行里含该脚本名(nohup 启动串),pkill -f 把执行命令的 bash 也杀了(exit 143)。
- **解**:用 `pgrep -f ... | xargs kill` 按 PID,或更精确的 pattern;别在同一条含目标名的命令里 pkill。

### B5. gf3 路径/同步机制
- gf3 数据在 **/vePFS-North-E**(gpfs,和 cnsh 的 /vePFS **不同 FS,不共享**);repo 在 `/vePFS-North-E/vis_robot/workspace/deepdive_kai0`。
- gf3 **每 1min git pull `reset --hard` main** → **改代码在别处 push,等它 pull;别在 gf3 直接改**(会被覆盖)。
- 临时文件放 `.../deepdive_kai0/temp/`。env 用 `kai0/.venv/bin/python`(fastwam 坏:`_ctypes` 缺失)。gf3 RAM 1.8TB → 123GB grids 全进内存,训练 GPU-bound 不卡 IO。

### B6. H20 上 **fp32 比 bf16 慢 ~4×**
- **证据**:LaWM 训练 fp32 ~0.85s/步 → 8000步 110分钟。
- **解**:**训练/评估都套 `torch.autocast("cuda", bfloat16)`**(bf16 无需 GradScaler)。已加。

### B8. 编码空间必须**训练=部署=可视化统一**(否则 OOD)
- **现象**:可视化视频里连"真实帧解码"都很差。排查:现场 `cv2+encode_pooled` 与缓存特征库 **cos 仅 0.86**(库当初用了不同的视频解码/预处理)。→ 解码器 + prod 模型(都在库空间训)拿到 OOD 输入 → 解码烂、预测被拖累。
- **解**:**统一到唯一编码入口 `crave.encoders.encode_pooled`(DINOv3-H)+ cv2 读帧(= 部署路径)**,重编码全部帧(`reencode_pooled_unified.py`)→ 新空间 cos 自洽 1.0;`rebuild_derived.py` 保里程碑语义(与原标注 100% 匹配)重算 prototype/medoid/pairs;重训 prod + 解码器。结果:离散持平、subgoal cos 0.882→0.909、train==deploy 一致。
- **教训**:**特征缓存 = 训练契约**;任何"现场编码"必须与建库时**逐比特同路径**(同解码器/分辨率/预处理),否则静默 OOD。缓存里最好存 build 配置(编码器/reader/res)以便复现。

### B7. 本地 vePFS **96% 满**;长训练走集群/gf3
- 本地 /vePFS 仅 2.5TB free,131GB 本地缓存要谨慎;长训练不压本地 2 卡(本地只跑短任务/特征/评估/可视化)。

## C. 验证方法库(可复用)

- **kNN 上界探针**:MLP≈kNN → 判断是否触表示天花板。
- **广播-pooled 消融**:同 transformer 喂真实 grid vs 广播 pooled → 隔离"空间信息 vs 架构/参数"(H7)。
- **oracle / best-of-N**:判断多模态是否可分、生成头是否有救。
- **单变量 A/B 电池**(`local_ablations.py`):prev-latent/多头/current-latent/CVaR/λ 一次一变量。
- **诚实指标**:对真实未来评估;top-k + NLL + subgoal cos,不只 top1;跨 session 数字要 fresh 重测。
