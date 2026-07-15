# 跑通 LaWAM 流程 + 用 kai0 自有数据训练一个 VLA — 研究与规划(2026-07-12)

> 目标:先把**外部开源 LaWAM**(RLinf,arXiv 2606.15768)全流程跑通,再用**我们自己的 kai0 叠衣数据集**SFT 出一个可用的 latent-subgoal VLA。作为后续"milestone 注入"的可用底座 + 参考实现。
> ⚠️ 命名:此处 **LaWAM = 外部参考工作**(github.com/RLinf/LaWAM),**不是**我们的伞形项目 LMVLA(=LAWAM 别名,见 [[project_lawam_lmvla_alias]])。二者是"上游孪生"关系:我们的 `lmwm` 就是它的重实现。

---

## 1. LaWAM 是什么(架构)

**一句话**:LaWAM 预测**冻结视觉特征空间里的未来观测特征**,把它当作 **latent visual subgoal** 注入策略来生成动作。两阶段:

```
① LAM / 世界模型(latent_action_model/):
     DINOv3-ViTB16(冻结) 编码观测 → VAE 学 latent-action / 预测未来特征
② LaWAM policy(starVLA/, 基座):
     Qwen3-VL-2B-Instruct(VLM) + action model
     LAM 出的 latent subgoal 注入 → 生成动作
     训练: 从 lawam_pretrain 初始化 → benchmark SFT
```

- **基座 VLM**:Qwen3-VL-2B-Instruct(注意:**不是 π0.5/PaliGemma**——见 §5 战略点)。
- **代码基**:starVLA(MIT)+ GR00T 式 LeRobot dataloader。
- **和我们 lmwm 的关系**:我们的 LMWM(预测器+生成器,SigLIP 空间)≈ LaWAM 的 LAM;我们此前的注入设计针对 π0.5,LaWAM 针对 starVLA/Qwen3-VL。

---

## 2. 开源程度:全套 released(极高)

| 类型 | 资源(HF/GitHub) | 本地状态 |
|---|---|---|
| **代码** | github.com/RLinf/LaWAM(项目页 rlinf.github.io/LaWAM) | ✅ 已 vendored `lmvla/lmwm/vendor/LaWAM/`(有 .git,可 pull) |
| 基座 VLM | Qwen/Qwen3-VL-2B-Instruct | ❌ 待下(→`results/Checkpoints/qwen3_weights`) |
| LAM 视觉编码器 | facebook/dinov3-vitb16-pretrain-lvd1689m | ❌ 待下(→`weights/dinov3-vitb16-...`) |
| **LAM ckpt** | jialei02/lawam_lam | ✅ **已下** `vendor/LaWAM/ckpts_dl/`(2.8G, `dino_large_vae.yaml`+`checkpoints/pytorch_model.pt`) |
| **SFT 初始化 ckpt** | jialei02/lawam_pretrain | ❌ 待下(→`results/Checkpoints/pretrain/lawam_pretrain`) |
| LIBERO SFT ckpt | jialei02/lawam_libero_sft_release | ❌ 待下(可选,做 sanity) |
| RoboTwin SFT ckpt | jialei02/lawam_robotwin_sft_release | ❌ 待下(可选) |
| LIBERO 数据集 | jialei02/libero_merged_no_noops_20hz(LeRobot 3.0) | ❌ 待下(可选,验流程) |
| RoboTwin 数据集 | jialei02/robotwin_merged(LeRobot 3.0, EEF) | ❌ 待下(可选) |

**结论**:代码 + LAM + pretrain + 两个 benchmark 的 SFT ckpt/数据集**全公开**。我们只需下 Qwen3-VL-2B + DINOv3 + lawam_pretrain 即可训练;LAM 已在本地。→ **复现门槛很低**。

---

## 3. 我们自己的数据集 vs LaWAM 期望

| 维度 | kai0 叠衣(我们) | LaWAM 期望 | Gap / 动作 |
|---|---|---|---|
| 格式 | LeRobot **v2.1** | LeRobot **3.0**(dataloader=GR00T 式) | ⚠️ **需 v2.1→3.0 转换**(主要工作量) |
| 本体/动作 | agilex 双臂 **14 维关节** | LIBERO 各自 / RoboTwin **EEF** | 需注册新 data_mix + 动作空间 + norm stats |
| 相机 | top_head / hand_left / hand_right(3) | starVLA 视图约定 | 需映射相机键 |
| fps | 30 | libero 20hz / robotwin 30hz | 与 robotwin 一致,较省事 |
| 规模 | Task_A 793G / 110k parquet(多子集:kai0_base/dagger/advantage/vis_*) | — | 首训**取子集**(如 kai0_base 或 flatten-fold 一档),别全量 |

data_mix 注册位置:`starVLA/dataloader/gr00t_lerobot/mixtures.py` + `datasets.py` + `lerobot_datasets.py`。

---

## 4. 分阶段规划(每阶段 kill criteria + 算力落点)

> 算力:本地 2×A100-80G(空闲)· gf3 8×H20(现被 GigaWorld 占满,见 [[project_lmvla_compute_env]])。下载走 hf-mirror(见 download_methods,别用代理)。

### P0 · 环境 + 复现 sanity(不碰我们数据,先证 pipeline 通)
1. `git -C vendor/LaWAM pull` 对齐最新(注意 vendored 配置是 `train_{libero,robotwin}.yaml`,README 提的是 `starvla_train_*`——**先核对配置名差异**)。
2. 建 `lawam` conda(py3.10 + requirements.txt + flash-attn 2.8.3 + `pip install -e .`);跑 README 的 import check。
3. 下 Qwen3-VL-2B + DINOv3 + lawam_pretrain(+ 可选 libero SFT ckpt/数据)。
4. **最便宜 sanity**:用**已发布的 LIBERO SFT ckpt** 跑一次 LIBERO 推理出 SR。
   - kill:装不上/import 失败/SR≈0 → 先修环境,不进 P1。
   - 算力:本地 2×A100 足够(2B VLM 推理)。
   - ⚠️ LIBERO/RoboTwin 模拟器是**另一套 env**(README 明示),sanity 可只做 LIBERO(轻),RoboTwin 重(且我们已有 RoboTwin env,见 robotwin_sim_env_setup.md,可复用)。

### P1 · kai0 数据接入(v2.1→3.0 + 注册 data_mix)
1. 选**首训子集**(建议 `kai0_base` 单档,先小)。
2. **LeRobot v2.1 → 3.0 转换**:先查 GR00T dataloader 到底要不要 3.0(可能兼容 2.1),不兼容再写转换(LeRobot 官方有 2.1→3.0 脚本)。
3. 在 `mixtures.py` 注册 `data_mix: kai0_fold`:相机键映射(top_head/hand_left/hand_right)、动作维(14 关节)、proprio、算 norm stats。
4. **最小 dataloader 冒烟**:能取一个 batch、shape/norm 对 → 过。
   - kill:转换后 dataloader 取不出正确 batch → 停在数据层,别急着训。
   - 算力:CPU/本地。

### P2 · kai0 SFT(训练我们的 VLA)
1. 复制 `train_robotwin.yaml`(动作空间最接近双臂)→ `train_kai0.yaml`,改 `data_mix: kai0_fold` + Qwen3-VL/LAM 路径 + 从 lawam_pretrain 初始化。
2. `bash train_lawam.sh --config_yaml train_kai0.yaml --run_id kai0_fold_sft`(单机);多卡用 `train_lawam_distributed.sh`。
3. 先**短跑**(小 step)验 loss 下降 + 不崩;再拉正式训练。
   - kill:loss 不降/NaN/显存爆 → 调 bs/精度/config。
   - 算力:本地 2×A100 先短跑冒烟;正式训练等 gf3 空卡(8×H20)或本地长跑。

### P3 · 评测我们的 VLA
- offline:val action-MAE。online:RoboTwin(已有 env)或真机叠衣。
- 对标:LaWAM released ckpt 在其 benchmark 的数;我们 kai0 SFT 在 kai0 域的 SR/MAE。

### P4(后续,非本次)· 接我们的 milestone 注入
- LaWAM 的 LAM latent subgoal ↔ 我们 LMWM 的 milestone;把 [`INJECTION_DEEP_ANALYSIS`](INJECTION_DEEP_ANALYSIS_latent_milestone_2026-07-10.md) 的 T0/T1/T2 在 starVLA 上落地(比 π0.5 更易,因 LaWAM 本就是 latent-subgoal 架构)。

---

## 5. 一个战略点需要你拍板(影响 P2 之后)

**LaWAM 基座是 Qwen3-VL-2B + starVLA,不是我们之前定的 π0.5(PaliGemma)。** 两条路:

| 选项 | 含义 | 优劣 |
|---|---|---|
| **A. 以 LaWAM/starVLA 为底座**(推荐先做) | 直接在开源 LaWAM 上 SFT kai0,后续 milestone 注入也落 starVLA | ✅ 全套开源、最快出一个能跑的 latent-subgoal VLA;✅ 架构天生支持 subgoal 注入。✗ 换了基座(非 π0.5) |
| **B. 只把 LaWAM 当参考,底座仍 π0.5** | 读 LaWAM 实现学 LAM/注入,回 π0.5 自己接 | ✗ 慢、要自己实现注入;✅ 保持 π0.5 主线(kai0 已有大量 π0.5 资产) |

**我的建议**:P0–P3 先走 A(用 LaWAM 快速拿到一个 kai0-VLA 底座 + 打通全流程),同时把学到的注入经验用于 π0.5(B 作为并行/后续)。是否同意以 A 为主线?

---

## 6. 立即可做(不等拍板、不占 gf3)
- P0.1 `git pull` vendored LaWAM + 核对配置 + 建 lawam conda env。
- P0.3 下载 Qwen3-VL-2B / DINOv3 / lawam_pretrain(hf-mirror,几十 GB,后台挂着下)。
- P1.1 选定 kai0 首训子集 + 查 GR00T dataloader 对 LeRobot 版本的兼容性。

## 7. 主要风险
1. **LeRobot 2.1→3.0 转换**可能有坑(视频/meta schema 变化)——P1 最大不确定性。
2. flash-attn 2.8.3 与本地 CUDA 版本匹配(README 已警示,可能要手装 wheel)。
3. Qwen3-VL-2B 下载体积 + hf-mirror 速度(用后台 + aria2c)。
4. 14 维关节动作空间的 norm/tokenize 是否被 starVLA 动作头直接支持(RoboTwin 用 EEF,不同)。
5. gf3 被 GigaWorld 占满 → 正式训练排期不确定;本地 2 卡可先冒烟但正式训练偏慢。

## 8. 引用
- LaWAM arXiv 2606.15768 · github.com/RLinf/LaWAM · vendored `lmvla/lmwm/vendor/LaWAM/`
- 本地环境 [[project_lmvla_compute_env]] · 注入分析 [`INJECTION_DEEP_ANALYSIS_latent_milestone_2026-07-10.md`](INJECTION_DEEP_ANALYSIS_latent_milestone_2026-07-10.md) · 下载法 `docs/download_methods.md`
