# LMWM × VLA 总执行规划(2026-07-08 整合更新)

> **一句话**:先搭好 lerobot + RoboTwin + pi05 baseline 的可复现闭环环境(E0),再按"辅助监督验证有没有用(E1)→ SigLIP 原生 prefix 注入主方案(E2)→ 真预测器+集群+真机(E3)"推进 LMWM 与 π0.5 的结合。
> **整合的既有文档**(本文是总纲,细节下钻):
> - 注入机制 → [`INJECTION_DESIGN_2026-07.md`](INJECTION_DESIGN_2026-07.md)(主推=SigLIP 虚拟图像 token 进 prefix + KI)
> - world-model 架构 → [`LMWM2_FINAL_ARCHITECTURE.md`](LMWM2_FINAL_ARCHITECTURE.md)(proto teacher + prev_ẑ + 密度弃权)
> - 旧接线草案 → `archive/lmwam_v2_plan_20260704.md`(**已被 INJECTION_DESIGN 取代**:旧的"4 输入 Alternate-DiT"改为"prefix 虚拟图像 token + KI")
> - GT-first 验证铁律 → `archive/next_milestone_vla_validation_plan.md`(**sim 从 LIBERO 换成 RoboTwin**,kill criteria 保留)

---

## 0. 以终为始:最终产出

一个 **LMWAM = LMWM × kai0 π0.5** 系统,其 action expert 被 latent milestone 引导,在 **RoboTwin 2.0(sim,验方法)+ kai0 叠衣真机(验域)** 上测下游 SR,与 LaWM 98.6% 摆一起。**唯一裁决 = SR / action-MAE**(intrinsic 已达标,见 LMWM2_FINAL)。

---

## 1. 深度整合:三处对既有规划的更新

### 1a. 注入机制:旧"4 输入 Alternate-DiT" → 新"SigLIP 虚拟图像 token + KI"
`lmwam_v2_plan` 曾计划把 u/û_T/ẑ/language 四路照搬 LaWM 的 Alternate-DiT 注入。**深调研(π*0.6 + KI)推翻了它**:
- PI 自己注入新高层条件走的是 **VLM prefix 的 token 化通路**(不用 cross-attn、不用 adaRMS);
- 我们的 milestone **就在 SigLIP 空间 = PaliGemma 视觉空间** → 当"虚拟未来图像 token"经原图像投影进 prefix,近零 distribution-shift;
- 保预训练靠 **KI stop-grad**(连续 expert→backbone 的 K/V 套 sg()),不是冻 backbone。
→ 主方案 P 定于此(细节见 INJECTION_DESIGN §3)。

### 1b. sim 基准:LIBERO → RoboTwin 2.0
`next_milestone_validation` 原定 LIBERO-Long。**改用 RoboTwin 2.0**,因为:
- RoboTwin 是**双臂**操作基准,匹配 kai0 双臂叠衣本体(LIBERO 单臂,本体 gap 大);
- 本地已有全套(`/vePFS/shock/vla/RoboTwin` + conda env + X-VLA eval client),复用成本低;
- 有真机化 eval 协议(SR + 子阶段)。
**代价**:CRAVE milestone 需在 RoboTwin demo 上重挖(sim 域);故 E0 要先备好 RoboTwin demo 数据。

### 1c. GT-first 铁律:保留
先用 **GT milestone** 隔离"注入机制/子目标条件本身有没有用",再换真预测器——否则预测误差(top1 ~0.4)会污染注入方案的结论。这是 `next_milestone_validation` 的核心,继续遵守。

---

## 2. 环境资产盘点(E0 起点)

| 资产 | 位置 | 状态 |
|---|---|---|
| lerobot | kai0/.venv(0.1.0) | ✅ 已装 |
| RoboTwin 2.0 全套 | `/vePFS/shock/vla/RoboTwin`(assets/envs/policy/eval) | ✅ 可复用(需确认版本) |
| RoboTwin conda env | `/vePFS/HuanQian/conda_envs/RoboTwin`(py3.10) | ✅ 可复用 |
| RoboTwin eval client | `X-VLA/evaluation/robotwin-2.0`(tim 自己的) | ✅ |
| pi05 base ckpt | `kai0/checkpoints/pi05_base` | ✅ 本地 |
| pi05 fold/awbc 变体 | `kai0/checkpoints/pi05_flatten_fold_*` | ✅ |
| openpi 训练/推理 | `kai0/scripts/train_pytorch.py` | ✅ |
| LMWM provider | `lmwm/checkpoints/teach_proto_3task.pt` | ✅ |

**关键未知(E0 要解决)**:① RoboTwin 用哪个 python env(自带 conda vs kai0venv,SAPIEN/vulkan 依赖);② pi05 在 RoboTwin 上的推理 client 接口(obs 格式/action 归一化);③ RoboTwin demo 数据是否已采(供 CRAVE 挖矿)。

---

## 3. 阶段计划(E0→E3,每步 kill criteria)

### E0 · 环境搭通 + pi05 baseline 复现(**当前阶段**)
> 目标:一条命令能在 RoboTwin 上跑 pi05 rollout 出 SR。这是后续一切注入实验的地基。

| 步 | 做法 | 判据 |
|---|---|---|
| E0.1 | 摸清 `/vePFS/shock/vla/RoboTwin` 版本 + env(SAPIEN/vulkan 渲染);跑通官方 demo task 一条 rollout | 仿真能出图、能 step |
| E0.2 | 接 pi05_base 到 RoboTwin eval client(复用 X-VLA client 或 shock policy 接口);单任务闭环 | pi05 在 1 个 RoboTwin 任务上跑完 rollout 出 SR 数 |
| E0.3 | pi06 若要:确认 π*0.6/π0.6 ckpt 是否可得(PI 未开源→大概率跳过,以 pi05 为基线) | 有则接,无则记录、以 pi05 为准 |
| E0.4 | RoboTwin demo 数据落地(采集或复用 shock/data)→ 供 E2 CRAVE 挖矿 | demo parquet/hdf5 就绪 |

**kill criteria**:E0.2 跑不通(env/接口)→ 退回本地 kai0 离线 action-MAE 作首级 eval(不阻塞 E1)。

### E1 · 辅助监督验证(INJECTION_DESIGN 备选 F,最省,GT milestone)
> 一天出信号:milestone 条件对策略到底有没有正增量。不改推理前向。

- 加 K≈4-8 个可学习 milestone query token + FLARE 式辅助对齐损失(对齐 GT milestone pooled 嵌入),LoRA action expert,本地 2 卡短跑;
- eval:offline action-MAE(有辅助 vs 无);
- **kill criteria**:≈0 → milestone 对 π0.5+语言无信息增量,诚实收口(转纯 world-model 论文,不接 VLA);>0 → 进 E2。

### E2 · 主方案 P 接线(SigLIP 虚拟图像 token + KI)
> INJECTION_DESIGN 主推 P 的完整实现。

- milestone grid(GT 先行)→ SigLIP 原图像投影 → 池化到 64 token → 摆 prefix 的 language 之后 + type-emb + KI stop-grad + CFG dropout;
- 新 `Pi0Config` 变体 + 改 `pi0_pytorch.py embed_prefix`;LoRA action expert;
- eval 阶梯:kai0 离线 action-MAE(天级)→ RoboTwin SR(GT milestone);
- **漂移检查**:测语言跟随/原任务是否退化(KI 是否顶住)→ 退化则切备选 A(零初始化门控 cross-attn);
- **kill criteria**:+milestone(GT)不超 base → 注入无效,回 E1 结论收口。

### E3 · 真预测器 + 集群 + 真机
- GT milestone → 真 LMWM provider(teach_proto + prev_ẑ 自递归 + 密度弃权);量化预测误差吃掉多少 GT 增量;
- 集群提交微调(submit-training-job,不本地长训);
- RoboTwin SR(sim 验方法)→ kai0 叠衣真机(验域)→ 对齐 LaWM 98.6%;
- **milestone-horizon vs 固定 1.2s** 消融(论文生态位主张,LMWM2_FINAL §8)。

---

## 4. 双轨原则(sim 验方法 / 真机验域)

- **sim 轨(RoboTwin)**:快迭代、验注入方法/机制有效性;CRAVE 在 RoboTwin demo 上挖 milestone;
- **真机轨(kai0 叠衣)**:验域迁移;用现有 kai0 π0.5 + kai0 CRAVE milestone;
- 两轨解耦:方法在 sim 上定,域迁移在真机上验。首选 sim 因为不占真机、迭代快。

---

## 5. 风险与止损

| 风险 | 概率 | 止损 |
|---|---|---|
| RoboTwin env(SAPIEN/vulkan)本地配不通 | 中 | 复用 shock conda env;实在不行退 kai0 离线 action-MAE 作 eval |
| pi06 ckpt 不可得 | 高(PI 未开源) | 以 pi05 为基线,记录 |
| milestone 对策略无增量(E1 kill) | 中 | 长程任务测;诚实收口为纯 WM 论文 |
| prefix 注入掉语言跟随(KI 没顶住) | 中 | 切备选 A(零初始化 cross-attn) |
| RoboTwin milestone bank 质量差 | 中 | bank 验收三关卡(LMWM2_FINAL §6)前置 |
| sim→真机 gap | 高 | 双轨解耦,sim 验方法真机验域 |

---

## 6. 下一步(立即)
E0.2:接 pi05_base 到 RoboTwin eval client(client-server),单任务闭环出 SR。

---

## 附录 A · RoboTwin 环境配置(E0.1 实测,可复现)

**版本**:RoboTwin 2.0(arXiv 2506.18088),双臂 aloha-agilex。
**conda env**:`/vePFS/HuanQian/conda_envs/RoboTwin/bin/python`(py3.10 · sapien 3.0.0b1 · torch 2.4.1+cu121 · curobo/mplib)。
**可写工作副本**:`/home/tim/workspace/RoboTwin`(代码 rsync 自 `/vePFS/shock/vla/RoboTwin`,`assets` 软链 16G 原目录,root-owned 原目录不可写)。

**三个坑 + 修复(必设环境变量)**:
```bash
# ① SAPIEN 无头 GPU 渲染:必须指向 sapien 自带的 nvidia_icd(系统 vulkan ICD 无 nvidia)
export VK_ICD_FILENAMES=/vePFS/HuanQian/conda_envs/RoboTwin/lib/python3.10/site-packages/sapien/vulkan_library/nvidia_icd.json
# ② curobo CUDA kernel JIT:CUDA_HOME 必设(空则 torch cpp_extension 找不到 nvcc 挂死),限 A100 架构编译更快
export CUDA_HOME=/usr/local/cuda           # cuda-12.8, 对 torch cu121 兼容
export TORCH_CUDA_ARCH_LIST="8.0"          # A100 sm_80
export PATH=/usr/local/cuda/bin:$PATH
# ③ curobo JIT stale 锁:若曾挂死中断,必须清缓存否则新 run 死等文件锁(低CPU/无nvcc/无报错的假象)
rm -rf ~/.cache/torch_extensions/py310_cu121
# ④ warp 1.13 API 迁移:curobo 用旧 `wp.torch.xxx`,warp 1.13 移到顶层 `wp.xxx`
#    warp/curobo 目录 root-only 不可改 → sitecustomize 运行时 shim(_shim/sitecustomize.py 别名 wp.torch)
export PYTHONPATH=/home/tim/workspace/RoboTwin/_shim:$PYTHONPATH   # 内含 sitecustomize.py:wp.torch=SimpleNamespace(...)
```
- curobo 首次编译 ~10min(ptxas 高 CPU),编完缓存到 `~/.cache/torch_extensions/py310_cu121`,后续 rollout 免编。
- 备用:HuanQian 已编好的 `.so` 在 `/vePFS/HuanQian/RoboTwin/envs/curobo/src/curobo/curobolib/`。

**跑一条 demo rollout(scripted expert 采集)**:
```bash
cd /home/tim/workspace/RoboTwin
python script/update_embodiment_config_path.py     # 首次:修 embodiment 路径
# 建小配置:sed 's/episode_num: 50/episode_num: 1/' task_config/demo_clean.yml > task_config/tim_smoke1.yml
python script/collect_data.py <task_name> tim_smoke1   # 数据落 data/<task>/tim_smoke1/
```
**eval(client-server)**:`policy_model_server.py`(policy 推理服务)+ RoboTwin client(`X-VLA/evaluation/robotwin-2.0/client.py` 或 `script/eval_policy_client.py`);tim 参考 `X-VLA/evaluation/robotwin-2.0/eval_robotwin.sh`。

---

## 附录 B · pi0 × RoboTwin eval 跑通流程(E0.2 实测,2026-07-09)

**架构**:client-server。server(JAX openpi,pi05/.venv,双 GPU)加载 ckpt 服务端口;client(RoboTwin conda env,sim)每步查 server 拿 action。**自包含脚本** `RoboTwin/run_eval_pi0.sh`(起server→等load→跑client→收尾)避免 server 被 session teardown 单独杀。

**ckpt**:社区 RoboTwin-finetuned openpi ckpt(HF)。用 `cjgogo/pi0-aloha-robotwin-lora_<task>`(每任务 LoRA)。放 `RoboTwin/policy/pi0/checkpoints/pi0_base_aloha_robotwin_lora/<task>/30000/`(params/ + assets/demo_clean/norm_stats.json)。**下载走 hf-mirror+huggingface_hub**(见 `docs/download_methods.md`);⚠️ hf-mirror 偶发缓存坏 blob(size 对但 zstd 解压失败)→ `force_download=True` 重下修复。

**RoboTwin 侧必打的补丁**(shock 的 vendored 副本 bug + client-server 未适配):
1. `pi_model.py`:缺 `import os` → 加。
2. `policy_config.py:62`:`data_config.asset_id = ...` 对 frozen dataclass 赋值 → 改 `dataclasses.replace(data_config, asset_id=...)`。
3. `pi_model.py` PI0:缺 `reset_model` 方法(client 调 'reset_model' 但 PI0 只有 `reset_obsrvationwindows`)→ 加别名。
4. `deploy_policy.py`:顶层 `import dill` + `from pi_model import *`(需 jax)→ 改懒加载(移进 get_model),client 侧无需 jax 即可 import。
5. `deploy_policy.eval()`:原写法直接访问 `model.observation_window`/多参数方法(给进程内 PI0 写的),ModelClient 无 `__getattr__` → 重写全用 `model.call(func_name=..., obs=...)`;多参数用 `obs=[a,b]` list。
6. `policy_model_server.py` 分发:`method(obs)` 单参 → 加 `if isinstance(obs,(list,tuple)): method(*obs)` 支持多参数方法(update_observation_window)。
7. `eval_policy_client.py`:`test_num=100` 硬编码 → 改 `usr_args.get("test_num",100)` 可配。

**跑法**:`bash RoboTwin/run_eval_pi0.sh <task> <test_num>`(env vars 见附录 A + `HF_ENDPOINT`/`XLA_PYTHON_CLIENT_MEM_FRACTION`/双 GPU;ckpt 2 分片保存 → server 必须 `CUDA_VISIBLE_DEVICES=0,1`)。

**E0.2 结果**:beat_block_hammer 闭环跑通(step 1→400 逐步执行 + 算 SR),harness 端到端验证 ✅。首轮 SR≈0%(cjgogo LoRA + `instruction_type=unseen`;偏低疑因 unseen 指令/obs 相机映射,属质量调优,不阻塞 harness 验证)。

**⚠️ harness 忠实性验证(关键)**:首轮 `instruction_type=unseen` 得 0/5;改 `seen`(匹配训练指令)后 episode 2 即成功(1/2=50%)。→ **harness 忠实**(策略真工作时正确记成功),0% 是 unseen 指令配置(模型没在没见过的语言上训过),非 harness bug。**baseline eval 用 `instruction_type=seen`**(匹配训练);unseen 是更难的泛化 eval。跑法:`bash run_eval_pi0.sh <task> <test_num> seen`。
