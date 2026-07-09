# RoboTwin 2.0 仿真评测环境 — 落地配置 & 使用手册

> 双机已验证(2026-07-09):**本地 di-\*(2×A100)** + **gf3(8×H20)**。
> 用于 pi0/pi05/RLinf checkpoint 在 RoboTwin 2.0 双臂 aloha-agilex 仿真里跑 eval(client-server 架构)。
> 本文覆盖:① 高层配置路径 ② 使用方法(一条命令跑 eval)③ 全部踩坑与注意事项。

---

## 0. TL;DR — 直接跑

```bash
# 本地 di-*
cd /home/tim/workspace/RoboTwin
bash run_eval_pi0.sh <task> <test_num> <instruction_type> <train_config_name>
# 例: RLinf 官方 pi0-SFT 对比 adjust_bottle
bash run_eval_pi0.sh adjust_bottle 25 seen pi0_base_aloha_robotwin_full

# gf3 (ssh root@124.174.16.237:7888, 密码 tim)
cd /vePFS-North-E/vis_robot/tim/RoboTwin
bash run_eval_pi0_gf3.sh beat_block_hammer 5 seen pi0_base_aloha_robotwin_lora
```

**已验证结果**:
| 机器 | ckpt | task | SR | 对照 |
|---|---|---|---|---|
| 本地 | RLinf 官方 pi0-SFT | adjust_bottle | **22/25=88.0%** | 官方 76.56% ✓一致略高 |
| 本地 | cjgogo LoRA | beat_block_hammer | 7/20=35% | 环境验证 |
| gf3 | cjgogo LoRA | beat_block_hammer | 3/5=60% | 环境验证 |

---

## 1. 架构:为什么是 client-server 两个 Python 环境

RoboTwin eval **必须拆成两个隔离的 Python 环境**,靠 socket 通信:

```
┌─ SERVER (openpi 推理) ────────┐        ┌─ CLIENT (RoboTwin sim) ──────────┐
│ policy_model_server.py        │◄─8012─►│ eval_policy_client.py            │
│ env: policy/pi05/.venv (uv)   │ socket │ env: huanqian conda RoboTwin     │
│ jax0.5.0 / torch / openpi     │        │ sapien3.0.0b1 + curobo + mplib   │
│ 加载 pi0/pi05 ckpt, 出 action │        │ 跑物理仿真, 打分 SR              │
└───────────────────────────────┘        └──────────────────────────────────┘
```

- **为什么分开**:openpi(JAX/新 torch)和 RoboTwin sim(sapien/curobo/老 torch)依赖互斥,同一 env 装不下。
- server 出 50 步 action chunk(`PI0_STEP=50`),client 执行并回传 obs。
- `run_eval_pi0*.sh` 是自包含脚本:起 server→等加载→跑 client→收 SR→杀 server。

---

## 2. 高层配置路径(两机对照)

| 组件 | 本地 di-\* | gf3(H20) |
|---|---|---|
| **代码根** | `/home/tim/workspace/RoboTwin` | `/vePFS-North-E/vis_robot/tim/RoboTwin` |
| **SERVER env** | `policy/pi05/.venv`(uv sync) | 同左(uv sync, cache 走 vePFS) |
| **CLIENT env** | `/vePFS/HuanQian/conda_envs/RoboTwin` | `/vePFS-North-E/vis_robot/huanqian/conda_envs/RoboTwin` |
| **curobo** | huanqian 自带(sm_80 OK) | ⚠️ tim 副本重编 sm_90(见 §5.3) |
| **eval 脚本** | `run_eval_pi0.sh` | `run_eval_pi0_gf3.sh` |
| **GPU** | 2×A100 sm_80 | 8×H20 sm_90 |

### 2.1 SERVER env 搭建(uv)
```bash
cd <repo>/policy/pi05
# 关键环境变量(gf3 尤其必须, 避 overlay 满 + 用阿里云 pip 镜像)
export UV_CACHE_DIR=<vePFS>/.uv_cache TMPDIR=<vePFS>/.tmp
export UV_HTTP_TIMEOUT=600 UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
unset ALL_PROXY all_proxy http_proxy https_proxy   # 清代理(见 download_methods.md)
uv sync                                             # 装 302 包: jax0.5.0+torch+openpi+lerobot
```

### 2.2 CLIENT env
用 huanqian 现成 conda RoboTwin(sapien3.0.0b1 + curobo + mplib + RoboTwin 任务代码)。**不要动 huanqian 共享目录**;需改的东西(curobo sm_90)放 tim 自己副本,靠 PYTHONPATH 前置覆盖。

### 2.3 checkpoint 布局
```
policy/pi0/checkpoints/<train_config_name>/<model_name>/<checkpoint_id>/
  ├── params/ 或 model.safetensors   # JAX=params目录 / PyTorch=单文件
  └── assets/<asset_id>/norm_stats.json
```
- **JAX ckpt**(cjgogo LoRA):`params/` 目录, server 走 JAX 路径。
- **PyTorch ckpt**(RLinf 官方):合并分片成单 `model.safetensors`(见 `setup_rlinf_ckpt.py`),server 走 `load_pytorch`。⚠️ RLinf norm_stats 在 repo 根 `physical-intelligence/robotwin/norm_stats.json`,**不在** assets/ 下,须手动 cp 到 `assets/robotwin/norm_stats.json`。

---

## 3. 使用方法

### 3.1 参数
`bash run_eval_pi0*.sh <task> <test_num> <instruction_type> <train_config_name>`
- `task`:RoboTwin 任务名(adjust_bottle / beat_block_hammer / ...)。
- `test_num`:评测 episode 数(不稳定场景 seed 会被跳过,不计入分母)。
- `instruction_type`:**`seen`**(与训练一致)/ `unseen`。⚠️ 用 `unseen` 常 0% 不是 bug,是指令分布 mismatch;对比官方数必须 `seen`。
- `train_config_name`:如 `pi0_base_aloha_robotwin_lora`(cjgogo) / `pi0_base_aloha_robotwin_full`(RLinf)。

### 3.2 输出
- SR 逐 episode 打印:`Success rate: 22/25 => 88.0%, current seed: ...`。
- 脚本尾部 `CLIENT_RC=0` = 干净完成。
- server 日志 `srv_eval.log`;client 日志见你 redirect 的文件。

### 3.3 对比官方数据的正确姿势
- `instruction_type=seen` + 对应 `train_config_name` + 正确 `checkpoint_id`。
- test_num 越大越准(25 ep 有 seed 方差);官方数通常 100+ ep。我们 88% vs 官方 76.56% 属一致量级。

---

## 4. 关键补丁清单(已应用, 换机/重装需重打)

RoboTwin openpi fork 原始代码跑不通,以下补丁已应用(本地+gf3):

| 文件 | 改动 | 原因 |
|---|---|---|
| `policy/pi0(+pi05)/pi_model.py` | 加 `import os`;加 `reset_model()` 别名 | 缺 import;命名不一致 |
| `policy/pi0(+pi05)/deploy_policy.py` | 删顶层 `import dill`;`from pi_model import *` 改懒加载;eval 用 `model.call(func_name=...)` | client 不该 import jax/dill;ModelClient 无 `__getattr__` |
| `policy/pi05/src/openpi/policies/policy_config.py` | `dataclasses.replace(data_config, asset_id=...)` | frozen dataclass |
| `policy/pi05/src/openpi/models/model.py:246` | `load_model(..., strict=False)` | RLinf ckpt 多 value_head/embed_tokens key |
| `policy/pi05/src/openpi/models_pytorch/pi0_pytorch.py:112` | `torch.compile(max-autotune)` 加 `OPENPI_DISABLE_COMPILE` 开关(默认禁) | **见 §5.1** |
| `script/policy_model_server.py` | dispatch `if isinstance(obs,(list,tuple)): method(*obs)` | 参数解包 |
| `script/eval_policy_client.py:134` | ModelClient `timeout=30→180` | PyTorch 推理慢, 防误判超时 |
| `policy/pi05/pyproject.toml` | `[tool.uv] override-dependencies=["av==14.2.0"]` | av14.4.0 无 wheel |
| `_shim/sitecustomize.py` | warp1.13 `wp.torch` alias | RoboTwin 用旧 warp API |

---

## 5. 注意事项 / 踩坑(重要!)

### 5.1 ⚠️ PyTorch pi0/RLinf eval 首推理静默 hang(GPU 0%)
`pi0_pytorch.py` 的 `torch.compile(mode="max-autotune")` 首次调用做 kernel 自动调优**数分钟**(编译期 GPU 0%),撞 socket 超时,报 `ConnectionError: Communication error: timed out`,**无任何 error/traceback**。
- **修**:`OPENPI_DISABLE_COMPILE=1`(现默认禁)→ get_action **1.31s**。`run_eval_pi0*.sh` server 端已带此 env。
- **诊断套路**:eval hang 先看 GPU util——**0%+显存占着=非算力问题**(编译/CPU/死锁),不是慢。写独立 probe 直调 get_action 隔离 socket。
- JAX ckpt(cjgogo)不涉及 torch.compile,不受影响。

### 5.2 ⚠️ RoboTwin "Objects is unstable in seed(...)" 是正常的
某些 seed 物体(如 001_bottle)物理settle 失败→该 seed 跳过、**不计入 SR 分母**(所以看到 seed 跳号)。不是 bug,不用管。

### 5.3 ⚠️ gf3(H20/sm_90)curobo "no kernel image"
gf3 是 H20(Hopper sm_90)。huanqian 的 curobo `.so` 是 A100(sm_80)编的,H20 上报 `RuntimeError: CUDA error: no kernel image is available`,**且被 eval loop 捕获后伪装成下游 `AttributeError: 'Robot' object has no attribute 'left_planner'`**(真错在 episode1 的 CuroboPlanner init,别被 left_planner 误导)。
- **修**(不动 huanqian 共享目录):为 sm_90 重编 tim 自己的 curobo 副本:
```bash
cd /vePFS-North-E/vis_robot/tim/RoboTwin/envs/curobo
rm -f src/curobo/curobolib/*.so; rm -rf build
CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
  TORCH_CUDA_ARCH_LIST="9.0" SETUPTOOLS_SCM_PRETEND_VERSION=0.7.6 \
  /vePFS-North-E/vis_robot/huanqian/conda_envs/RoboTwin/bin/python setup.py build_ext --inplace
```
  - `SETUPTOOLS_SCM_PRETEND_VERSION` 必设(tim 副本无 .git→setuptools-scm 报 LookupError)。
  - 用 huanqian python 编以匹配 torch ABI(nvcc12.8 编 torch cu121 OK,同大版本)。
- client PYTHONPATH **前置** tim curobo 覆盖 huanqian editable:
  `PYTHONPATH=<tim>/envs/curobo/src:<tim>/_shim:$PYTHONPATH`(已写进 `run_eval_pi0_gf3.sh`)。

### 5.4 ⚠️ SAPIEN headless 渲染找不到 Vulkan ICD
`VK_ICD_FILENAMES=<client_env>/lib/python3.10/site-packages/sapien/vulkan_library/nvidia_icd.json`(脚本已带,换机改路径)。

### 5.5 ⚠️ 端口 8012 "Address already in use"
上次 eval 的 server 没杀干净。重跑前:
`for p in $(ps aux|grep '[p]olicy_model_server'|awk '{print $2}'); do kill -9 $p; done; fuser -k 8012/tcp`
(脚本开头有清理,但异常退出会残留)。

### 5.6 ⚠️ 2-GPU ckpt 分片
cjgogo LoRA 在本地 2 卡存的,server 须 `CUDA_VISIBLE_DEVICES=0,1`。单卡 ckpt 用 `=0`。

### 5.7 ⚠️ curobo 首次 JIT 编译慢
client 首跑 curobo 会 JIT 编译 kinematics/geom/lbfgs kernel(几分钟,日志 "not found, JIT compiling")。正常,等它。之前需 `CUDA_HOME=/usr/local/cuda` + 清 `~/.cache/torch_extensions` �just in case。

### 5.8 ⚠️ 下载(HF ckpt / pip / lerobot git)
- HF 模型/pip:走域内镜像不走代理,见 `docs/download_methods.md`(hf-mirror + aliyun)。
- RLinf ckpt 是 Xet CAS 后端,hf-mirror 不支持→须走代理 7890,首下常损坏(MetadataIncompleteBuffer)重下即可。
- **lerobot git dep** 卡 github early EOF(gf3 尤甚):从本地 uv 缓存 rsync `~/.cache/uv/git-v0/db|checkouts/b2400a7a62d6a7cf`(hash 由 URL 决定,两机一致)到 gf3 `<vePFS>/.uv_cache/git-v0/`,uv sync 就跳过 github fetch。

---

## 6. 相关文档 / memory
- `docs/download_methods.md` — 下载方法(域内镜像绕代理)
- `lmwm/docs/MASTER_PLAN_lmwm_vla_2026-07.md` 附录 A/B — RoboTwin env 4 修 + pi0×RoboTwin 7 补丁
- memory: `reference_robotwin_pi0_pytorch_compile_hang`、`reference_gf3_robotwin_curobo_sm90`、`reference_local_download_bypass_proxy`
- gf3 凭据/布局:`docs/deployment/training_ops/ssh_and_credentials.md`
