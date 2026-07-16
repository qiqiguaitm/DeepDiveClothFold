# CLAUDE.md — deepdive_kai0 (χ₀ robotic manipulation)

**Task A**: T-shirt flatten & fold with dual-arm Agilex Piper. Built on [openpi](https://github.com/Physical-Intelligence/openpi).

## Hardware

| Machine | GPUs | Use |
|---------|------|-----|
| **sim01** (di-*) | 2×RTX 5090 32GB | Inference, IPC (cameras/CAN/ROS2) |
| **gf0**(本机) | 2×A100 80GB | 当前 dev 机(本会话所在,`/vePFS/tim/workspace/deepdive_kai0`);本地小规模训练/构建/下载。⚠️ gf1 已关闭,gf0 已从 8卡缩到 2卡 |
| **gsy** | volc 提交节点(自身无 GPU) | 北京 Robot-North-H20 队列数据同步/环境准备/任务提交入口, `ssh -p 16370 root@124.174.16.237`;训练跑在 volc 分配的 8×H20 节点。⚠️ gf3 单卡机 (:7888) 已于 2026-07 关闭 |

Source `setup_env.sh` first — auto-detects profile and sets `KAI0_DATA_ROOT`/`OPENPI_DATA_HOME`/`PYTORCH_CKPT_BASE`.

## Commands (run from `kai0/`)

```bash
# Env
GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Lint/format
uv run ruff check --fix . && uv run ruff format .

# Test
uv run pytest --strict-markers -m "not manual"

# JAX training (8+ GPU FSDP, config names in config.py)
uv run python scripts/compute_norm_states_fast.py --config-name <name>
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config> --exp_name=<name>

# PyTorch AE training (AdvantageEstimator, DDP)
uv run torchrun --standalone --nproc_per_node=8 scripts/train_pytorch.py <config> --exp_name=<name>

# Inference serve
uv run python scripts/serve_policy.py --config <config> --checkpoint <ckpt> --port 8000
```

## Architecture

- **`kai0/src/openpi/models/`** — JAX/Flax: `pi0.py`, `pi0_fast.py` (FSQ), `pi0_rtc.py` (RTC), `gemma.py`, `siglip.py`, `tokenizer.py`
- **`kai0/src/openpi/models_pytorch/`** — PyTorch AE: `pi0_pytorch.py` (value_head→tanh), patched HF transformers
- **`kai0/src/openpi/training/`** — `config.py` (central, all TrainConfigs), `data_loader.py`, `advantage_dataset.py`, checkpoints, optimizer
- **`kai0/src/openpi/policies/`** — Robot wrappers (agilex/aloha/libero), config registry in `policy_config.py`
- **`kai0/model_arithmetic/`** — Weight-space merging (6 methods, JAX+PyTorch)
- **`kai0/stage_advantage/`** — AE pipeline: GT→train→predict→discretize→AWBC. `annotation/eval.py`, `discretize_advantage.py`
- **`kai0/train_deploy_alignment/`** — Data aug (time/mirror), DAgger, temporal smoothing/ensembling

### Deployment (sim01)
`start_scripts/kai/`: `start_autonomy.sh`, `start_policy_node.sh`, `start_teleop.sh`, `start_data_collect.sh`, `test_inference_server.py --check latency|quality|all`

### Training launchers (gf0;北京队列经 gsy 提交 volc)
`train_scripts/kai/launch/`: `run_gf{0,1}.sh`, `run_*.sh`, `start_train.sh`; `kai/eval/`: `eval_awbc_compare.py`, `auto_eval.sh`; `kai/data/`: `fix_data.py`, `to_tos_file.py`; `kai/monitor/`: `check_progress.py`

### Piper tools (CAN + cameras)
`piper_tools/`: `setup_can.sh`, `diagnose_can.sh`, `test_hardware.py`, `test_cameras.py --mode ros2|direct`, `can_health_snap.sh`

## Key Configuration

- **Training configs**: `kai0/src/openpi/training/config.py` — `TrainConfig` dataclass: `repo_id`, `weight_loader`, batch/fsdp/lr/schedule/EMA
- **Policy configs**: `kai0/src/openpi/policies/policy_config.py`
- **AWBC prompts**: `"<task>. Advantage: positive/negative"` (**period** separator, not comma)
- **Hardware**: `config/pipers.yml` (CAN), `config/cameras.yml` (RealSense), `config/calibration.yml` (hand-eye)

### Data Transform Pipeline
`repack_transforms → data_transforms → Normalize(norm_stats) → model_transforms` (reversed at inference)

### Checkpoint Structure
```
checkpoints/<name>/  →  params/  assets/norm_stats.json  model.safetensors(PT)  wandb_id.txt
```

### Key Env Vars (beyond setup_env.sh auto-detect)
| Var | Default/Use |
|-----|-------------|
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `0.9` for training |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` |
| `JAX_COORDINATOR_ADDRESS:NUM_PROCESSES:PROCESS_INDEX` | Multi-node JAX |
| `CUDA_VISIBLE_DEVICES` | GPU selection |

## Data Layout (LeRobot v2.1)

`kai0/data/Task_{A,B,C}/{base,dagger}/` → `data/chunk-000/episode_*.parquet` (state[14]+action[14]), `videos/chunk-000/{top_head,hand_left,hand_right}/episode_*.mp4` (480×640 AV1 30fps), `meta/` (info.json, episodes.jsonl, tasks.jsonl)

## Downloads (di-* 本机 only)

**Bypass proxy** (1000× faster): `env -u http_proxy -u https_proxy ...` or `proxy-off`
- HF → `HF_ENDPOINT=https://hf-mirror.com` + `huggingface_hub.snapshot_download(...)` (~10 MB/s)
- pip/uv → `UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/` (6 MB/s)
- Full recipes: `docs/download_methods.md`

## Docs

- `docs/deployment/` — 部署 (strategy/training_ops/inference/data_collection/visualization/incidents). Entry: `README.md`
- `docs/training/` — 实验 (future_plans/plans/ + history/experiments/). Entry: `README.md`. 训练后归档规范见 `train_scripts/CLAUDE.md`
