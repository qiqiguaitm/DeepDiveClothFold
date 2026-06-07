# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This is the public **NVIDIA Cosmos** repo (`github.com/NVIDIA/cosmos`) — documentation and
runnable cookbooks for the **Cosmos 3** model family. It is **not** a Python package: there is no
`pyproject.toml`/`setup.py`, no installable module, and no test suite. The tracked content is
entirely Markdown docs (`README.md`, `RELEASE.md`, `inference_benchmarks.md`, `CONTRIBUTING.md`)
and the `cookbooks/cosmos3/` tree (Jupyter notebooks + per-cookbook READMEs + small media/JSON
assets). The model weights and the inference/training code live in **other** repos that the
cookbooks install (see "Backends" below).

"Development" here means editing the docs and notebooks and verifying that the cookbooks still run
against one of the backends. There is no lint/build/CI to run; validation is running the relevant
notebook end-to-end (per `CONTRIBUTING.md`: "Verify that existing cookbooks and examples still work").

> Note: the working directory also contains many **untracked** sibling folders (`dreamzero/`,
> `giga-world-policy/`, `datasets/`, `models/`, `miniconda3/`, conda/uv caches, etc.) that are
> separate workstreams and scratch data, not part of this git repo. `git ls-files` shows the real
> repo contents. Don't assume those folders are part of Cosmos.

## The big picture: two runtime surfaces, one model

Cosmos 3 is an omnimodal world model on a unified **Mixture-of-Transformers (MoT)** architecture
that exposes two surfaces from the *same* checkpoint:

- **Reasoner** — autoregressive transformer, inputs text+vision, **outputs text**. World
  understanding, grounding, planning, action forecasting. Architecture class
  `Cosmos3ReasonerForConditionalGeneration`.
- **Generator** — diffusion transformer, inputs text/vision/sound/action, **outputs
  vision/sound/action**. World generation/simulation, forward & inverse dynamics, policy.
  Diffusers pipeline `Cosmos3OmniPipeline`; served class `Cosmos3OmniDiffusersPipeline`.

The cookbook tree mirrors these surfaces:
`cookbooks/cosmos3/reasoner/` and `cookbooks/cosmos3/generator/{audiovisual,action}/`. Each
leaf has `run_*_with_<backend>.ipynb` notebooks; the file suffix is the backend it targets.

**Model family** (HF `nvidia/…`, mirrored on ModelScope `nv-community/…`): `Cosmos3-Nano` (16B),
`Cosmos3-Super` (64B), `Cosmos3-Super-Text2Image` (64B), `Cosmos3-Super-Image2Video` (64B),
`Cosmos3-Nano-Policy-DROID` (16B). Repos are **gated** — authenticate with HF before first use.

## Backends (each is a separate venv/container)

The single most important architectural fact: there is no one environment. Each integration is
installed into its **own** venv (or container) and **pins a CUDA build that must match the host
NVIDIA driver**. `cookbooks/cosmos3/README.md` is the shared setup hub; each cookbook README links
to the one backend section it needs.

| Backend | Surface | Install entry point |
| --- | --- | --- |
| **Diffusers** | Generator (audiovisual) | `Cosmos3OmniPipeline`; diffusers from git `main` |
| **Cosmos Framework** | Reasoner + Generator | `git clone github.com/NVIDIA/cosmos-framework` → `uv sync` (needs SSH access) |
| **vLLM** | Reasoner (OpenAI-compatible server) | `vllm` + `vllm-cosmos3` |
| **vLLM-Omni** | Generator (OpenAI-compatible server) | `vllm-omni`, `vllm serve … --omni` |
| **NIM** | Reasoner (prebuilt container, no venv) | `docker … nvcr.io/nim/...`, needs `NGC_API_KEY` |
| **Transformers** | Reasoner | Coming soon |

### CUDA tag must match the driver
This is the #1 source of breakage. `--torch-backend=auto` is **not** reliable for the vLLM
backends (they don't publish a wheel per CUDA minor). Pick the pair by driver:

- **Driver CUDA 13.x** → `cu130` (notebook default); e.g. `vllm==0.21.0`.
- **Driver CUDA 12.x** → `cu128`; e.g. `vllm==0.19.1`. For Cosmos Framework `uv sync` set
  `COSMOS3_UV_GROUP=cu128-train`; for Diffusers pass `--torch-backend=cu128`.

To check the host driver: `nvidia-smi --query-gpu=driver_version --format=csv`.

## Common commands

Authenticate once (gated repos):
```bash
uvx hf@latest auth login   # or: export HF_TOKEN=<token>
```

Diffusers generator env (the Python-first generator path):
```bash
uv venv --python 3.13 --seed --managed-python && source .venv/bin/activate
uv pip install --torch-backend=auto \
  "diffusers @ git+https://github.com/huggingface/diffusers.git" \
  accelerate av cosmos_guardrail huggingface_hub imageio imageio-ffmpeg torch torchvision transformers
```

Cosmos Framework (native PyTorch, launched with `torchrun`):
```bash
mkdir -p packages && git clone https://github.com/NVIDIA/cosmos-framework.git packages/cosmos3
cd packages/cosmos3
export GIT_LFS_SKIP_SMUDGE=1          # lerobot LFS test blobs aren't needed and break uv's git mirror
uv sync --all-extras --group=cu130-train   # or cu128-train on a CUDA 12.x driver
# Notebooks honor COSMOS3_UV_GROUP (default cu130-train).
```

Serve the Reasoner (vLLM) / Generator (vLLM-Omni):
```bash
vllm serve nvidia/Cosmos3-Nano \
  --hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}' \
  --async-scheduling --allowed-local-media-path / --port 8000
# If the build reports DeepGEMM unavailable: export VLLM_USE_DEEP_GEMM=0

vllm serve nvidia/Cosmos3-Nano --omni \
  --model-class-name Cosmos3OmniDiffusersPipeline \
  --allowed-local-media-path / --port 8000 --init-timeout 1800
```

Run a cookbook (no test runner — execute the notebook):
```bash
jupyter lab            # interactive, or:
jupyter nbconvert --to notebook --execute cookbooks/cosmos3/<path>/run_*.ipynb
```

## Gotchas worth knowing before you touch a cookbook

- **`uv >= 0.11.3` is required** for the Cosmos Framework `pyproject.toml` (parses `[tool.uv.audit]`,
  recognizes `--torch-backend=cu130`). Older uv fails on `uv sync` and on newer `--torch-backend`
  values. (The plain Diffusers `uv pip install` path tolerates older uv if you pin the torch index.)
- **Large checkpoint init**: 64B Super weights and guardrails make `vllm serve` startup slow —
  always pass `--init-timeout 1800`. Super needs multi-GPU: `--tensor-parallel-size N`, optionally
  `--enable-layerwise-offload`.
- **`curl --form-string`, not `-F`**, for `prompt`/`negative_prompt`/`extra_params`: `-F` treats
  `;` as a content-type separator and silently truncates the value.
- **Guardrails are on by default** (prompt screening + face blur). Disable per request with
  `extra_params={"guardrails":false}`, or server-wide via a `--deploy-config` YAML.
- **First generation run downloads the checkpoint** (tens of GiB) and diffusion is compute-heavy —
  long step times are expected, not a hang. Set `HF_HOME` to a large/shared disk.
- Cosmos-Framework and vLLM-Reasoner backends need access to `git@github.com:NVIDIA/cosmos-framework.git`.

## Editing docs

When you change a backend's install/serve instructions, the canonical copy is usually in the top
`README.md` *and* `cookbooks/cosmos3/README.md` (the per-cookbook READMEs link back to that hub).
Keep the CUDA-tag pairs (`cu130`↔`vllm==0.21.0`, `cu128`↔`vllm==0.19.1`) consistent across all
three when editing one. License is OpenMDW-1.1; contributions target the `main` branch.
