"""X-VLA multi-domain training script for Track X (X3.A/X3.B).

Usage (single node 8 GPU):
  cd /data/shared/ubuntu/workspace/xvla_scripts
  torchrun --nproc_per_node=8 xvla_train.py \\
    --config X3B_stage_a   --output_dir /data/shared/ubuntu/local_ckpts/xvla_x3b_stage_a

Multi-node 2-node × 8 GPU:
  on master (uc01): torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \\
    --master_addr=10.x.x.x --master_port=29500 xvla_train.py --config X3B_stage_a
  on worker (uc02): same with --node_rank=1
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
from multi_domain_dataset import LeRobotEE6DDataset, MultiDomainDataset, XVLAHdf5Dataset
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

# E1 (use_proprio=False): lerobot's EE6DActionSpace.preprocess unconditionally does
# `proprio_m[..., gripper_idx] = 0.0`, which IndexErrors when proprio has 0 columns
# (proprio_dim=0). Guard it for the empty-proprio case (no-op change when proprio has channels,
# so safe for normal use_proprio=True runs). Action gripper-zeroing is preserved.
from lerobot.policies.xvla import action_hub as _action_hub
_orig_ee6d_preprocess = _action_hub.EE6DActionSpace.preprocess
def _ee6d_preprocess_safe(self, proprio, action, mode="train"):
    if proprio.shape[-1] == 0:  # use_proprio=False → skip proprio gripper-zeroing
        action_m = action.clone()
        action_m[..., self.gripper_idx] = 0.0
        return proprio, action_m
    return _orig_ee6d_preprocess(self, proprio, action, mode)
_action_hub.EE6DActionSpace.preprocess = _ee6d_preprocess_safe

# ==================== CONFIGS ====================
DATA_ROOT = "/data/shared/ubuntu/workspace/dataset_ee6d"  # legacy (buggy pipeline) — superseded by SB
SB = os.environ.get("XVLA_SB", "/data/shared/ubuntu/workspace/deepdive_kai0/xvla/data/self_built")  # X-VLA self-built EE6D (fixed pipeline); override via XVLA_SB env for local run
CKPT_INIT = os.environ.get("XVLA_CKPT_INIT", "/data/shared/ubuntu/workspace/xvla_ckpts")  # base init model; override via XVLA_CKPT_INIT env
BART_TOK = os.environ.get("XVLA_BART_TOK", "facebook/bart-large")  # text-encoder tokenizer; point to a local dir on offline (HF_HUB_OFFLINE) nodes
PROMPT = "Flatten and fold the cloth."
# Exp-O (§0.NEW.7): official Soft-Fold hdf5 root + ee6d action cache (override via env on each cluster)
SOFTFOLD_ROOT = os.environ.get("XVLA_SOFTFOLD_ROOT", "/data/shared/ubuntu/workspace/deepdive_kai0/xvla/data/xvla_soft_fold")
SOFTFOLD_CACHE = os.environ.get("XVLA_SOFTFOLD_CACHE", f"{SOFTFOLD_ROOT}/action_ee6d_cache")

CONFIGS = {
    "X3B_stage_a": dict(
        datasets=[
            dict(root=f"{SB}/kai0_base", domain_id=19, prompt=PROMPT, weight=1.0),
            dict(root=f"{SB}/kai0_dagger", domain_id=19, prompt=PROMPT, weight=1.0),
            dict(root=f"{SB}/vis_v2_merged", domain_id=20, prompt=PROMPT, weight=7.0),
        ],
        steps=20_000,
        lr=1e-4,
        warmup_steps=1000,
        freeze_steps=1000,  # backbone frozen for first 1000 steps
        batch_size_per_gpu=8,
        vlm_lr_scale=0.1,
    ),
    "X3A_stage_a": dict(
        datasets=[
            dict(root=f"{SB}/kai0_base", domain_id=19, prompt=PROMPT, weight=1.0, type="parquet"),
            dict(root=f"{SB}/kai0_dagger", domain_id=19, prompt=PROMPT, weight=1.0, type="parquet"),
            dict(root=f"{SB}/vis_v2_merged", domain_id=20, prompt=PROMPT, weight=7.0, type="parquet"),
            dict(root="/data/shared/ubuntu/workspace/deepdive_kai0/xvla/data/xvla_soft_fold",
                 action_cache_dir=f"{SB}/xvla_soft_fold_action_cache",
                 domain_id=21, prompt=PROMPT, weight=2.0, type="hdf5"),
        ],
        steps=20_000,
        lr=1e-4,
        warmup_steps=1000,
        freeze_steps=1000,
        batch_size_per_gpu=8,
        vlm_lr_scale=0.1,
    ),
    "X3C_vis_only_direct": dict(
        # Ablation: skip Stage A continual pretrain, init from lerobot/xvla-base directly,
        # finetune on vis-only. Measures the actual contribution of Stage A multi-domain pretrain.
        datasets=[
            dict(root=f"{SB}/vis_v2_merged", domain_id=20, prompt=PROMPT, weight=1.0),
        ],
        steps=20_000,
        lr=5e-5,
        warmup_steps=500,
        freeze_steps=1000,  # longer freeze than Stage B (500): raw base needs more warmup
        batch_size_per_gpu=8,
        vlm_lr_scale=0.1,
    ),
    "X3_stage_b": dict(
        datasets=[
            dict(root=f"{SB}/vis_v2_merged", domain_id=20, prompt=PROMPT, weight=1.0),
        ],
        steps=10_000,
        lr=5e-5,
        warmup_steps=500,
        freeze_steps=500,
        batch_size_per_gpu=8,
        vlm_lr_scale=0.1,
    ),
    "A_0423_0527": dict(
        # Single-dataset direct finetune from xvla-base on A_0423_0527 (cloth-fold, 1085 ep,
        # 1.37M chunk-samples). EE6D built with FIXED converter (interleaved rot6d + binarized
        # gripper). Mirrors X3C_vis_only_direct (direct-from-base). 30k steps ≈ 1.40 epoch at
        # eff batch 64 — A_0423_0527 is ~32% larger than vis_v2_merged, so 30k matches/exceeds
        # X3.C's validated 1.23-epoch exposure. save every 2k → 15 ckpts, pick best by eval.
        datasets=[
            dict(root="/data/shared/ubuntu/workspace/deepdive_kai0/xvla/data/self_built/A_0423_0527", domain_id=20, prompt=PROMPT, weight=1.0),
        ],
        steps=30_000,
        lr=5e-5,
        warmup_steps=500,
        freeze_steps=1000,
        batch_size_per_gpu=8,
        vlm_lr_scale=0.1,
    ),
    # ===== 控制变量新 X3 三件套 (2026-05-29): 三者 vis 数据统一 = A_0423_0527, 同超参 (30k/5e-5/
    # warmup500/freeze1000), 唯一变量 = 域组成。X3.C = 上面的 `A_0423_0527` (vis-only)。=====
    "X3B_a0423": dict(
        # 新 X3.B: kai(base+dagger) + A_0423_0527(vis ×7)。测 kai 域对 vis 部署的贡献。
        datasets=[
            dict(root=f"{SB}/kai0_base", domain_id=19, prompt=PROMPT, weight=1.0),
            dict(root=f"{SB}/kai0_dagger", domain_id=19, prompt=PROMPT, weight=1.0),
            dict(root=f"{SB}/A_0423_0527", domain_id=20, prompt=PROMPT, weight=7.0),
        ],
        steps=30_000,
        lr=5e-5,
        warmup_steps=500,
        freeze_steps=1000,
        batch_size_per_gpu=8,
        vlm_lr_scale=0.1,
    ),
    "X3A_a0423": dict(
        # 新 X3.A: kai(base+dagger) + A_0423_0527(vis ×7) + xvla_soft_fold(×2)。测第三方 XVLA 域贡献。
        datasets=[
            dict(root=f"{SB}/kai0_base", domain_id=19, prompt=PROMPT, weight=1.0, type="parquet"),
            dict(root=f"{SB}/kai0_dagger", domain_id=19, prompt=PROMPT, weight=1.0, type="parquet"),
            dict(root=f"{SB}/A_0423_0527", domain_id=20, prompt=PROMPT, weight=7.0, type="parquet"),
            dict(root="/data/shared/ubuntu/workspace/deepdive_kai0/xvla/data/xvla_soft_fold",
                 action_cache_dir=f"{SB}/xvla_soft_fold_action_cache",
                 domain_id=21, prompt=PROMPT, weight=2.0, type="hdf5"),
        ],
        steps=30_000,
        lr=5e-5,
        warmup_steps=500,
        freeze_steps=1000,
        batch_size_per_gpu=8,
        vlm_lr_scale=0.1,
    ),
    # ===== §0.NEW 控制变量 X3 三件套 (2026-05-31): vis = A_new_smooth_800 (811 ep, X1 cleaned,
    # 真机已验证 work). 同超参 30k/5e-5/warmup500/freeze1000, 唯一变量=域组成. 取代 A_0423_0527 版
    # (那版 eval 是 fit 非泛化 + 真机未验证). EE6D fixed 转换器, vis 权重 ×7. =====
    "X3C_smooth800": dict(
        # X3.C: vis-only (smooth_800).
        datasets=[
            dict(root=f"{SB}/A_new_smooth_800_xvla", domain_id=20, prompt=PROMPT, weight=1.0),
        ],
        steps=30_000,
        lr=5e-5,
        warmup_steps=500,
        freeze_steps=1000,
        batch_size_per_gpu=8,
        vlm_lr_scale=0.1,
    ),
    # P0 (2026-06-02): X3.C 修 R1 (ImageNet 归一化, dataset 层无条件生效) + 对齐官方配方.
    # 官方 finetune 示例: iters=50k warmup=2000 lr=1e-4 wd=0.0 freeze=1000 (X-VLA/README + train.py).
    # 适配: eff batch 64 (8gpu×8) vs 官方 16; 步数 60k (≈4.3 epoch, 官方 50k 量级 + R1 后视觉
    # 前端重适应余量); lr schedule 保留 cosine (定长训练比官方 constant 更稳); 加 ColorJitter.
    "X3C_smooth800_p0": dict(
        datasets=[
            dict(root=f"{SB}/A_new_smooth_800_xvla", domain_id=20, prompt=PROMPT, weight=1.0),
        ],
        steps=60_000,
        lr=1e-4,            # 对齐官方 (配 vlm_lr_scale=0.1 → VLM lr 1e-5)
        warmup_steps=2000,  # 对齐官方
        freeze_steps=1000,  # 对齐官方
        weight_decay=0.0,   # 对齐官方 (默认 1e-4)
        batch_size_per_gpu=8,
        vlm_lr_scale=0.1,
        image_aug=True,     # 对齐官方 ColorJitter(0.2)
    ),
    # E1 (2026-06-09): vision-blind 确诊实验 — 单变量 = use_proprio=False (关 proprio 输入),
    # 其余与 X3C_smooth800_p0 完全相同。根因认证: p0/d5anchor 离线 vision-ablation 视觉/本体
    # 影响比=0.000 (纯开环, 靠 proprio 捷径), 见 docs/training/future_plans/plans/
    # xvla_proprio_shortcut_openloop_fix.md。判据: 训完跑 eval_xvla_vision_ablation_offline.py,
    # 视觉影响比若从 0.000 抬到 ≳0.5 = 捷径修复路径成立 (此版为确诊, 非最终模型)。
    "X3C_smooth800_noproprio": dict(
        datasets=[
            dict(root=f"{SB}/A_new_smooth_800_xvla", domain_id=20, prompt=PROMPT, weight=1.0),
        ],
        steps=60_000,
        lr=1e-4,
        warmup_steps=2000,
        freeze_steps=1000,
        weight_decay=0.0,
        batch_size_per_gpu=8,
        vlm_lr_scale=0.1,
        image_aug=True,
        use_proprio=False,  # ⭐ E1: 关 proprio 输入 → 强制模型读视觉 (proprio_dim=0)
    ),
    # D5 修复 (2026-06-07): 单变量 = action 表示从"30 连续帧(1s 稠密)"改为
    # "30 anchor 铺在 2s(intention abstraction, action_qdur=2.0)", 对齐官方 X-VLA。
    # 其余与 X3C_smooth800_p0 完全相同。验证: offline 欠到位(pred chunk 位移/GT 60-80%)是否消失。
    # 见 docs/training/analysis/xvla_vs_official_gap_rootcause.md §7。
    "X3C_smooth800_d5anchor": dict(
        datasets=[
            dict(root=f"{SB}/A_new_smooth_800_xvla", domain_id=20, prompt=PROMPT, weight=1.0),
        ],
        steps=60_000,
        lr=1e-4,
        warmup_steps=2000,
        freeze_steps=1000,
        weight_decay=0.0,
        batch_size_per_gpu=8,
        vlm_lr_scale=0.1,
        image_aug=True,
        action_qdur=2.0,    # ⭐ D5 修复: 30 anchor over 2s (对齐官方 real_world.py qdur=2.0)
    ),
    # Exp-O (§0.NEW.7): official Soft-Fold, SAME pipeline as Exp-S (X3C_smooth800_d5anchor) —
    # only the data differs (official 1532ep hdf5 via XVLAHdf5Dataset + action_qdur=2.0 + ImageNet + jitter).
    # Isolates whether OUR dataset content is the problem. domain_id=21 (xvla).
    "X3_official_softfold_d5anchor": dict(
        datasets=[
            dict(type="hdf5", root=SOFTFOLD_ROOT, action_cache_dir=SOFTFOLD_CACHE,
                 domain_id=21, prompt=PROMPT, weight=1.0),
        ],
        steps=60_000,
        lr=1e-4,
        warmup_steps=2000,
        freeze_steps=1000,
        weight_decay=0.0,
        batch_size_per_gpu=8,
        vlm_lr_scale=0.1,
        image_aug=True,
        action_qdur=2.0,    # D5 anchor (XVLAHdf5Dataset linspace), identical to Exp-S
    ),
    "X3C_smooth800_100k": dict(
        # §0.NEW.5: X3.C vis-only 延长到 100k step (≈7 epoch) 验证 X-VLA 是否欠训。
        # 与 X3C_smooth800 完全相同, 仅 steps 30k→100k (cosine decay 自动拉到 100k)。
        datasets=[
            dict(root=f"{SB}/A_new_smooth_800_xvla", domain_id=20, prompt=PROMPT, weight=1.0),
        ],
        steps=100_000,
        lr=5e-5,
        warmup_steps=500,
        freeze_steps=1000,
        batch_size_per_gpu=8,
        vlm_lr_scale=0.1,
    ),
    "X3B_smooth800": dict(
        # X3.B: kai(base+dagger) + smooth_800(vis ×7).
        datasets=[
            dict(root=f"{SB}/kai0_base", domain_id=19, prompt=PROMPT, weight=1.0),
            dict(root=f"{SB}/kai0_dagger", domain_id=19, prompt=PROMPT, weight=1.0),
            dict(root=f"{SB}/A_new_smooth_800_xvla", domain_id=20, prompt=PROMPT, weight=7.0),
        ],
        steps=30_000,
        lr=5e-5,
        warmup_steps=500,
        freeze_steps=1000,
        batch_size_per_gpu=8,
        vlm_lr_scale=0.1,
    ),
    "X3A_smooth800": dict(
        # X3.A: kai(base+dagger) + smooth_800(vis ×7) + xvla_soft_fold(×2).
        datasets=[
            dict(root=f"{SB}/kai0_base", domain_id=19, prompt=PROMPT, weight=1.0, type="parquet"),
            dict(root=f"{SB}/kai0_dagger", domain_id=19, prompt=PROMPT, weight=1.0, type="parquet"),
            dict(root=f"{SB}/A_new_smooth_800_xvla", domain_id=20, prompt=PROMPT, weight=7.0, type="parquet"),
            dict(root="/data/shared/ubuntu/workspace/deepdive_kai0/xvla/data/xvla_soft_fold",
                 action_cache_dir=f"{SB}/xvla_soft_fold_action_cache",
                 domain_id=21, prompt=PROMPT, weight=2.0, type="hdf5"),
        ],
        steps=30_000,
        lr=5e-5,
        warmup_steps=500,
        freeze_steps=1000,
        batch_size_per_gpu=8,
        vlm_lr_scale=0.1,
    ),
    # ===== E0 (2026-06-18): 完全按官方 X-VLA train.py 训练配方, vis 数据 = Task_A v1 (action≠state,
    # 真实动作非 relabel 自述), 6 个日期目录 (639 ep, 04-23..04-30)。根因链: vision-blind 的真因是
    # action≡state relabel (开环复述观测), v1 满足 action≠state → 切断捷径数据源。阳性对照已证官方
    # 模型读视觉 (d_img 12.87mm), 同架构 → 根因在数据链, 本实验用官方配方 + 真实动作数据复现读视觉。
    # 官方配方对齐 (X-VLA train.py + README finetune):
    #   bf16 mixed | wd=0.0 | 4 param groups (vlm&soft_prompt ×0.1, transformer_core&action_head ×1.0)
    #   | freeze vlm+transformer_core 前 1000 步 | constant LR + warmup 2000 | lr 1e-4 | iters 50k
    #   | per-device batch 16 | action_qdur=2.0 (intention abstraction) | ImageNet norm + ColorJitter。
    # Plan: docs/training/future_plans/plans/xvla_proprio_shortcut_openloop_fix.md (E0)
    "E0_v1_official": dict(
        datasets=[
            dict(root=f"{SB}/A_v1_noRelabel_ee6d/2026-04-23", domain_id=20, prompt=PROMPT, weight=1.0),
            dict(root=f"{SB}/A_v1_noRelabel_ee6d/2026-04-24", domain_id=20, prompt=PROMPT, weight=1.0),
            dict(root=f"{SB}/A_v1_noRelabel_ee6d/2026-04-25", domain_id=20, prompt=PROMPT, weight=1.0),
            dict(root=f"{SB}/A_v1_noRelabel_ee6d/2026-04-28", domain_id=20, prompt=PROMPT, weight=1.0),
            dict(root=f"{SB}/A_v1_noRelabel_ee6d/2026-04-29", domain_id=20, prompt=PROMPT, weight=1.0),
            dict(root=f"{SB}/A_v1_noRelabel_ee6d/2026-04-30", domain_id=20, prompt=PROMPT, weight=1.0),
        ],
        steps=50_000,                 # 官方 iters
        lr=1e-4,                      # 官方 base lr (配 vlm_lr_scale=0.1 → vlm/soft_prompt 1e-5)
        warmup_steps=2000,            # 官方
        freeze_steps=1000,            # 官方 (4group → 冻 vlm+transformer_core)
        weight_decay=0.0,             # 官方
        batch_size_per_gpu=16,        # 官方 per-device 16
        vlm_lr_scale=0.1,             # 官方 learning_coef
        image_aug=True,               # 官方 ColorJitter(0.2)
        action_qdur=2.0,              # 官方 intention abstraction (real_world.py qdur=2.0)
        static_skip=True,             # 官方 domain_handler/base.py: 丢弃未来首步双臂几乎不动的退化帧
        bf16=True,                    # 官方 accelerate --mixed_precision bf16
        param_groups="4group_official",
        lr_schedule="constant",       # 官方 use_cosine_decay 默认 OFF
    ),
    # ===== E1_v1_official (2026-06-22): 两链叠加 — E0_v1_official 配方 + use_proprio=False。
    # 由来: E1 (断架构链 use_proprio=False, 旧 action≡state 数据) 与 E0 (断数据链 真实 action≠state
    # v1 数据, proprio ON) 各自单独都实测仍 vision-blind (d_img=0.00mm)。两次失败互补证明 action≡state
    # 捷径 *和* proprio 早融合捷径都各自足以让模型完全开环 → 唯一未证伪路径 = 同时切断两条链。
    # 仅相对 E0_v1_official 改一个变量: use_proprio=False (proprio_dim→0, 切 action_encoder.fc 列)。
    # 判据: eval_xvla_vision_ablation_dataset.py 的 d_img 绝对值 ≫ 0 (~10mm 对齐官方 SoftFold 12.87mm);
    #       不看 d_img/d_state 比值 (proprio 关掉后 d_state→0 会假阳, 见 E1 教训)。
    # Plan: docs/training/future_plans/plans/xvla_proprio_shortcut_openloop_fix.md §4.5
    "E1_v1_official": dict(
        datasets=[
            dict(root=f"{SB}/A_v1_noRelabel_ee6d/2026-04-23", domain_id=20, prompt=PROMPT, weight=1.0),
            dict(root=f"{SB}/A_v1_noRelabel_ee6d/2026-04-24", domain_id=20, prompt=PROMPT, weight=1.0),
            dict(root=f"{SB}/A_v1_noRelabel_ee6d/2026-04-25", domain_id=20, prompt=PROMPT, weight=1.0),
            dict(root=f"{SB}/A_v1_noRelabel_ee6d/2026-04-28", domain_id=20, prompt=PROMPT, weight=1.0),
            dict(root=f"{SB}/A_v1_noRelabel_ee6d/2026-04-29", domain_id=20, prompt=PROMPT, weight=1.0),
            dict(root=f"{SB}/A_v1_noRelabel_ee6d/2026-04-30", domain_id=20, prompt=PROMPT, weight=1.0),
        ],
        steps=50_000,                 # 同 E0
        lr=1e-4,
        warmup_steps=2000,
        freeze_steps=1000,
        weight_decay=0.0,
        batch_size_per_gpu=16,
        vlm_lr_scale=0.1,
        image_aug=True,
        action_qdur=2.0,
        static_skip=True,
        bf16=True,
        param_groups="4group_official",
        lr_schedule="constant",
        use_proprio=False,            # ⭐ 两链叠加: 关 proprio 输入 (proprio_dim=0) — 唯一相对 E0 的变量
    ),
}

# ==================== TRAIN ====================

def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world, local_rank
    return 0, 1, 0


def is_main(rank: int): return rank == 0


def build_dataset(cfg):
    datasets = []
    weights = []  # per-dataset weight
    for d in cfg["datasets"]:
        dtype = d.get("type", "parquet")
        if dtype == "hdf5":
            ds = XVLAHdf5Dataset(
                root=d["root"], action_cache_dir=d["action_cache_dir"],
                domain_id=d["domain_id"], task_prompt=d["prompt"],
                action_qdur=cfg.get("action_qdur", None),
                image_aug=cfg.get("image_aug", False),
            )
        else:
            ds = LeRobotEE6DDataset(d["root"], domain_id=d["domain_id"], task_prompt=d["prompt"],
                                    image_aug=cfg.get("image_aug", False),
                                    action_qdur=cfg.get("action_qdur", None),
                                    static_skip=cfg.get("static_skip", False))
        datasets.append(ds)
        weights.append(d["weight"])
    multi = MultiDomainDataset(datasets)
    # Per-sample weights (normalized within dataset, then weighted by domain weight)
    per_sample_w = []
    for w, ds in zip(weights, datasets):
        per_sample_w.extend([w / len(ds)] * len(ds))
    return multi, torch.tensor(per_sample_w, dtype=torch.float32)


def main(args):
    rank, world, local_rank = setup_distributed()
    cfg = CONFIGS[args.config]
    if args.max_steps is not None:  # smoke: cap loop length only (scheduler/warmup unchanged)
        cfg = {**cfg, "steps": args.max_steps}
        if is_main(rank): print(f"⚠ SMOKE: steps capped to {args.max_steps}")
    device = torch.device(f"cuda:{local_rank}")

    if is_main(rank):
        print(f"=== Track X {args.config} ===")
        print(f"world_size={world}, rank={rank}, local_rank={local_rank}")
        cfg_log = {k: v for k, v in cfg.items() if k != "datasets"}
        print(f"config: {json.dumps(cfg_log, indent=2)}")
        for d in cfg["datasets"]:
            print("  domain " + str(d["domain_id"]) + " (" + Path(d["root"]).name + ") weight=" + str(d["weight"]))

    # Data
    multi, sample_weights = build_dataset(cfg)
    if is_main(rank):
        print(f"total samples: {len(multi)}")

    # Sampler: WeightedRandom for single-node; DistributedSampler for multi
    # For balanced multi-domain in DDP, we use WeightedRandomSampler with per-rank slice
    # Simpler: just shuffle + random sampling (assume domain ratio approximated over many batches)
    if world > 1:
        sampler = DistributedSampler(multi, num_replicas=world, rank=rank, shuffle=True)
    else:
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, num_samples=len(multi), replacement=True
        )

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(BART_TOK)

    fixed_prompt = "Flatten and fold the cloth."
    cached_tokens = tok([fixed_prompt], padding="max_length", max_length=50, truncation=True, return_tensors="pt")["input_ids"][0]
    def collate(batch):
        out = {}
        for k in batch[0].keys():
            if isinstance(batch[0][k], torch.Tensor):
                out[k] = torch.stack([s[k] for s in batch])
        # Add language tokens (same prompt for all samples - cheap broadcast)
        out["observation.language.tokens"] = cached_tokens.unsqueeze(0).expand(len(batch), -1).contiguous()
        return out

    loader = DataLoader(
        multi, batch_size=cfg["batch_size_per_gpu"], sampler=sampler,
        num_workers=args.workers, collate_fn=collate, pin_memory=True, drop_last=True,
    )

    # Model
    if is_main(rank): print(f"loading {CKPT_INIT}...")
    # use_proprio 是 init ckpt config.json 的字段 (默认 True)。cfg 给 False 时 (E1 vision-blind 确诊)
    # 用 use_proprio=False 构造 → proprio_dim=0, _prepare_state 返回空 → 强制读视觉。
    # 关键: proprio 不是独立层, 而是 action_encoder.fc 这个共享 per-domain Linear 的输入列
    #   (输入 = cat[action, proprio, time], soft_transformer.py:381; fc.weight 形状随 proprio_dim 变)。
    # lerobot from_pretrained 把 strict 写死成 True (modeling_xvla.py:489) → 无法吞下这个形状变化
    #   (不是 unexpected key, 是 present key 的 shape mismatch)。故手动加载: 把 init ckpt 的
    #   action_encoder.fc.weight 切掉 proprio 那几列再 load (保留 action/time 预训练列), 其余照常。
    if cfg.get("use_proprio", True) is False:
        import os, safetensors.torch
        from lerobot.configs.policies import PreTrainedConfig
        _ovr = PreTrainedConfig.from_pretrained(CKPT_INIT)
        _ovr.use_proprio = False
        model = XVLAPolicy(_ovr)
        sd = safetensors.torch.load_file(os.path.join(CKPT_INIT, "model.safetensors"))
        ek = "model.vlm.language_model.model.encoder.embed_tokens.weight"
        shk = "model.vlm.language_model.model.shared.weight"
        if ek in sd: sd[shk] = sd[ek]
        _T = model.model.transformer
        da, dt, dp = _T.dim_action, _T.dim_time, _ovr.max_state_dim   # init-ckpt proprio block width
        out = _T.hidden_size
        key = "model.transformer.action_encoder.fc.weight"
        nd = sd[key].shape[0]
        assert sd[key].shape[1] == (da + dp + dt) * out, \
            f"fc layout mismatch: {tuple(sd[key].shape)} vs ({da}+{dp}+{dt})*{out}"
        # forward views fc as [in, out] (soft_transformer.py:251); input cols = [action, proprio, time]
        w = sd[key].view(nd, da + dp + dt, out)
        w = torch.cat([w[:, :da, :], w[:, da + dp:, :]], dim=1)       # drop proprio rows
        sd[key] = w.reshape(nd, (da + dt) * out).contiguous()
        missing, unexpected = model.load_state_dict(sd, strict=False)
        model.model._apply_dtype()
        model = model.to(device)
        if is_main(rank):
            print(f"⭐ E1 use_proprio=False: proprio_dim→0, sliced action_encoder.fc {da+dp+dt}→{da+dt} cols/domain "
                  f"(load missing={len(missing)} unexpected={len(unexpected)})")
    else:
        model = XVLAPolicy.from_pretrained(CKPT_INIT).to(device)
    if world > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Optimizer: param-group mode selected per-config (backward compatible default = legacy 2-group)
    inner = model.module if world > 1 else model
    group_mode = cfg.get("param_groups", "2group")
    if group_mode == "4group_official":
        # 官方 4 组 (X-VLA train.py): lr base, learning_coef 0.1 →
        #   vlm & soft_prompts ×0.1 (1e-5); transformer_core & action_heads ×1.0 (1e-4)。
        # 关键: 用 ".vlm." 匹配整个 VLM (含 language_model.*) — 修旧 2-group "florence"/"vision"
        #   name-keying 漏掉 model.vlm.language_model.* 导致 VLM 文本编码器被 10× LR 训练的 bug。
        buckets = {"vlm": [], "soft_prompts": [], "action_heads": [], "transformer_core": []}
        for n, p in inner.named_parameters():
            if not p.requires_grad: continue
            if ".vlm." in n:
                buckets["vlm"].append(p)
            elif "transformer.soft_prompt_hub" in n:
                buckets["soft_prompts"].append(p)
            elif "transformer.action_encoder" in n or "transformer.action_decoder" in n:
                buckets["action_heads"].append(p)
            elif ".transformer." in n:
                buckets["transformer_core"].append(p)
            else:
                raise RuntimeError(f"4group_official: param not bucketed: {n}")
        s = cfg["vlm_lr_scale"]
        optimizer = torch.optim.AdamW([
            {"params": buckets["vlm"], "lr": cfg["lr"] * s},
            {"params": buckets["soft_prompts"], "lr": cfg["lr"] * s},
            {"params": buckets["transformer_core"], "lr": cfg["lr"]},
            {"params": buckets["action_heads"], "lr": cfg["lr"]},
        ], weight_decay=cfg.get("weight_decay", 1e-4), betas=(0.9, 0.95))
        # 官方 freeze_steps: 冻结 vlm + transformer_core (仅 soft_prompts + action_heads 训练)
        freeze_params = buckets["vlm"] + buckets["transformer_core"]
        if is_main(rank):
            print("4group_official param groups: " + ", ".join(
                f"{k}={len(buckets[k])}t" for k in buckets))
    else:
        vlm_params, other_params = [], []
        for n, p in inner.named_parameters():
            if not p.requires_grad: continue
            if "florence" in n.lower() or "vision" in n.lower():
                vlm_params.append(p)
            else:
                other_params.append(p)
        optimizer = torch.optim.AdamW([
            {"params": vlm_params, "lr": cfg["lr"] * cfg["vlm_lr_scale"]},
            {"params": other_params, "lr": cfg["lr"]},
        ], weight_decay=cfg.get("weight_decay", 1e-4), betas=(0.9, 0.95))
        freeze_params = vlm_params  # legacy: only VLM frozen during freeze_steps
    sched_mode = cfg.get("lr_schedule", "cosine")
    if sched_mode == "constant":
        from transformers import get_constant_schedule_with_warmup
        scheduler = get_constant_schedule_with_warmup(optimizer, cfg["warmup_steps"])
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, cfg["warmup_steps"], cfg["steps"])

    # Output dir
    if is_main(rank):
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        json.dump({**cfg, "world_size": world}, open(f"{args.output_dir}/config.json", "w"), indent=2)

    model.train()
    step = 0
    t0 = time.time()
    data_iter = iter(loader)
    while step < cfg["steps"]:
        try: batch = next(data_iter)
        except StopIteration:
            if world > 1 and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(step)
            data_iter = iter(loader); batch = next(data_iter)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Freeze backbone for first freeze_steps (4group_official: vlm+transformer_core; legacy: vlm)
        if step == 0 and cfg["freeze_steps"] > 0:
            for p in freeze_params:
                p.requires_grad = False
            if is_main(rank): print(f"backbone frozen ({len(freeze_params)} tensors, first {cfg['freeze_steps']} steps)")
        if step == cfg["freeze_steps"] and cfg["freeze_steps"] > 0:
            for p in freeze_params:
                p.requires_grad = True
            if is_main(rank): print(f"backbone UNFROZEN at step {step}")

        if cfg.get("bf16", False):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, log_dict = (model.module.forward(batch) if world > 1 else model.forward(batch))
        else:
            loss, log_dict = (model.module.forward(batch) if world > 1 else model.forward(batch))
        optimizer.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(inner.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % 50 == 0 and is_main(rank):
            elapsed = time.time() - t0
            rate = (step + 1) / elapsed
            eta_h = (cfg["steps"] - step) / rate / 3600
            n_steps = cfg["steps"]
            print(f"step {step}/{n_steps} loss={loss.item():.4f} gnorm={gnorm.item():.3f} rate={rate:.2f}it/s eta={eta_h:.1f}h")

        if step % 2000 == 0 and step > 0 and is_main(rank):
            ckpt_path = f"{args.output_dir}/step_{step:06d}"
            Path(ckpt_path).mkdir(parents=True, exist_ok=True)
            # Use torch.save to bypass draccus serialization bug
            torch.save({"model_state": inner.state_dict(), "step": step}, f"{ckpt_path}/state_dict.pt")
            print(f"saved {ckpt_path}")
        step += 1

    if is_main(rank):
        ckpt_path = f"{args.output_dir}/step_final"
        Path(ckpt_path).mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": inner.state_dict(), "step": step}, f"{ckpt_path}/state_dict.pt")
        print(f"=== DONE: {ckpt_path} ===")

    if world > 1: dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS.keys()))
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=None, help="smoke: cap training loop steps")
    args = ap.parse_args()
    main(args)
