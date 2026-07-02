#!/usr/bin/env python
"""Run unified LMWM inference and export VLA-ready latent subgoal predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lmwm.models import UnifiedLMWM  # noqa: E402


def entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--dataset_npz", type=Path)
    ap.add_argument("--graph_npz", type=Path)
    ap.add_argument("--output_dir", type=Path)
    ap.add_argument("--max_samples", type=int)
    ap.add_argument("--device")
    ap.add_argument("--greedy_conf_threshold", type=float)
    ap.add_argument("--max_product_conf_threshold", type=float)
    ap.add_argument(
        "--weak_milestones",
        default=None,
        help="Comma-separated current milestone ids that should use graph fallback.",
    )
    args = ap.parse_args()
    cfg = {}
    if args.config:
        with args.config.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    checkpoint = args.checkpoint or Path(cfg.get("checkpoint", ""))
    dataset_npz = args.dataset_npz or Path(cfg.get("dataset_npz", ""))
    graph_npz = args.graph_npz or Path(cfg.get("graph_npz", ""))
    output_dir = args.output_dir or Path(cfg.get("output_dir", ""))
    if not checkpoint or not dataset_npz or not graph_npz or not output_dir:
        raise SystemExit("--checkpoint, --dataset_npz, --graph_npz, and --output_dir are required unless provided by --config")
    max_samples = int(args.max_samples if args.max_samples is not None else cfg.get("max_samples", 1024))
    device_name = args.device or str(cfg.get("device", "cuda:0"))
    greedy_conf_threshold = float(
        args.greedy_conf_threshold
        if args.greedy_conf_threshold is not None
        else cfg.get("greedy_conf_threshold", 0.93)
    )
    max_product_conf_threshold = float(
        args.max_product_conf_threshold
        if args.max_product_conf_threshold is not None
        else cfg.get("max_product_conf_threshold", 0.93)
    )
    weak_cfg = cfg.get("weak_milestones", [])
    if args.weak_milestones is not None:
        weak_text = args.weak_milestones
    elif isinstance(weak_cfg, list):
        weak_text = ",".join(str(x) for x in weak_cfg)
    else:
        weak_text = str(weak_cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(checkpoint, map_location="cpu")
    z = np.load(dataset_npz)
    g = np.load(graph_npz)
    n = min(max_samples, int(z["current"].shape[0]))
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    input_dim = int(z["current"].shape[1])
    latent_dim = int(g["prototype_table"].shape[1])
    num_milestones = int(g["transition_probs"].shape[0])
    model_cfg = ckpt["config"].get("model", {})
    model = UnifiedLMWM(
        input_dim,
        latent_dim,
        num_milestones,
        int(model_cfg.get("hidden_dim", 512)),
        int(model_cfg.get("depth", 2)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    current = torch.from_numpy(z["current"][:n].astype(np.float32)).to(device)
    current_m = z["current_milestone"][:n].astype(np.int64)
    greedy_target = g["greedy_next"].astype(np.int64)[current_m]
    max_product_target = g["max_product_next"].astype(np.int64)[current_m]
    proto = torch.from_numpy(g["prototype_table"].astype(np.float32)).to(device)
    weak_milestones = {
        int(x)
        for x in weak_text.split(",")
        if x.strip()
    }
    weak_mask_np = np.array([m in weak_milestones for m in current_m], dtype=bool)

    with torch.no_grad():
        out = model(current)
        transition_probs = F.softmax(out["transition_logits"], dim=-1)
        greedy_probs = F.softmax(out["greedy_logits"], dim=-1)
        max_product_probs = F.softmax(out["max_product_logits"], dim=-1)
        greedy_pred = greedy_probs.argmax(dim=-1)
        max_product_pred = max_product_probs.argmax(dim=-1)
        greedy_conf = greedy_probs.max(dim=-1).values
        max_product_conf = max_product_probs.max(dim=-1).values
        transition_conf = transition_probs.max(dim=-1).values
        transition_entropy = entropy(transition_probs)
        greedy_proto = out["greedy_proto"]
        max_product_proto = out["max_product_proto"]
        greedy_target_proto = proto[torch.from_numpy(greedy_target).to(device)]
        max_product_target_proto = proto[torch.from_numpy(max_product_target).to(device)]
        greedy_cos = F.cosine_similarity(greedy_proto, greedy_target_proto, dim=-1)
        max_product_cos = F.cosine_similarity(max_product_proto, max_product_target_proto, dim=-1)

        graph_greedy = torch.from_numpy(greedy_target).to(device)
        graph_max_product = torch.from_numpy(max_product_target).to(device)
        weak_mask = torch.from_numpy(weak_mask_np).to(device)
        greedy_fallback_mask = (greedy_conf < greedy_conf_threshold) | weak_mask
        max_product_fallback_mask = (max_product_conf < max_product_conf_threshold) | weak_mask
        hybrid_greedy = torch.where(greedy_fallback_mask, graph_greedy, greedy_pred)
        hybrid_max_product = torch.where(max_product_fallback_mask, graph_max_product, max_product_pred)
        graph_greedy_proto = proto[graph_greedy]
        graph_max_product_proto = proto[graph_max_product]
        hybrid_greedy_proto = torch.where(greedy_fallback_mask[:, None], graph_greedy_proto, greedy_proto)
        hybrid_max_product_proto = torch.where(max_product_fallback_mask[:, None], graph_max_product_proto, max_product_proto)
        hybrid_greedy_cos = F.cosine_similarity(hybrid_greedy_proto, greedy_target_proto, dim=-1)
        hybrid_max_product_cos = F.cosine_similarity(hybrid_max_product_proto, max_product_target_proto, dim=-1)

    result_npz = output_dir / "predictions.npz"
    np.savez_compressed(
        result_npz,
        current_milestone=current_m,
        greedy_pred=greedy_pred.cpu().numpy().astype(np.int64),
        max_product_pred=max_product_pred.cpu().numpy().astype(np.int64),
        greedy_target=greedy_target.astype(np.int64),
        max_product_target=max_product_target.astype(np.int64),
        transition_probs=transition_probs.cpu().numpy().astype(np.float32),
        greedy_subgoal_latent=greedy_proto.cpu().numpy().astype(np.float32),
        max_product_subgoal_latent=max_product_proto.cpu().numpy().astype(np.float32),
        graph_greedy=greedy_target.astype(np.int64),
        graph_max_product=max_product_target.astype(np.int64),
        graph_greedy_subgoal_latent=graph_greedy_proto.cpu().numpy().astype(np.float32),
        graph_max_product_subgoal_latent=graph_max_product_proto.cpu().numpy().astype(np.float32),
        hybrid_greedy=hybrid_greedy.cpu().numpy().astype(np.int64),
        hybrid_max_product=hybrid_max_product.cpu().numpy().astype(np.int64),
        hybrid_greedy_subgoal_latent=hybrid_greedy_proto.cpu().numpy().astype(np.float32),
        hybrid_max_product_subgoal_latent=hybrid_max_product_proto.cpu().numpy().astype(np.float32),
        greedy_fallback_mask=greedy_fallback_mask.cpu().numpy().astype(bool),
        max_product_fallback_mask=max_product_fallback_mask.cpu().numpy().astype(bool),
        greedy_confidence=greedy_conf.cpu().numpy().astype(np.float32),
        max_product_confidence=max_product_conf.cpu().numpy().astype(np.float32),
        transition_confidence=transition_conf.cpu().numpy().astype(np.float32),
        transition_entropy=transition_entropy.cpu().numpy().astype(np.float32),
        greedy_proto_cos=greedy_cos.cpu().numpy().astype(np.float32),
        max_product_proto_cos=max_product_cos.cpu().numpy().astype(np.float32),
        hybrid_greedy_proto_cos=hybrid_greedy_cos.cpu().numpy().astype(np.float32),
        hybrid_max_product_proto_cos=hybrid_max_product_cos.cpu().numpy().astype(np.float32),
    )
    greedy_pred_np = greedy_pred.cpu().numpy()
    max_product_pred_np = max_product_pred.cpu().numpy()
    hybrid_greedy_np = hybrid_greedy.cpu().numpy()
    hybrid_max_product_np = hybrid_max_product.cpu().numpy()
    summary = {
        "config": str(args.config) if args.config else None,
        "checkpoint": str(checkpoint),
        "dataset_npz": str(dataset_npz),
        "graph_npz": str(graph_npz),
        "output_npz": str(result_npz),
        "num_samples": n,
        "greedy_top1": float((greedy_pred_np == greedy_target).mean()),
        "max_product_top1": float((max_product_pred_np == max_product_target).mean()),
        "hybrid_greedy_top1": float((hybrid_greedy_np == greedy_target).mean()),
        "hybrid_max_product_top1": float((hybrid_max_product_np == max_product_target).mean()),
        "greedy_fallback_rate": float(greedy_fallback_mask.float().mean().item()),
        "max_product_fallback_rate": float(max_product_fallback_mask.float().mean().item()),
        "greedy_conf_threshold": float(greedy_conf_threshold),
        "max_product_conf_threshold": float(max_product_conf_threshold),
        "weak_milestones": sorted(weak_milestones),
        "greedy_confidence_mean": float(greedy_conf.mean().item()),
        "max_product_confidence_mean": float(max_product_conf.mean().item()),
        "transition_confidence_mean": float(transition_conf.mean().item()),
        "transition_entropy_mean": float(transition_entropy.mean().item()),
        "greedy_proto_cos_mean": float(greedy_cos.mean().item()),
        "max_product_proto_cos_mean": float(max_product_cos.mean().item()),
        "hybrid_greedy_proto_cos_mean": float(hybrid_greedy_cos.mean().item()),
        "hybrid_max_product_proto_cos_mean": float(hybrid_max_product_cos.mean().item()),
        "api_fields": [
            "current_milestone",
            "transition_probs",
            "greedy_pred",
            "max_product_pred",
            "greedy_subgoal_latent",
            "max_product_subgoal_latent",
            "hybrid_greedy",
            "hybrid_max_product",
            "hybrid_greedy_subgoal_latent",
            "hybrid_max_product_subgoal_latent",
            "greedy_confidence",
            "max_product_confidence",
            "transition_entropy",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
