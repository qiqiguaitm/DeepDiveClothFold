# SPDX-License-Identifier: OpenMDW-1.1
"""Cosmos3-Nano-Policy (wam_fold-adapted) open-loop eval + HTML report.

Mirrors giga_world_policy's ``scripts/wam_pipeline/episode_report.py`` report style
1:1 (same CSS/HTML template, 14-D action curve PNGs, 2-row GT/pred 3-view videos,
aggregate action-MAE table vs PI05) and reuses eval_watch's metric functions
(``video_metrics_gpu`` PSNR/SSIM/temporal/LPIPS; ``build_window_indices``;
action mae@{1,10,24,48}) so the numbers are directly comparable.

WHAT IT DOES
  0. (optional) export the trained DCP checkpoint -> a loadable HF safetensors dir
     via ``cosmos_framework.scripts.export_model`` (skipped if export dir exists).
  1. Load the exported Cosmos3 OmniMoTModel once (OmniInference.create, mirroring
     ``cosmos_framework/scripts/action_policy_server_robolab.py``).
  2. For N sampled val episodes, enumerate eval windows with eval_watch's
     ``build_window_indices`` (coverage='exec', exec_horizon stride). Per window:
       - build the wam_fold "policy" batch (first-frame 3-view concat image +
         task text + domain wam_fold[16], raw_action_dim=14, zeros action) and run
         ``model.generate_samples_from_batch`` -> pred action [chunk,14] (normalized)
         + pred vision latent -> decode to frames.
       - de-normalize the action with the SAME quantile mapping the dataset used
         (``action_normalization.normalize_action(method="quantile")`` inverse).
  3. Compute metrics identical to eval_watch (video PSNR/SSIM/temporal/LPIPS;
     action_mae + mae@horizon).
  4. Emit ``report.html`` reusing episode_report's exact template: per-viz-episode
     14-D action curve PNG (GT black-dashed vs pred red) + 2-row 3-view GT/pred
     videos, and an aggregate 3-way comparison table
     (Cosmos3-Policy(adapted) vs GigaWorld-Policy vs tau-0-wm vs PI05).
  5. Sharded eval (--shard_id/--num_shards) like eval_watch, + an --aggregate mode;
     also works single-process. Robust: if pred video decode fails, still emits
     action metrics.

CITATIONS (real APIs this script targets)
  - inference call / batch construction (mode="policy", domain wam_fold, zeros
    action, raw_action_dim=14, sequence_plan, image_size):
      cosmos_framework/inference/action.py:74  build_action_batch
      cosmos_framework/inference/action.py:109 build_sequence_plan_from_mode("policy",chunk+1,chunk)
      cosmos_framework/scripts/action_policy_server_robolab.py:565 generate_samples_from_batch(...)
      cosmos_framework/scripts/action_policy_server_robolab.py:573 action = samples["action"][0][:, :action_dim]
      cosmos_framework/scripts/action_policy_server_robolab.py:596 video = model.decode(samples["vision"][0])
      cosmos_framework/model/vfm/omni_mot_model.py:2109 generate_samples_from_batch signature
  - de-normalization (inverse of training-time quantile normalize):
      cosmos_framework/data/vfm/action/action_normalization.py:39 normalize_action(method="quantile")
      cosmos_framework/data/vfm/action/datasets/wam_fold_dataset.py:265 normalize_action(action,"quantile",stats)
      => norm = 2*(a-q01)/(q99-q01)-1  =>  denorm a = (norm+1)/2*(q99-q01)+q01
      wam_fold is ABSOLUTE 14-D joint (NO add_state_to_action / delta-mask, unlike GWP).
  - concat_view video construction (head on top, [left|right] wrist halves bottom):
      cosmos_framework/data/vfm/action/datasets/wam_fold_dataset.py:216 _load_concat_video
  - model load (exported HF dir): OmniInference.create + OmniSetupOverrides
      cosmos_framework/scripts/action_policy_server_robolab.py:419 _build_setup_args
      cosmos_framework/inference/inference.py:1010 OmniInference._create
  - export: cosmos_framework/scripts/export_model.py (Args: --checkpoint-path, --config-file, -o)
  - report template (CSS/HTML/PNG/video) copied verbatim from:
      giga_world_policy/scripts/wam_pipeline/episode_report.py
  - metric fns reused from:
      giga_world_policy/scripts/wam_pipeline/eval_watch.py (video_metrics_gpu, build_window_indices)

AUTOREGRESSIVE WM ROLLOUT (to reach GWP's action_chunk=48 horizon)
  The wam_fold posttrain config trains chunk_length=16 (configs/.../wam_fold_nano.py:298), so the
  native model predicts 16 action steps and 16 video sub-frames per call. To report at GWP's horizons
  {1,10,24,48} we roll the WM out n_sub=ceil(action_chunk/model_chunk)=3 sub-chunks per window:
    - ACTION (POLICY, teacher-forced): each sub-chunk i is re-anchored on the *real* GT obs frame at
      f+i*model_chunk -> predict 16 actions; concat -> pred_action[:48,14] (deploy-style rollout).
    - VIDEO (FORWARD_DYNAMICS, autoregressive): init frame = GT concat at f; feed the GT action chunk
      (NORMALIZED with the dataset's quantile stats) per sub-chunk + the current frame; decode the
      predicted video sub-chunk; set the next frame = the LAST predicted frame (observations free-run);
      concat -> predicted video over 48 steps, sampled to 5 delta keyframes [0,12,24,36,48].
    - If FORWARD_DYNAMICS video decode is unavailable, fall back to POLICY-mode pred video.
  This makes mae@{1,10,24,48} all measurable (no "n/a (chunk=16)" cells).
  * generate_samples_from_batch returns {"vision":[lat],"action":[act]} (one per sample);
    action is normalized [chunk, max_action_dim], we slice [:, :14]. Verified against
    omni_mot_model.py:2221 docstring + robolab server slice.
  * Pred vision latent decodes via model.decode(latent)->[1,C,T,H,W] in [-1,1]
    (robolab server:597). The 3-view concat is the SAME layout as training, so pred
    video can be compared region-for-region against the GT concat at the model's
    output resolution.
  * num_steps/guidance default to 10 / 0.0 to match GWP eval (eval_watch steps=10,
    guidance_scale=0.0); change with --num_steps/--guidance.
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict

import numpy as np

# ---- locate cosmos3 + giga_world_policy on sys.path (env may already set PYTHONPATH) ----
_COSMOS3_DIR = os.environ.get(
    "COSMOS3_DIR", "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/packages/cosmos3"
)
_GWP_DIR = os.environ.get(
    "GWP_DIR", "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/giga_world_policy"
)
for _p in (_COSMOS3_DIR, os.path.join(_GWP_DIR, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ============================ report template (copied verbatim from episode_report.py) ===========
VK = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
DIM = [f"L_j{i}" for i in range(6)] + ["L_grip"] + [f"R_j{i}" for i in range(6)] + ["R_grip"]
PI05 = {1: 0.0219, 10: 0.0425, 24: 0.0743, 48: 0.1155}
TASK_TEXT = "Flatten and fold the cloth."
DOMAIN_NAME = "wam_fold"   # domain_id 16, raw_action_dim 14 (domain_utils.py)

CSS = """<style>
body{font-family:-apple-system,Segoe UI,monospace;max-width:1180px;margin:24px auto;padding:0 18px;color:#222;line-height:1.5}
h2{border-bottom:2px solid #3a6;padding-bottom:6px} h3{margin-top:26px;color:#3a6}
table{border-collapse:collapse;margin:10px 0} th,td{border:1px solid #ccc;padding:6px 14px;text-align:center}
th{background:#eef5f0} tr:nth-child(even) td{background:#fafafa}
details{margin:12px 0;border:1px solid #ddd;border-radius:8px;padding:10px 14px;background:#fcfcfc}
summary{cursor:pointer;font-weight:600;padding:4px 0;font-size:15px} summary:hover{color:#3a6}
video{margin:6px 8px 6px 0;border:1px solid #ccc;border-radius:6px;vertical-align:top} img{border:1px solid #eee;border-radius:6px}
.vids{display:flex;flex-wrap:wrap;gap:8px} .note{color:#666;font-size:13px}
ul{line-height:1.7} code{background:#f0f0f0;padding:1px 5px;border-radius:3px}
</style>"""


def _label(frames, text):
    import cv2
    out = np.ascontiguousarray(frames.copy())
    for t in range(len(out)):
        cv2.putText(out[t], text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
    return out


def _save_2row(gt, raw, path, fps=5):
    """GT/raw vertical 2-row (frames carry row name). Each [T,H,W,C] uint8. Mirror episode_report._save_2row."""
    import torch
    import torchvision
    T = min(len(gt), len(raw))
    cat = np.concatenate([_label(gt[:T], "GT"), _label(raw[:T], "pred")], axis=1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchvision.io.write_video(path, torch.from_numpy(cat), fps=fps)


# ============================ eval_watch metric fns (import; copy fallback) ===========
try:
    from wam_pipeline.eval_watch import build_window_indices, video_metrics_gpu  # type: ignore
except Exception:  # pragma: no cover - fallback so the file is self-contained
    def build_window_indices(val_root, coverage, stride, action_chunk, exec_horizon=16):
        eps = [json.loads(l) for l in open(os.path.join(val_root, "meta", "episodes.jsonl")) if l.strip()]
        eps = sorted(eps, key=lambda e: int(e["episode_index"]))
        _auto = {"episode": action_chunk, "exec": exec_horizon, "frames": 1}.get(coverage, exec_horizon)
        st = stride if stride and stride > 0 else _auto
        idxs, gs, info = [], 0, {}
        for e in eps:
            ei, L = int(e["episode_index"]), int(e["length"])
            last = max(1, L - action_chunk)
            for s in range(0, last, st):
                gi = gs + s; idxs.append(gi); info[gi] = (ei, s)
            gs += L
        return idxs, gs, info

    def video_metrics_gpu(pred_thwc, gt_thwc, lpips_fn=None, device="cuda"):
        import torch
        import torch.nn.functional as F
        T = min(pred_thwc.shape[0], gt_thwc.shape[0])
        p = pred_thwc[:T].permute(0, 3, 1, 2).contiguous()
        g = gt_thwc[:T].permute(0, 3, 1, 2).contiguous()
        C = p.shape[1]
        mse = ((p - g) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
        out = {"psnr": float((10.0 * torch.log10((255.0 ** 2) / mse)).mean().item())}
        ws = 11
        c = torch.arange(ws, device=device, dtype=p.dtype) - (ws - 1) / 2.0
        gw = torch.exp(-(c ** 2) / (2 * 1.5 ** 2)); gw = gw / gw.sum()
        w = (gw[:, None] * gw[None, :]).expand(C, 1, ws, ws); pad = ws // 2
        cv = lambda x: F.conv2d(x, w, padding=pad, groups=C)
        mu1, mu2 = cv(p), cv(g); mu1s, mu2s, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
        s1, s2, s12 = cv(p * p) - mu1s, cv(g * g) - mu2s, cv(p * g) - mu12
        C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        out["ssim"] = float((((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1s + mu2s + C1) * (s1 + s2 + C2))).mean().item())
        if T >= 2:
            out["temporal_absdiff_ratio"] = float(((p[1:] - p[:-1]).abs().mean() / ((g[1:] - g[:-1]).abs().mean() + 1e-6)).item())
        if lpips_fn is not None:
            import torch as _t
            with _t.no_grad():
                out["lpips"] = float(lpips_fn(p / 127.5 - 1.0, g / 127.5 - 1.0).mean().item())
        return out


# ============================ baselines for the 3-way comparison table ===========
def _load_baselines(args):
    """Read GWP summary.json + tau-0-wm eval json (tag p3_final) for the comparison table."""
    gwp, tau0 = {}, {}
    try:
        s = json.load(open(args.gwp_summary))
        gwp = {int(k): float(v) for k, v in s.get("raw_mae", {}).items()}
        gwp_vid = {k: s.get("latency", {}).get(k) for k in ("action_ms", "video_ms")}
    except Exception as e:
        print(f"[baseline] GWP summary unavailable: {e}", flush=True); gwp_vid = {}
    try:
        rows = json.load(open(args.tau0_json))
        row = next((r for r in rows if r.get("tag") == args.tau0_tag), rows[-1] if rows else {})
        tau0 = row
    except Exception as e:
        print(f"[baseline] tau0 json unavailable: {e}", flush=True)
    return gwp, gwp_vid, tau0


# ============================ de-normalization (inverse quantile) ===========
def _load_action_qstats(stats_path):
    raw = json.load(open(stats_path))["global"]["action"]
    return np.asarray(raw["q01"], np.float32), np.asarray(raw["q99"], np.float32)


def _denorm_action_quantile(norm_act, q01, q99):
    """Inverse of normalize_action(method='quantile'): a = (norm+1)/2*(q99-q01)+q01.
    norm_act: [chunk, 14] in [-1,1]. Returns [chunk, 14] in raw joint units."""
    denom = np.maximum(q99 - q01, 1e-8)
    return (norm_act + 1.0) / 2.0 * denom + q01


def _norm_action_quantile(raw_act, q01, q99):
    """Forward of action_normalization.normalize_action(method='quantile'):
    norm = clamp(2*(a-q01)/(q99-q01)-1, -1, 1). raw_act: [T,14] raw joint units.
    Returns [T,14] in [-1,1]. Mirrors action_normalization.py:39-42 (clamp included).
    Used to feed GT actions into FORWARD_DYNAMICS exactly as the dataset normalized them
    (wam_fold_dataset.py:265 normalize_action(action,"quantile",stats))."""
    denom = np.maximum(q99 - q01, 1e-8)
    return np.clip(2.0 * (raw_act - q01) / denom - 1.0, -1.0, 1.0)


# ============================ concat_view video (matches WamFoldLeRobotDataset._load_concat_video) ===========
def _build_concat_view(high, left, right):
    """high/left/right: torch [C,Tn,H,W] float[0,1] (or [C,H,W]). Returns torch [C,Tn,H2,W2] float[0,1].
    Head on top full-width; bottom = [left|right] each resized to (H/2, W/2). Mirrors wam_fold_dataset:230."""
    import torch
    import torch.nn.functional as F
    def _4d(x):
        return x.unsqueeze(1) if x.ndim == 3 else x
    high, left, right = _4d(high), _4d(left), _4d(right)
    _, _, h_h, w_h = high.shape
    half_h, half_w = h_h // 2, w_h // 2
    # interpolate expects [N,C,H,W]; treat T as N
    def _resize(x):
        c, t = x.shape[0], x.shape[1]
        x = x.permute(1, 0, 2, 3)  # [T,C,H,W]
        x = F.interpolate(x, size=(half_h, half_w), mode="bilinear", align_corners=False)
        return x.permute(1, 0, 2, 3)  # [C,T,h,w]
    left, right = _resize(left), _resize(right)
    bottom = torch.cat([left, right], dim=-1)
    return torch.cat([high, bottom], dim=-2)


# ===== GWP-style horizontal 3-view rendering (mirror episode_report build_ref_image: cam_high|cam_left|cam_right) =====
# GWP report shows each frame as the 3 views side-by-side (768x192). We render BOTH GT and pred in this exact
# layout so the report's videos "completely reference gigaworld-policy". Per-view tile = (GWP_VH, GWP_VW).
GWP_VH, GWP_VW = 192, 256  # 3 tiles -> 192 x 768, matching episode_report's --height 192 --width 768

def _resize_hwc_u8(img_hwc_u8, hw):
    """img [H,W,C] uint8 -> [hw0,hw1,C] uint8 (bilinear)."""
    import torch, torch.nn.functional as F
    t = torch.from_numpy(np.ascontiguousarray(img_hwc_u8)).permute(2, 0, 1).float().unsqueeze(0)
    t = F.interpolate(t, size=hw, mode="bilinear", align_corners=False)
    return t.squeeze(0).permute(1, 2, 0).round().clamp(0, 255).byte().numpy()

def _gwp_3view_from_raw(high_hwc, left_hwc, right_hwc):
    """3 raw camera frames [H,W,C] uint8 -> horizontal collage [GWP_VH, 3*GWP_VW, C] uint8 (cam_high|cam_left|cam_right)."""
    tiles = [_resize_hwc_u8(v, (GWP_VH, GWP_VW)) for v in (high_hwc, left_hwc, right_hwc)]
    return np.concatenate(tiles, axis=1)

def _gwp_3view_from_cosmos_concat(frame_hwc_u8):
    """Split the cosmos vertical concat (head over [left|right], head=top 2/3) back into 3 views and
    re-arrange horizontally cam_high|cam_left|cam_right -> [GWP_VH, 3*GWP_VW, C] uint8.
    Geometry mirrors _build_concat_view: rows[0:2H/3]=high(full W); rows[2H/3:]=bottom, cols[:W/2]=left, [W/2:]=right."""
    H, W = frame_hwc_u8.shape[:2]
    split = (H * 2) // 3
    high = frame_hwc_u8[:split, :, :]
    bottom = frame_hwc_u8[split:, :, :]
    left = bottom[:, : W // 2, :]
    right = bottom[:, W // 2:, :]
    return _gwp_3view_from_raw(high, left, right)


# ============================ per-episode frame cache (3-view mp4 sequential decode once) ===========
class EpisodeFrameCache:
    """Mirror eval_watch.EpisodeFrameCache: sequential-decode each episode's 3 mp4 once (no random seek)."""
    def __init__(self, val_root, cache_size=2):
        self.root = val_root; self.cap = max(1, int(cache_size))
        self.cache = {}; self.order = []
        self._vdir = os.path.join(val_root, "videos")

    def _vpath(self, cam, ep):
        for chunk in sorted(glob.glob(os.path.join(self._vdir, "*"))):
            p = os.path.join(chunk, cam, f"episode_{ep:06d}.mp4")
            if os.path.isfile(p):
                return p
        return None

    def get(self, ep):
        if ep in self.cache:
            return self.cache[ep]
        import av
        frames = {}
        for cam in VK:
            p = self._vpath(cam, ep)
            c = av.open(p)
            frames[cam] = np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])  # [L,H,W,C] uint8
            c.close()
        self.cache[ep] = frames; self.order.append(ep)
        if len(self.order) > self.cap:
            self.cache.pop(self.order.pop(0), None)
        return frames


# ============================ model wrapper (mirrors robolab RobolabPolicyService) ===========
class CosmosFoldPolicy:
    """Loads the exported Cosmos3 model and runs wam_fold policy inference per window."""

    def __init__(self, args, device="cuda"):
        import torch
        # init_script() must run before cosmos-framework runtime imports take effect
        # (mirrors action_policy_server_robolab.py:26 / export_model.py:8). Idempotent.
        from cosmos_framework.inference.common.init import init_script
        init_script()
        from cosmos_framework.inference.inference import OmniInference
        from cosmos_framework.inference.args import OmniSetupOverrides, ModelMode
        from cosmos_framework.inference.common.init import init_output_dir
        from cosmos_framework.scripts.action_policy_server_utils import (
            disable_runtime_ema_for_frozen_config, maybe_init_distributed,
        )
        from cosmos_framework.inference.action import build_action_batch
        self._torch = torch
        self._ModelMode = ModelMode
        self._build_action_batch = build_action_batch
        self.device = device
        self.args = args
        maybe_init_distributed()
        overrides = OmniSetupOverrides.model_validate({
            "checkpoint_path": args.export_dir,
            "output_dir": args.out_dir + "/_omni_tmp",
            "sampler": args.sampler,
            "guardrails": False,  # eval: skip guardrail model download (not needed, blocked offline)
        })
        setup_args = overrides.build_setup()
        init_output_dir(setup_args.output_dir)
        setup_args = disable_runtime_ema_for_frozen_config(setup_args)
        self.pipe = OmniInference.create(setup_args)
        self.model = self.pipe.model
        self.model.eval()
        self.max_action_dim = int(getattr(self.model.config, "max_action_dim", 64))
        self.q01, self.q99 = _load_action_qstats(args.stats_path)
        self._seed = 0

    def _video_from_frame(self, frame_chw01, chunk):
        """frame_chw01 torch[C,H,W] float[0,1] -> uint8 video [C,chunk+1,H,W] (first frame = obs,
        rest repeated; build_action_batch only conditions on frame 0 for POLICY/FORWARD_DYNAMICS)."""
        torch = self._torch
        img_u8 = (frame_chw01.clamp(0, 1) * 255.0).round().to(torch.uint8)  # [C,H,W]
        return img_u8.unsqueeze(1).repeat(1, chunk + 1, 1, 1)              # [C,T,H,W]

    def _decode_video(self, lat):
        """vision latent -> uint8 video [T,H,W,3] in pixel space (model.decode -> [1,C,T,H,W] in [-1,1])."""
        torch = self._torch
        video_out = self.model.decode(lat)  # [1,C,T,H,W] in [-1,1]   (robolab server:597)
        v = video_out[0].clamp(-1.0, 1.0)
        v = ((v + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 3, 0)  # [T,H,W,3]
        return v.detach().cpu().numpy()

    def infer(self, obs_concat_chw01, want_video=True):
        """Single POLICY call: condition on ONE obs frame, jointly predict model_chunk actions AND the
        dense predicted video sub-chunk (POLICY mode also generates vision, args.py:297
        condition_frame_indexes_vision=[0]). This is the unit of a free-running autoregressive rollout:
        feed the model's OWN predicted last frame back in as obs_concat_chw01 for the next sub-chunk.

        obs_concat_chw01: torch [C,H,W] float[0,1] cosmos vertical concat-view (head over [left|right]).
        Returns (pred_raw [model_chunk,14] raw joint units denormalized,
                 pred_video uint8 [Tv,H,W,3] cosmos vertical concat or None if want_video=False/decode fails).
        Tv is the model's native video sub-chunk frame count (~model_chunk+1, decoded from samples["vision"]).
        """
        torch = self._torch
        chunk = self.args.model_chunk
        video = self._video_from_frame(obs_concat_chw01, chunk)
        action = torch.zeros(chunk, self.max_action_dim, dtype=torch.float32)  # zeros (policy mode)
        data_batch = self._build_action_batch(
            video=video, action=action, raw_action_dim=14, prompt=TASK_TEXT,
            view_point="concat_view", domain_name=DOMAIN_NAME,
            model_mode=self._ModelMode.POLICY, action_chunk_size=chunk,
            fps=int(self.args.fps_cond), resolution=str(self.args.resolution),
            input_video_key=self.model.input_video_key, batch_size=1, device=self.device,
        )
        self._seed += 1
        with torch.inference_mode():
            samples = self.model.generate_samples_from_batch(
                data_batch, guidance=self.args.guidance, seed=[self._seed],
                num_steps=self.args.num_steps, shift=self.args.shift)
        act = samples["action"][0][:, :14].detach().float().cpu().numpy()  # [chunk,14] normalized
        pred_raw = _denorm_action_quantile(act, self.q01, self.q99)        # [chunk,14] raw joint
        pred_vid = None
        if want_video and samples.get("vision"):
            try:
                pred_vid = self._decode_video(samples["vision"][0])  # [Tv,H,W,3] uint8 cosmos vertical concat
            except Exception as e:
                print(f"[infer] POLICY-mode video decode failed: {e}", flush=True)
        return pred_raw, pred_vid

    def run_policy_subchunk(self, obs_concat_chw01):
        """POLICY mode (teacher-forced): condition on the GT obs frame, predict model_chunk actions.
        Returns pred_raw [model_chunk,14] in raw joint units (denormalized)."""
        torch = self._torch
        chunk = self.args.model_chunk
        video = self._video_from_frame(obs_concat_chw01, chunk)
        action = torch.zeros(chunk, self.max_action_dim, dtype=torch.float32)  # zeros (policy mode)
        data_batch = self._build_action_batch(
            video=video, action=action, raw_action_dim=14, prompt=TASK_TEXT,
            view_point="concat_view", domain_name=DOMAIN_NAME,
            model_mode=self._ModelMode.POLICY, action_chunk_size=chunk,
            fps=int(self.args.fps_cond), resolution=str(self.args.resolution),
            input_video_key=self.model.input_video_key, batch_size=1, device=self.device,
        )
        self._seed += 1
        with torch.inference_mode():
            samples = self.model.generate_samples_from_batch(
                data_batch, guidance=self.args.guidance, seed=[self._seed],
                num_steps=self.args.num_steps, shift=self.args.shift)
        act = samples["action"][0][:, :14].detach().float().cpu().numpy()  # [chunk,14] normalized
        return _denorm_action_quantile(act, self.q01, self.q99)           # [chunk,14] raw joint

    def run_policy_video_subchunk(self, obs_concat_chw01):
        """Fallback when FORWARD_DYNAMICS video decode is unavailable: run POLICY mode and decode its
        predicted vision latent (POLICY also generates video, args.py:297 condition_frame_indexes_vision=[0]).
        Returns pred_video uint8 [T,H,W,3] (cosmos vertical concat) or None."""
        torch = self._torch
        chunk = self.args.model_chunk
        video = self._video_from_frame(obs_concat_chw01, chunk)
        action = torch.zeros(chunk, self.max_action_dim, dtype=torch.float32)
        data_batch = self._build_action_batch(
            video=video, action=action, raw_action_dim=14, prompt=TASK_TEXT,
            view_point="concat_view", domain_name=DOMAIN_NAME,
            model_mode=self._ModelMode.POLICY, action_chunk_size=chunk,
            fps=int(self.args.fps_cond), resolution=str(self.args.resolution),
            input_video_key=self.model.input_video_key, batch_size=1, device=self.device,
        )
        self._seed += 1
        with torch.inference_mode():
            samples = self.model.generate_samples_from_batch(
                data_batch, guidance=self.args.guidance, seed=[self._seed],
                num_steps=self.args.num_steps, shift=self.args.shift)
        if "vision" not in samples or not samples["vision"]:
            return None
        try:
            return self._decode_video(samples["vision"][0])
        except Exception as e:
            print(f"[infer] POLICY-mode video decode failed: {e}", flush=True)
            return None

    def run_fd_subchunk(self, cur_frame_chw01, gt_action_norm_chunk):
        """FORWARD_DYNAMICS mode: feed GT actions + a (predicted-or-init) first frame, generate the
        predicted video sub-chunk. gt_action_norm_chunk: torch [model_chunk,14] NORMALIZED (quantile,
        same as the dataset). Action is padded to max_action_dim here -> build_action_batch passes it
        through verbatim (action.py:42 _load_actions does pad_action_to_max_dim for FORWARD_DYNAMICS;
        we replicate that so the batch action shape == [chunk, max_action_dim]).
        Returns pred_video uint8 [T,H,W,3] (decoded sub-chunk) or None on decode failure."""
        torch = self._torch
        from cosmos_framework.data.vfm.action.transforms import pad_action_to_max_dim
        chunk = self.args.model_chunk
        video = self._video_from_frame(cur_frame_chw01, chunk)
        action = pad_action_to_max_dim(gt_action_norm_chunk.float(), self.max_action_dim)  # [chunk,max_action_dim]
        data_batch = self._build_action_batch(
            video=video, action=action, raw_action_dim=14, prompt=TASK_TEXT,
            view_point="concat_view", domain_name=DOMAIN_NAME,
            model_mode=self._ModelMode.FORWARD_DYNAMICS, action_chunk_size=chunk,
            fps=int(self.args.fps_cond), resolution=str(self.args.resolution),
            input_video_key=self.model.input_video_key, batch_size=1, device=self.device,
        )
        self._seed += 1
        try:
            with torch.inference_mode():
                samples = self.model.generate_samples_from_batch(
                    data_batch, guidance=self.args.guidance, seed=[self._seed],
                    num_steps=self.args.num_steps, shift=self.args.shift)
            if "vision" not in samples or not samples["vision"]:
                return None
            return self._decode_video(samples["vision"][0])  # [T,H,W,3] uint8
        except Exception as e:
            print(f"[infer] FORWARD_DYNAMICS sub-chunk failed ({type(e).__name__}: {e})", flush=True)
            return None


# ============================ export DCP -> HF ===========
def maybe_export(args):
    if os.path.isdir(args.export_dir) and (
        os.path.isfile(os.path.join(args.export_dir, "config.json"))
        and (glob.glob(os.path.join(args.export_dir, "*.safetensors"))
             or os.path.isfile(os.path.join(args.export_dir, "model.safetensors.index.json")))
    ):
        print(f"[export] export dir already present, skip: {args.export_dir}", flush=True)
        return
    if not args.checkpoint_path:
        raise SystemExit(
            "[export] export dir missing and --checkpoint_path not given. Provide the iter_* DCP dir "
            "or pre-export with cosmos_framework.scripts.export_model."
        )
    # --no-vit: the action-policy decode path does not use the Qwen3-VL visual tower,
    # and the ViT-merge step needs the full Qwen3-VL-8B safetensors cached locally
    # (fails under HF_HUB_OFFLINE if any shard is missing). The model loader tolerates
    # absent language_model.visual.* keys (model.py:_DiffusersLoadPlanner). Pass
    # --export_vit to opt back in if you have the ViT shards cached.
    cmd = [sys.executable, "-m", "cosmos_framework.scripts.export_model",
           "--checkpoint-path", args.checkpoint_path,
           "--config-file", args.config_file,
           ("--vit" if args.export_vit else "--no-vit"),
           "-o", args.export_dir]
    print("[export] running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=_COSMOS3_DIR)


# ============================ planning (which episodes / windows) ===========
def plan(args):
    idx, _, info = build_window_indices(args.val_root, "exec", args.exec_horizon, args.action_chunk, args.exec_horizon)
    ep2win = OrderedDict()
    for gi in idx:
        ep2win.setdefault(info[gi][0], []).append(gi)
    eps = list(ep2win.keys())
    metric = eps[: args.n_metric_eps]
    if metric:
        viz = [metric[i] for i in np.unique(np.linspace(0, len(metric) - 1, min(args.n_viz_eps, len(metric))).astype(int))]
    else:
        viz = []
    return info, ep2win, metric, viz


# ============================ metrics helper (eval_watch-identical action mae) ===========
def _action_metrics(pred, gt, horizons):
    L = min(len(pred), len(gt))
    ae = np.abs(pred[:L] - gt[:L])
    m = {"action_mae": float(ae.mean()), "action_mse": float(((pred[:L] - gt[:L]) ** 2).mean())}
    for h in horizons:
        if h <= L:
            m[f"mae@{h}"] = float(ae[h - 1].mean())
    return m


# ============================ main worker ===========
def main():
    args = get_args()
    import torch
    # Horizons reported = GWP's {1,10,24,48} (= {1,10,action_chunk//2,action_chunk}). The WM is rolled out
    # autoregressively (n_sub = ceil(action_chunk/model_chunk) sub-chunks) so pred_action spans the full
    # action_chunk and mae@{1,10,24,48} are all measurable vs GT.
    HOR = sorted({h for h in (1, 10, args.action_chunk // 2, args.action_chunk) if 1 <= h <= args.action_chunk})
    info, ep2win, metric_eps, viz_eps = plan(args)
    if args.aggregate:
        return aggregate(args, viz_eps, HOR)

    os.makedirs(os.path.join(args.out_dir, "shards"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "episodes"), exist_ok=True)
    if args.shard_id == 0 and not args.no_export:
        maybe_export(args)

    device = "cuda"
    fc = EpisodeFrameCache(args.val_root, cache_size=4)
    lpips_fn = None
    if not args.no_lpips:
        try:
            import lpips as lpips_mod
            lpips_fn = lpips_mod.LPIPS(net="alex").to(device).eval()
        except Exception as e:
            print(f"[eval] LPIPS unavailable ({type(e).__name__}); PSNR/SSIM/temporal only", flush=True)

    policy = CosmosFoldPolicy(args, device=device)

    sid, n = args.shard_id, args.num_shards
    my_metric = metric_eps[sid::n]
    vset = set(viz_eps)
    print(f"[report shard {sid}/{n}] metric {len(my_metric)} viz {len(vset & set(my_metric))} "
          f"model_chunk={args.model_chunk} horizons={HOR}", flush=True)

    def _hwc01(fr):  # cache frame [H,W,C] uint8 -> torch [C,H,W] float[0,1]
        return torch.from_numpy(fr).permute(2, 0, 1).float() / 255.0

    def _concat_at(ep, j):
        """GT concat_view frame at absolute episode frame index j -> torch [C,H,W] float[0,1]."""
        fr = fc.get(ep); Lf = fr[VK[0]].shape[0]; j = min(int(j), Lf - 1)
        high = _hwc01(fr[VK[0]][j]); left = _hwc01(fr[VK[1]][j]); right = _hwc01(fr[VK[2]][j])
        return _build_concat_view(high, left, right)[:, 0]  # [C,H,W]

    def _gt_concat_video(ep, f):
        """Dense GT video: the action_chunk REAL frames at [f .. f+action_chunk-1] (clamped to episode
        length), each rendered to the GWP horizontal 3-view (cam_high|cam_left|cam_right) collage.
        Returns [<=action_chunk, GWP_VH, 3*GWP_VW, C] uint8. (Was 5 delta-offset keyframes; now dense,
        to align frame-for-frame with the free-running dense pred rollout.)"""
        fr = fc.get(ep); Lf = fr[VK[0]].shape[0]
        ck = args.action_chunk
        out = []
        for o in range(ck):
            j = min(f + o, Lf - 1)
            out.append(_gwp_3view_from_raw(fr[VK[0]][j], fr[VK[1]][j], fr[VK[2]][j]))
        return np.stack(out)  # [<=action_chunk, GWP_VH, 3*GWP_VW, C] uint8

    import math as _math
    n_sub = _math.ceil(args.action_chunk / args.model_chunk)  # sub-chunks to span action_chunk (48/16 -> 3)

    def infer_window(gi, want_video):
        """WM rollout to GWP's action_chunk horizon (=48) from the native model_chunk (=16), n_sub sub-chunks.

        FREE mode (default, --rollout free): a single closed-loop ("imagination") rollout. Only the FIRST
        sub-chunk sees the real GT observation (frame f, same as GWP's single 48-step rollout that only
        sees frame 0). Each subsequent sub-chunk conditions on the model's OWN predicted last frame. One
        POLICY call per sub-chunk yields BOTH the action sub-chunk AND its dense predicted video sub-chunk,
        so action AND video are jointly the policy's own free-running prediction (fair vs GWP's single
        48-step rollout — no GT injected mid-rollout). pred_action = concat(acts)[:action_chunk];
        pred_video_concat = concatenate(vids) -> dense (~action_chunk frames spanning the 48-step horizon).

        TEACHER mode (--rollout teacher): the old GT-anchored upper-bound (RTC deployment). Action: each
        sub-chunk re-anchors on the real GT obs at f+i*model_chunk (teacher-forced POLICY). Video:
        autoregressive FORWARD_DYNAMICS fed the GT (normalized) action chunk, observations free-run.

        Returns (ep, f, pred_raw[:action_chunk,14] raw, gt[action_chunk,14] raw,
                 pred_v dense GWP-3view [Tp,GWP_VH,3*GWP_VW,C] uint8 or None,
                 gt_v  dense GWP-3view [Tg,GWP_VH,3*GWP_VW,C] uint8 or None)."""
        ep, f = info[int(gi)]
        mc = args.model_chunk
        # GT action chunk from parquet (absolute 14-D joint), aligned to window start f.
        gt = _read_gt_action(args.val_root, ep, f, args.action_chunk)  # [action_chunk,14]

        if args.rollout == "free":
            # ---- FREE-RUNNING autoregressive rollout (action + video jointly, policy mode) ----
            cur = _concat_at(ep, f)  # [C,H,W] float[0,1] cosmos concat — only frame 0 is ground truth
            acts, vids = [], []
            for i in range(n_sub):
                a, v = policy.infer(cur, want_video=want_video)  # a:[mc,14] raw ; v:[Tv,H,W,3] uint8 concat or None
                acts.append(a)
                if want_video:
                    if v is None or len(v) == 0:
                        break  # decode failed mid-rollout: keep whatever dense frames we have
                    vids.append(v)
                    # free-run: feed the model's OWN last predicted frame as the next anchor. The model
                    # output is the cosmos VERTICAL concat (head over [left|right]); we need it back as a
                    # [C,H,W] float[0,1] concat-view for the next policy.infer first-frame.
                    last = v[-1]  # [H,W,3] uint8 cosmos vertical concat
                    cur = torch.from_numpy(np.ascontiguousarray(last)).permute(2, 0, 1).float() / 255.0
            pred_raw = np.concatenate(acts, axis=0)[: args.action_chunk]  # [action_chunk,14] raw joint
            pred_v, gt_v = None, None
            if want_video and vids:
                full_pred = np.concatenate(vids, axis=0)  # [sum Tv, H, W, 3] dense, ~action_chunk frames
                pred_v = np.stack([_gwp_3view_from_cosmos_concat(k) for k in full_pred])  # dense GWP 3-view
                gt_v = _gt_concat_video(ep, f)  # dense action_chunk REAL frames as GWP 3-view
            return ep, f, pred_raw, gt, pred_v, gt_v

        # ---- TEACHER (GT-anchored) rollout: RTC deployment upper-bound ----
        # ACTION (teacher-forced POLICY)
        pred_subs = []
        for i in range(n_sub):
            obs = _concat_at(ep, f + i * mc)
            pred_subs.append(policy.run_policy_subchunk(obs))  # [mc,14] raw
        pred_raw = np.concatenate(pred_subs, axis=0)[: args.action_chunk]  # [action_chunk,14] raw joint

        pred_v, gt_v = None, None
        if want_video:
            gt_v = _gt_concat_video(ep, f)  # dense action_chunk REAL frames as GWP 3-view

            # VIDEO rollout (autoregressive FORWARD_DYNAMICS, GT actions normalized)
            gt_norm = _norm_action_quantile(gt, policy.q01, policy.q99)  # [action_chunk,14] in [-1,1]
            cur_frame = _concat_at(ep, f)  # init = GT concat first frame
            fd_frames = []
            fd_ok = True
            for i in range(n_sub):
                a0 = i * mc
                sub = gt_norm[a0: a0 + mc]                  # [<=mc,14] normalized
                if sub.shape[0] < mc:                       # pad short tail (last window) by repeating last step
                    sub = np.concatenate([sub, np.repeat(sub[-1:], mc - sub.shape[0], axis=0)], axis=0)
                a_t = torch.from_numpy(np.ascontiguousarray(sub)).float()  # [mc,14] -> padded inside run_fd_subchunk
                sub_vid = policy.run_fd_subchunk(cur_frame, a_t)  # [Tv,H,W,3] uint8 (cosmos vertical concat) or None
                if sub_vid is None or len(sub_vid) == 0:
                    fd_ok = False
                    break
                fd_frames.append(sub_vid)
                # autoregress: next sub-chunk conditions on the LAST predicted frame (free-run observations).
                last = sub_vid[-1]  # [H,W,3] uint8 cosmos vertical concat
                cur_frame = torch.from_numpy(np.ascontiguousarray(last)).permute(2, 0, 1).float() / 255.0
            if fd_ok and fd_frames:
                full_pred = np.concatenate(fd_frames, axis=0)  # [sum Tv, H, W, 3] dense over action_chunk span
                pred_v = np.stack([_gwp_3view_from_cosmos_concat(k) for k in full_pred])  # dense GWP 3-view
            else:
                # Fallback: POLICY-mode pred video (single sub-chunk decode at f) so the report still renders.
                print(f"[infer] FORWARD_DYNAMICS video unavailable for ep{ep} f{f}; "
                      f"falling back to POLICY-mode pred video", flush=True)
                pol_vid = policy.run_policy_video_subchunk(_concat_at(ep, f))  # [Tv,H,W,3] uint8 or None
                if pol_vid is not None and len(pol_vid) > 0:
                    pred_v = np.stack([_gwp_3view_from_cosmos_concat(k) for k in pol_vid])
        return ep, f, pred_raw, gt, pred_v, gt_v

    # ---- optional latency probe (shard 0) ----
    latency = {}
    if sid == 0 and my_metric:
        g0 = ep2win[my_metric[0]][0]
        for _ in range(1):
            infer_window(g0, False)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(3):
            infer_window(g0, False)
        torch.cuda.synchronize(); latency["action_ms"] = (time.time() - t) / 3 * 1000
        try:
            torch.cuda.synchronize(); t = time.time()
            for _ in range(2):
                infer_window(g0, True)
            torch.cuda.synchronize(); latency["video_ms"] = (time.time() - t) / 2 * 1000
        except Exception as e:
            print(f"[latency] video probe failed: {e}", flush=True)

    rows = {}
    for k, ep in enumerate(my_metric):
        wins = ep2win[ep]; is_v = ep in vset
        em = {kk: [] for kk in ["action_mae", "action_mse"] + [f"mae@{h}" for h in HOR]}
        vid_keys = ["psnr", "ssim", "temporal_absdiff_ratio"] + (["lpips"] if lpips_fn is not None else [])
        vm = {kk: [] for kk in vid_keys}
        if not is_v:
            ws = wins if len(wins) <= args.max_win_per_ep else [
                wins[i] for i in np.unique(np.linspace(0, len(wins) - 1, args.max_win_per_ep).astype(int))]
            for gi in ws:
                _, _, pa, gt, _, _ = infer_window(gi, False)
                for kk, v in _action_metrics(pa, gt, HOR).items():
                    em.setdefault(kk, []).append(v)
            rows[int(ep)] = {"ep": int(ep), "n_win": len(wins),
                             **{kk: float(np.mean(em[kk])) for kk in em if em[kk]}}
        else:
            tp, tg = {}, {}; gt_ep, raw_ep = [], []; bvids = []
            # Each viz window now costs n_sub action + up to n_sub video inferences; cap the viz
            # episode's full-video to --max_full_windows evenly-spaced windows so it doesn't run
            # hundreds of inferences (a 600-frame episode at exec_horizon 16 has ~35 windows).
            vwins = wins if len(wins) <= args.max_full_windows else [
                wins[i] for i in np.unique(np.linspace(0, len(wins) - 1, args.max_full_windows).astype(int))]
            vid_wins = set(vwins[:: max(1, len(vwins) // max(1, args.n_vid_per_ep))][: args.n_vid_per_ep])
            for gi in vwins:
                _, f, pa, gt, pv, gv = infer_window(gi, True)
                for kk, v in _action_metrics(pa, gt, HOR).items():
                    em.setdefault(kk, []).append(v)
                if pv is not None and gv is not None:
                    gt_t = torch.from_numpy(gv).to(device, torch.float32)
                    pv_t = torch.from_numpy(pv).to(device, torch.float32)
                    for kk, v in video_metrics_gpu(pv_t, gt_t, lpips_fn=lpips_fn, device=device).items():
                        vm.setdefault(kk, []).append(v)
                # De-overlapped action trajectory: each window contributes exactly exec_horizon steps
                # (stride == exec_horizon) -> a continuous stitch. pred_raw spans action_chunk(48) >= exec_horizon.
                h = args.exec_horizon
                tp[f] = pa[:h].tolist(); tg[f] = gt[:h].tolist()
                if pv is not None and gv is not None:
                    gt_ep.append(gv); raw_ep.append(pv)
                if gi in vid_wins and pv is not None and gv is not None:
                    bp = f"episodes/ep{ep}_w{f}.mp4"
                    try:
                        _save_2row(gv, pv, os.path.join(args.out_dir, bp), args.fps); bvids.append(bp)
                    except Exception as e:
                        print(f"[viz] window mp4 failed ep{ep} w{f}: {e}", flush=True)
            full = None
            if args.full_video and gt_ep:
                full = f"episodes/ep{ep}_full.mp4"
                try:
                    _save_2row(np.concatenate(gt_ep), np.concatenate(raw_ep), os.path.join(args.out_dir, full), args.fps)
                except Exception as e:
                    print(f"[viz] full mp4 failed ep{ep}: {e}", flush=True); full = None
            tpng = None
            if tp:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fs = sorted(tp.keys())
                P = np.concatenate([np.array(tp[f]) for f in fs]); G = np.concatenate([np.array(tg[f]) for f in fs])
                x = np.arange(len(P)); fig, axes = plt.subplots(7, 2, figsize=(13, 15)); axes = axes.flatten()
                for dd in range(14):
                    axes[dd].plot(x, G[:, dd], "k--", lw=1.5, label="GT")
                    axes[dd].plot(x, P[:, dd], "r-", lw=1.2, label="pred")
                    axes[dd].set_title(DIM[dd], fontsize=8)
                    if dd == 0:
                        axes[dd].legend(fontsize=7)
                fig.suptitle(f"episode {ep} — deploy-style action traj (exec_h={args.exec_horizon})")
                tpng = f"episodes/ep{ep}_traj.png"
                fig.savefig(os.path.join(args.out_dir, tpng), dpi=70, bbox_inches="tight"); plt.close(fig)
            rows[int(ep)] = {"ep": int(ep), "n_win": len(wins),
                             **{kk: float(np.mean(em[kk])) for kk in em if em[kk]},
                             **{kk: float(np.mean(vm[kk])) for kk in vm if vm[kk]},
                             "traj_png": tpng, "vids": bvids, "full_video": full}
        if k % 2 == 0:
            print(f"[shard {sid}] {k+1}/{len(my_metric)} ep{ep}{' viz' if is_v else ''}", flush=True)

    del policy
    torch.cuda.empty_cache()
    json.dump({"metric": rows, "latency": latency},
              open(os.path.join(args.out_dir, "shards", f"shard_{sid}.json"), "w"))
    print(f"[report shard {sid}] done -> shards/shard_{sid}.json", flush=True)
    if args.num_shards == 1:
        aggregate(args, viz_eps, HOR)


def _read_gt_action(val_root, ep, start, action_chunk):
    """Read absolute 14-D joint GT action [chunk,14] from the per-episode parquet (wam_fold layout)."""
    import pyarrow.parquet as pq
    path = None
    for chunk in sorted(glob.glob(os.path.join(val_root, "data", "chunk-*"))):
        cand = os.path.join(chunk, f"episode_{ep:06d}.parquet")
        if os.path.isfile(cand):
            path = cand; break
    if path is None:
        raise FileNotFoundError(f"no parquet for episode {ep} under {val_root}/data")
    table = pq.read_table(path)
    acts = table.column("action").to_pylist()[start: start + action_chunk]
    return np.asarray(acts, dtype=np.float32)[:, :14]


# ============================ aggregate -> report.html (mirrors episode_report.aggregate) ===========
def aggregate(args, viz_eps, HOR):
    sh = sorted(glob.glob(os.path.join(args.out_dir, "shards", "shard_*.json")))
    metric, lat = {}, {}
    for f in sh:
        d = json.load(open(f))
        metric.update({int(k): v for k, v in d["metric"].items()})
        lat.update(d.get("latency", {}))
    print(f"[aggregate] merged {len(sh)} shards, {len(metric)} ep", flush=True)
    agg = {f"mae@{h}": float(np.mean([r[f"mae@{h}"] for r in metric.values() if r.get(f"mae@{h}") is not None]))
           for h in HOR if any(r.get(f"mae@{h}") is not None for r in metric.values())}
    vid_keys = ["psnr", "ssim", "temporal_absdiff_ratio", "lpips"]
    vagg = {kk: float(np.mean([r[kk] for r in metric.values() if r.get(kk) is not None]))
            for kk in vid_keys if any(r.get(kk) is not None for r in metric.values())}
    gwp, gwp_vid, tau0 = _load_baselines(args)

    def b64(p):
        if not p:
            return ""
        fp = os.path.join(args.out_dir, p)
        return "data:image/png;base64," + base64.b64encode(open(fp, "rb").read()).decode() if os.path.isfile(fp) else ""

    # ---- per-episode viz blocks (same structure as episode_report) ----
    blocks = []
    for ep in viz_eps:
        r = metric.get(ep)
        if not r or not r.get("traj_png"):
            continue
        full = (f'<p class=note><b>A full-episode video</b> (concat over all windows):</p>'
                f'<video src="{r["full_video"]}" controls width="820"></video>') if r.get("full_video") else ""
        bvs = "".join(f'<video src="{v}" controls width="400"></video>' for v in r.get("vids", []))
        bsec = f'<p class=note><b>B representative-window videos</b> (1s each):</p><div class=vids>{bvs}</div>' if bvs else ""
        m1 = r.get("mae@1"); mlast = r.get(f"mae@{max(HOR)}")
        summ = f'ep {ep} &nbsp;|&nbsp; n_win={r["n_win"]}'
        if m1 is not None:
            summ += f' &nbsp; mae@1={m1:.4f}'
        if mlast is not None:
            summ += f' &nbsp; mae@{max(HOR)}={mlast:.4f}'
        blocks.append(
            f'<details><summary>{summ}</summary>'
            f'<p class=note>action curve (pred vs GT, deploy-style concat over exec_horizon):</p>'
            f'<img src="{b64(r["traj_png"])}" width="920">{full}{bsec}</details>')

    # ---- 3-way comparison table: action mae@horizon ----
    def _cell(v):
        return f"{v:.4f}" if v is not None else "—"
    # tau0 reports mae@{1,10,16,33} (chunk 32); map to closest displayed horizon.
    tau0_map = {1: tau0.get("mae@1"), 10: tau0.get("mae@10"), 24: tau0.get("mae@25", tau0.get("mae@33")),
                48: tau0.get("mae@33")}
    head_hor = sorted(set(list(PI05.keys()) + HOR))
    th = "".join(f"<th>mae@{h}</th>" for h in head_hor)
    def _row(name, getter, note=""):
        cells = "".join(f"<td>{_cell(getter(h))}</td>" for h in head_hor)
        return f"<tr><td style='text-align:left'>{name}{note}</td>{cells}</tr>"
    cosmos_get = lambda h: agg.get(f"mae@{h}") if h in HOR else None
    cosmos_note = (f" (adapted, {args.num_steps}-step, native chunk={args.model_chunk} "
                   f"rolled out to {args.action_chunk})")
    table_action = (
        f"<table><tr><th style='text-align:left'>model</th>{th}</tr>"
        + _row("<b>Cosmos3-Nano-Policy</b>", cosmos_get, cosmos_note)
        + _row("GigaWorld-Policy (best)", lambda h: gwp.get(h))
        + _row(f"tau-0-wm ({args.tau0_tag})", lambda h: tau0_map.get(h))
        + _row("π0.5 reference", lambda h: PI05.get(h))
        + "</table>")

    # ---- 3-way comparison table: video metrics ----
    def _vcell(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "—"
    tv = (f"<table><tr><th style='text-align:left'>model</th><th>PSNR</th><th>SSIM</th>"
          f"<th>temporal_absdiff_ratio</th><th>LPIPS</th></tr>"
          f"<tr><td style='text-align:left'><b>Cosmos3-Nano-Policy</b></td>"
          f"<td>{_vcell(vagg.get('psnr'))}</td><td>{_vcell(vagg.get('ssim'))}</td>"
          f"<td>{_vcell(vagg.get('temporal_absdiff_ratio'))}</td><td>{_vcell(vagg.get('lpips'))}</td></tr>"
          f"<tr><td style='text-align:left'>tau-0-wm ({args.tau0_tag})</td>"
          f"<td>{_vcell(tau0.get('psnr'))}</td><td>{_vcell(tau0.get('ssim'))}</td>"
          f"<td>{_vcell(tau0.get('temporal_absdiff_ratio'))}</td><td>{_vcell(tau0.get('lpips'))}</td></tr>"
          f"</table>")

    la = (f"action-only <b>{lat.get('action_ms', 0):.0f} ms</b> · with-video {lat.get('video_ms', 0):.0f} ms "
          f"· {args.num_steps} denoise steps · guidance {args.guidance}")
    name = "Cosmos3-Nano-Policy (wam_fold adapted)"
    html = f"""<!doctype html><html><head><meta charset=utf-8><title>Episode Report — {name}</title>{CSS}</head><body>
<h2>Episode Report — {name}</h2>
<p class=note>held-out: metric {len(metric)} ep / viz {len([e for e in viz_eps if e in metric])} ep · exec_horizon={args.exec_horizon} · model_chunk={args.model_chunk} · <b>open-loop (not closed-loop SR)</b></p>
<h3>Video legend</h3><ul>
<li><b>Rollout:</b> {args.rollout}-running autoregressive rollout — the <b>full dense rollout</b> is shown (≈{args.action_chunk} frames/window, the policy's own closed-loop imagination; only frame 0 is GT, fair vs GWP's single {args.action_chunk}-step rollout)</li>
<li>Each video has <b>2 rows</b> (row name burned in): <b>row1 GT (ground truth) / row2 pred (Cosmos3 decoded)</b>, horizontal 3-view</li>
<li>Each row shows <b>3 views side-by-side: cam_high (head) | cam_left_wrist | cam_right_wrist</b> (GWP build_ref_image layout; pred's cosmos concat is split back into the 3 views)</li>
<li><b>A full-episode video</b> (<code>ep*_full.mp4</code>): GWP overlapping-window concatenation — every window's full dense clip stitched end-to-end (2-row GT/pred horizontal 3-view)</li>
<li><b>B representative-window videos</b> (<code>ep*_w*.mp4</code>): {args.n_vid_per_ep} windows, each the full dense clip (≈{args.action_chunk} frames)</li></ul>
<h3>Inference performance</h3><p>{la}</p>
<h3>Aggregate action MAE — 3-way comparison (all {len(metric)} ep)</h3>
<p class=note>Cosmos3 is an <b>absolute</b> 14-D joint policy (no delta-mask / add-state); MAE is in raw joint units, directly comparable to GWP/tau0/π0.5. The native {args.model_chunk}-step model is rolled out to the full {args.action_chunk}-step horizon in <b>{args.rollout}</b> mode ({"free-running autoregressive: only frame 0 is GT, each sub-chunk conditions on the policy's OWN predicted last frame — fair vs GWP's single 48-step rollout" if args.rollout == "free" else "teacher-forced: each sub-chunk re-anchored on the real GT obs (RTC deployment upper-bound)"}), so mae@{{1,10,24,48}} are all measured.</p>
{table_action}
<h3>Aggregate video metrics</h3>
<p class=note>PSNR/SSIM/temporal_absdiff_ratio/LPIPS via eval_watch.video_metrics_gpu on the decoded concat_view vs GT concat_view (resized to pred resolution). Computed on the <b>dense</b> aligned frames (min(len(pred_dense), action_chunk)) — more frames give a better estimate; this differs from GWP's 5-keyframe basis.</p>
{tv}
<h3>Per-episode (sampled viz)</h3>{''.join(blocks)}
</body></html>"""
    open(os.path.join(args.out_dir, "report.html"), "w").write(html)
    json.dump({"n_metric_eps": len(metric), "latency": lat, "model_chunk": args.model_chunk,
               "raw_mae": {h: agg.get(f"mae@{h}") for h in HOR}, "video": vagg, "pi05": PI05,
               "gwp": gwp, "tau0_tag": args.tau0_tag},
              open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)
    print("[aggregate] cosmos mae@: " + " ".join(f"@{h} {agg.get(f'mae@{h}', float('nan')):.4f}" for h in HOR)
          + f" -> {args.out_dir}/report.html", flush=True)


# ============================ args ===========
def get_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val_root", default="/mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/wam_fold_v1/visrobot01_val")
    ap.add_argument("--out_dir", default="/mnt/pfs/p46h4f/cosmos/wam_fold_policy_runs/reports")
    ap.add_argument("--export_dir", default="/mnt/pfs/p46h4f/cosmos/wam_fold_policy_runs/exported/Cosmos3-Nano-Policy-wam_fold")
    ap.add_argument("--checkpoint_path", default="",
                    help="iter_* DCP dir to export (only needed if export_dir missing)")
    ap.add_argument("--config_file",
                    default="/mnt/pfs/p46h4f/cosmos/wam_fold_policy_runs/train_out_2node/cosmos3/action/wam_fold_nano/config.yaml")
    ap.add_argument("--no_export", action="store_true", help="never run export (assume export_dir is ready)")
    ap.add_argument("--export_vit", action="store_true",
                    help="export the Qwen3-VL ViT tower too (needs full Qwen3-VL-8B shards cached); default --no-vit")
    ap.add_argument("--stats_path", default="/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_policy/data/stats/visrobot01.json")
    # comparison baselines
    ap.add_argument("--gwp_summary",
                    default="/mnt/pfs/p46h4f/cosmos/deepdive_kai0/giga_world_policy/runs/visrobot01_fold_aihc_latent_5x/report_step50000/summary.json")
    ap.add_argument("--tau0_json", default="/mnt/pfs/p46h4f/cosmos/deepdive_kai0/tau-0-wm/runs/eval_gigaworld.json")
    ap.add_argument("--tau0_tag", default="p3_final")
    # eval sizing — each window now costs n_sub (=ceil(action_chunk/model_chunk)=3) action inferences
    # + up to n_sub video inferences (autoregressive WM rollout), so keep these small by default.
    ap.add_argument("--n_metric_eps", type=int, default=20)
    ap.add_argument("--n_viz_eps", type=int, default=10)
    ap.add_argument("--n_vid_per_ep", type=int, default=3)
    ap.add_argument("--max_win_per_ep", type=int, default=6,
                    help="metric (non-viz) episodes: subsample to this many windows")
    ap.add_argument("--max_full_windows", type=int, default=12,
                    help="viz episodes: cap the full-video / metric'd windows to this many (each costs up to "
                         "2*n_sub WM inferences) so a viz episode doesn't run hundreds of inferences")
    ap.add_argument("--full_video", type=int, default=1)
    ap.add_argument("--exec_horizon", type=int, default=16, help="GWP window stride (overlapping; < action_chunk)")
    ap.add_argument("--action_chunk", type=int, default=48, help="GT/report horizon window (for the table)")
    ap.add_argument("--model_chunk", type=int, default=16, help="action steps the model predicts (trained chunk_length=16)")
    ap.add_argument("--resolution", default="480", help="cosmos action transform resolution bucket (trained: 480)")
    ap.add_argument("--fps_cond", type=int, default=30, help="conditioning_fps (trained: 30)")
    ap.add_argument("--num_steps", type=int, default=10)
    ap.add_argument("--guidance", type=float, default=0.0)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--sampler", default="unipc", choices=["unipc", "edm"])
    ap.add_argument("--fps", type=int, default=24,
                    help="output mp4 fps (dense rollout -> near real-time per window; was 5 for the old "
                         "5-keyframe basis)")
    ap.add_argument("--rollout", choices=["free", "teacher"], default="free",
                    help="free: free-running autoregressive rollout (default; only frame 0 is GT, fair vs "
                         "GWP single 48-step rollout — action+video are the policy's own imagination). "
                         "teacher: old GT-anchored upper-bound (RTC deployment; sub-chunk i re-anchored on "
                         "GT obs at f+i*model_chunk, video FORWARD_DYNAMICS fed GT actions).")
    ap.add_argument("--no_lpips", action="store_true")
    # sharding / aggregate
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--aggregate", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    main()
