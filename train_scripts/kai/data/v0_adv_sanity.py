"""V0 sanity for the AWBC traditional route: run the trained Advantage Estimator on ONE episode
and check the advantage DIRECTION (does absolute_value track task progress?).

Our data has no stage_progress_gt → use frame_index/(N-1) as the progress proxy (like the ViVa V0).
Reuses eval_adv_est.py's exact AdvantageEstimator inference (sample_values, 6-image obs).

Usage: kai0/.venv/bin/python train_scripts/kai/data/v0_adv_sanity.py [ep_idx] [dataset_dir]
"""
import sys, os
sys.path.insert(0, "kai0/src")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import dataclasses, numpy as np, pandas as pd, torch, safetensors.torch, cv2
from types import SimpleNamespace
from openpi.training import config as _config
from openpi.models_pytorch.pi0_pytorch import AdvantageEstimator
import openpi.models.tokenizer as _tokenizer
from openpi.shared import image_tools

EP = int(sys.argv[1]) if len(sys.argv) > 1 else 400
DS = sys.argv[2] if len(sys.argv) > 2 else "kai0/data/Task_A/self_built/A_smooth800_dagger_full"
CKPT = "kai0/checkpoints/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/adv_est_v1/100000"
dev = torch.device("cuda:0")

cfg = _config.get_config("ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD")
model = AdvantageEstimator(dataclasses.replace(cfg.model)).to(dev).eval()
safetensors.torch.load_model(model, f"{CKPT}/model.safetensors", strict=True)
tok = _tokenizer.PaligemmaTokenizer(cfg.model.max_token_len)
tokens, masks = tok.tokenize("Flatten and fold the cloth.", state=None)
print(f"[load] AdvantageEstimator + ckpt step 100000 OK", flush=True)


def load_video(p):
    cap = cv2.VideoCapture(p); fr = []
    while True:
        ret, f = cap.read()
        if not ret: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release(); return fr


def proc(rgb):
    t = torch.from_numpy(rgb).float() / 255.0 * 2 - 1
    return image_tools.resize_with_pad_torch(t, 224, 224).permute(2, 0, 1)


def to_t(imgs):
    return torch.stack([proc(i) for i in imgs]).to(dev)


b = f"{DS}/videos/chunk-000"
top = load_video(f"{b}/observation.images.top_head/episode_{EP:06d}.mp4")
left = load_video(f"{b}/observation.images.hand_left/episode_{EP:06d}.mp4")
right = load_video(f"{b}/observation.images.hand_right/episode_{EP:06d}.mp4")
N = min(len(top), len(left), len(right))
print(f"[data] ep{EP} frames={N} (dataset={DS.split('/')[-1]})", flush=True)
init = (proc(top[0]).unsqueeze(0).to(dev), proc(left[0]).unsqueeze(0).to(dev), proc(right[0]).unsqueeze(0).to(dev))

abs_vals = []
bs = 48
for s in range(0, N, bs):
    e = min(s + bs, N); nb = e - s; idx = list(range(s, e))
    cur = (to_t([top[i] for i in idx]), to_t([left[i] for i in idx]), to_t([right[i] for i in idx]))
    tb = torch.from_numpy(np.tile(tokens[None], (nb, 1))).to(dev)
    mb = torch.from_numpy(np.tile(masks[None], (nb, 1))).to(dev)
    st = torch.zeros((nb, 32), dtype=torch.float32, device=dev)
    obs = SimpleNamespace(state=st, images={
        "base_-100_rgb": init[0].expand(nb, -1, -1, -1), "left_wrist_-100_rgb": init[1].expand(nb, -1, -1, -1),
        "right_wrist_-100_rgb": init[2].expand(nb, -1, -1, -1),
        "base_0_rgb": cur[0], "left_wrist_0_rgb": cur[1], "right_wrist_0_rgb": cur[2]},
        image_masks={}, tokenized_prompt=tb, tokenized_prompt_mask=mb)
    with torch.no_grad():
        v = model.sample_values(dev, obs).cpu().numpy().reshape(-1)
    abs_vals.extend(v.tolist())

abs_vals = np.array(abs_vals[:N])
prog = np.arange(N) / max(1, N - 1)
corr = float(np.corrcoef(abs_vals, prog)[0, 1])
direction = "递增=value≈进度 ✅" if corr > 0.3 else ("递减=value≈剩余比例(advantage 须翻符号) ✅" if corr < -0.3 else "弱/噪声 ⚠️")
print(f"\n===== V0 RESULT ep{EP} =====")
print(f"absolute_value: start={abs_vals[0]:.3f}  q25={abs_vals[int(N*.25)]:.3f}  mid={abs_vals[N//2]:.3f}  q75={abs_vals[int(N*.75)]:.3f}  end={abs_vals[-1]:.3f}")
print(f"corr(absolute_value, frame_progress) = {corr:+.3f}  -> {direction}")
print(f"|corr|>=0.5 (好 episode)? {'YES' if abs(corr)>=0.5 else 'NO'}")
