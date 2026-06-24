"""Joint-14 dataloader for wam_fold_v1 (tau0 fine-tuning).

Proprio convention (verified on data):
  - parquet `observation.state`[t] and `action`[t] are 14-dim ABSOLUTE joints; action[t]==state[t].
  - the policy predicts the future trajectory: action_chunk = action[t : t+H] (== future states).
  - delta transform (matches GigaWorld / inference_server.add_state_to_action):
        for delta_mask dims:  a_delta[k] = action[t+k] - state[t]
        for gripper dims (0): a_delta[k] = action[t+k]            (absolute)
  - normalize with tau0 statistics_{emb}.json: (x - mean)/std.

Video path: pluggable. `mode='latent'` reads GigaWorld-style cached vae_latent/*.pt
({stride,starts,visual,ref}); `mode='frames'` decodes mp4 (needs tau0 VAE at train time).
This module fully implements + self-tests the proprio path; the video path is an interface.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DELTA_MASK = np.array([1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0], dtype=bool)  # False = gripper = absolute


def _load_stats(path):
    d = json.load(open(path))
    a, s = d["action"], d["state"]
    return {
        "a_mean": np.array(a["mean"], np.float32), "a_std": np.array(a["std"], np.float32) + 1e-6,
        "s_mean": np.array(s["mean"], np.float32), "s_std": np.array(s["std"], np.float32) + 1e-6,
        "delta_mask": np.array(d.get("_meta", {}).get("delta_mask", DELTA_MASK.astype(int)), bool),
    }


def proprio_sample(state_t, action_chunk_abs, stats):
    """state_t: (14,) abs joints at t. action_chunk_abs: (H,14) abs future actions.
    Returns (norm_state (1,14), norm_action (H,14)) as float32 — delta'd + normalized."""
    dm = stats["delta_mask"]
    a = action_chunk_abs.astype(np.float32).copy()
    a[:, dm] = a[:, dm] - state_t[dm]                 # delta for non-gripper dims
    a = (a - stats["a_mean"]) / stats["a_std"]        # normalize action
    s = (state_t.astype(np.float32) - stats["s_mean"]) / stats["s_std"]
    return s[None, :], a


def undo_proprio_action(norm_action, state_t, stats):
    """Inverse of proprio_sample's action transform (for deployment parity check)."""
    dm = stats["delta_mask"]
    a = norm_action * stats["a_std"] + stats["a_mean"]
    a[:, dm] = a[:, dm] + state_t[dm]
    return a


class LeRobotJointDataset(Dataset):
    def __init__(self, data_path, stats_path, action_chunk=33, embodiment="visrobot01",
                 embed_id=0, video_mode="latent", min_tail=1):
        self.data_path = data_path
        self.stats = _load_stats(stats_path)
        self.H = action_chunk
        self.embodiment = embodiment
        self.embed_id = embed_id
        self.video_mode = video_mode
        # index (episode_parquet, start_frame) windows
        self.parquets = sorted(glob.glob(os.path.join(data_path, "data", "chunk-*", "episode_*.parquet")))
        self._index = []  # (pq_path, n_frames)
        meta = {}
        ep_jsonl = os.path.join(data_path, "meta", "episodes.jsonl")
        if os.path.exists(ep_jsonl):
            for line in open(ep_jsonl):
                j = json.loads(line)
                meta[j["episode_index"]] = j
        self.meta = meta
        for p in self.parquets:
            self._index.append(p)

    def __len__(self):
        return len(self._index)

    def load_proprio(self, pq_path, t=None):
        df = pd.read_parquet(pq_path, columns=["observation.state", "action"])
        s = np.stack(df["observation.state"].values).astype(np.float32)
        a = np.stack(df["action"].values).astype(np.float32)
        n = len(s)
        if t is None:
            t = np.random.randint(0, max(1, n - 1))
        # future action chunk, clamp+pad-by-repeat at episode end
        idx = np.clip(np.arange(t, t + self.H), 0, n - 1)
        chunk = a[idx]
        ns, na = proprio_sample(s[t], chunk, self.stats)
        return dict(state=torch.from_numpy(ns), action=torch.from_numpy(na),
                    embed_id=self.embed_id, t=t, n=n, pq=pq_path)

    def __getitem__(self, i):
        return self.load_proprio(self._index[i])


class LatentJointDataset(Dataset):
    """Reuses GigaWorld vae_latent + t5 caches (verified VAE-identical to tau0).

    Per __getitem__: pick episode -> random window i -> returns
      video_latent z0 = visual[i]   [C=48, T=2, h=12, W=48(views concat)]
      ref           = ref[i]        [C=48, T=1, h=12, W=48]   (frame-0 conditioning)
      state (1,14), action (33,14)  (delta+normalized, aligned to starts[i])
      t5            [L,4096]
    Episode-grouped (one window/episode per draw); LRU-cached episode tensors.
    """
    def __init__(self, data_path, stats_path, action_chunk=33, embed_id=0, lru=8, latent_subdir=None):
        self.data_path = data_path
        self.stats = _load_stats(stats_path)
        self.H = action_chunk
        self.embed_id = embed_id
        self.lru = lru
        self._cache = {}
        self._order = []
        latent_subdir = latent_subdir or os.environ.get("TAU0_LATENT_DIR", "vae_latent")
        lat_dir = os.path.join(data_path, latent_subdir)
        self.lat_files = sorted(glob.glob(os.path.join(lat_dir, "episode_*.pt")))
        # map episode index -> (parquet, t5) by filename stem
        self.entries = []
        for lf in self.lat_files:
            stem = os.path.basename(lf).replace(".pt", "")          # episode_000001
            idx = int(stem.split("_")[1])
            pq = os.path.join(data_path, "data", f"chunk-{idx // 1000:03d}", f"{stem}.parquet")
            t5 = os.path.join(data_path, "t5_embedding", f"{stem}.pt")
            if os.path.exists(pq) and os.path.exists(t5):
                self.entries.append((lf, pq, t5))

    def __len__(self):
        return len(self.entries)

    def _load_episode(self, i):
        if i in self._cache:
            return self._cache[i]
        lf, pq, t5 = self.entries[i]
        lat = torch.load(lf, map_location="cpu")
        df = pd.read_parquet(pq, columns=["observation.state", "action"])
        s = np.stack(df["observation.state"].values).astype(np.float32)
        a = np.stack(df["action"].values).astype(np.float32)
        t5e = torch.load(t5, map_location="cpu").float()
        obj = dict(visual=lat["visual"], ref=lat["ref"], starts=list(lat["starts"]), s=s, a=a, t5=t5e)
        self._cache[i] = obj
        self._order.append(i)
        if len(self._order) > self.lru:
            old = self._order.pop(0)
            self._cache.pop(old, None)
        return obj

    def __getitem__(self, i):
        ep = self._load_episode(i)
        nw = ep["visual"].shape[0]
        w = np.random.randint(0, nw)
        t = ep["starts"][w]
        n = len(ep["s"])
        idx = np.clip(np.arange(t, t + self.H), 0, n - 1)
        ns, na = proprio_sample(ep["s"][t], ep["a"][idx], self.stats)
        return dict(
            video_latent=ep["visual"][w].float(),   # [48,2,12,48]
            ref=ep["ref"][w].float(),                # [48,1,12,48]
            state=torch.from_numpy(ns), action=torch.from_numpy(na),
            t5=ep["t5"], embed_id=self.embed_id,
        )


# ---- self-test (proprio only; no GPU/VAE) ----
if __name__ == "__main__":
    import sys
    base = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/wam_fold_v1"
    fdir = os.path.dirname(os.path.abspath(__file__))
    cases = [
        ("visrobot01_train", f"{fdir}/assets/statistics_visrobot01.json", "visrobot01", 0),
        ("kairobot01", f"{fdir}/assets/statistics_kairobot01.json", "kairobot01", 1),
    ]
    for sub, stats, emb, eid in cases:
        ds = LeRobotJointDataset(f"{base}/{sub}", stats, action_chunk=33, embodiment=emb, embed_id=eid)
        print(f"[{sub}] episodes={len(ds)}  stats={os.path.basename(stats)}")
        b = ds.load_proprio(ds._index[0], t=0)
        st, ac = b["state"], b["action"]
        print(f"   state {tuple(st.shape)} mean={st.mean():.3f} std={st.std():.3f}")
        print(f"   action(chunk) {tuple(ac.shape)} mean={ac.mean():.3f} std={ac.std():.3f}")
        # round-trip parity: undo -> should recover abs action chunk
        s_raw = np.stack(pd.read_parquet(ds._index[0], columns=["observation.state"])
                         ["observation.state"].values).astype(np.float32)[0]
        a_back = undo_proprio_action(ac.numpy(), s_raw, ds.stats)
        a_true = np.stack(pd.read_parquet(ds._index[0], columns=["action"])["action"].values
                          ).astype(np.float32)[0:33]
        err = np.abs(a_back - a_true).mean()
        print(f"   round-trip |undo(norm)-abs_action| = {err:.6e}  {'OK' if err < 1e-4 else 'FAIL'}")
