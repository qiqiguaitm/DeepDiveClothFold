"""Path A (BUG_AUDIT CRITICAL-1): 把 starVLA 世界模型目标 h_t1_gt 从 t+7 近未来帧
换成 **milestone+1 帧特征**。高内聚: 所有 milestone-target 逻辑在此单模块;
framework/dataloader 只加最小 env 门控 hook。

用法(env 门控):
  LMWM_MILESTONE_TARGET=<pairs.npz 路径>   # 开启 + 指定 CRAVE 训练对
  LMWM_FEAT_DIR=<libero_dinov3base 目录>    # milestone+1 目标特征源(离线 DINOv3-vitb16 grid)
Provider 用 pairs(cur_ep,cur_fi,tgt_fi) 建 (ep,cur_fi)->tgt_fi 映射, LRU 缓存 episode 特征,
get_target(ep_ids, frame_ids) 返回 (target_feat[B,256,768], valid_mask[B])。
无 milestone+1 的帧 valid=False(下游对这些帧的 perceptual loss 置零, 退回不监督)。

特征是 stride=2 抽的 → batch 原始 frame_index f 映射到特征索引 i=f//2。
"""
from __future__ import annotations
import os
from collections import OrderedDict
import numpy as np
import torch


class MilestoneTargetProvider:
    def __init__(self, pairs_npz: str, feat_dir: str = "", compact_npz: str = "",
                 stride: int = 2, lru: int = 400):
        P = np.load(pairs_npz)
        ce, cf, tf = P["cur_ep"], P["cur_fi"], P["tgt_fi"]
        # (ep, cur_fi) -> tgt_fi
        self.map: dict[tuple[int, int], int] = {(int(ce[i]), int(cf[i])): int(tf[i]) for i in range(len(ce))}
        self.feat_dir = feat_dir
        self.stride = stride
        self.lru_cap = lru
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        # 紧凑模式(gf3): 只存 unique(ep,tgt_fi) 目标特征, 免 39GB 全特征
        self.compact = None
        if compact_npz:
            C = np.load(compact_npz)
            self.compact = C["feat"]                   # [Nu,256,768] fp16
            self.ct_row = {(int(C["ep"][i]), int(C["tgt_fi"][i])): i for i in range(len(C["ep"]))}
            print(f"[LMWM] compact target store: {len(self.ct_row)} targets, {self.compact.nbytes/1e9:.2f}GB", flush=True)

    def _feat(self, ep: int) -> np.ndarray | None:
        if ep in self._cache:
            self._cache.move_to_end(ep)
            return self._cache[ep]
        path = os.path.join(self.feat_dir, f"ep{ep}.npz")
        if not os.path.exists(path):
            return None
        g = np.load(path)["grid"]                      # [N,256,768] fp16
        self._cache[ep] = g
        if len(self._cache) > self.lru_cap:
            self._cache.popitem(last=False)
        return g

    def _target_feat(self, ep: int, tfi: int) -> np.ndarray | None:
        """返回 (ep,tgt_fi) 的目标特征 [256,768]; 优先紧凑存储, 否则全特征。"""
        if self.compact is not None:
            row = self.ct_row.get((ep, tfi))
            return None if row is None else self.compact[row]
        g = self._feat(ep)
        return None if (g is None or tfi >= len(g)) else g[tfi]

    @torch.no_grad()
    def get_target(self, ep_ids, frame_ids, out_shape, device, dtype):
        """ep_ids/frame_ids: [B] long tensors(原始 episode_index / frame_index)。
        返回 target_feat[B,256,768](无效帧填 0), valid_mask[B] bool。"""
        B = len(ep_ids)
        K, D = out_shape                                # 256, 768
        tgt = torch.zeros(B, K, D, device=device, dtype=dtype)
        valid = torch.zeros(B, dtype=torch.bool, device=device)
        ep_np = ep_ids.detach().cpu().numpy(); fr_np = frame_ids.detach().cpu().numpy()
        for b in range(B):
            ep = int(ep_np[b]); i = int(fr_np[b]) // self.stride
            tfi = self.map.get((ep, i))
            if tfi is None:
                continue
            tf = self._target_feat(ep, tfi)
            if tf is None:
                continue
            tgt[b] = torch.from_numpy(tf.astype(np.float32)).to(device=device, dtype=dtype)
            valid[b] = True
        return tgt, valid


_PROVIDER: MilestoneTargetProvider | None = None


def get_provider() -> MilestoneTargetProvider | None:
    """env 门控单例。LMWM_MILESTONE_TARGET 未设则返回 None(退回原 t+7 目标)。"""
    global _PROVIDER
    pairs = os.environ.get("LMWM_MILESTONE_TARGET")
    if not pairs:
        return None
    if _PROVIDER is None:
        feat_dir = os.environ.get("LMWM_FEAT_DIR", "")
        compact = os.environ.get("LMWM_TARGET_COMPACT", "")   # gf3: 紧凑目标存储, 免 39GB 全特征
        if not feat_dir and not compact:
            feat_dir = "/home/tim/workspace/deepdive_kai0/lmvla/lmwm/data/libero_dinov3base"
        _PROVIDER = MilestoneTargetProvider(pairs, feat_dir=feat_dir, compact_npz=compact,
                                            stride=int(os.environ.get("LMWM_FEAT_STRIDE", "2")))
        print(f"[LMWM] milestone-target provider: {len(_PROVIDER.map)} pairs from {pairs}", flush=True)
    return _PROVIDER
