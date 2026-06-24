"""CRAVE 标准连续读出: 在 DP 阶梯上叠移动平均连续化(同 build_ds_A_from_mv.py)。
window 随帧率缩放: w = round(41 * fps/30)。30fps→41, 3Hz→4。保结构+保退步, 仅去硬台阶感。
所有可视化/打标脚本统一 import 这个, 不直接画硬阶梯。
用法: from crave_readout import smooth_monotone  ;  v_cont = smooth_monotone(v, fps=30)
"""
import numpy as np


def smooth_monotone(v, fps=30.0, w=None):
    """边缘填充移动平均 + re-clip[0,1]。w 默认按 fps 缩放(基准 41@30fps)。"""
    v = np.asarray(v, dtype=np.float64)
    if w is None:
        w = max(2, int(round(41 * fps / 30.0)))
    if len(v) < 3 or w < 2:
        return v.astype(np.float32)
    h = w // 2
    vp = np.concatenate([np.full(h, v[0]), v, np.full(h, v[-1])])
    k = np.ones(w, dtype=np.float64) / w
    vs = np.convolve(vp, k, mode="valid")[: len(v)]
    return np.clip(vs, 0.0, 1.0).astype(np.float32)
