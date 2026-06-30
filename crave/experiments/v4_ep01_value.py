"""用已挖好的 kai0 DINOv3-H 特征(temp/crave_full_dinov3h, 3055ep@3Hz, 1280维纯视觉)处理 v4 ep01 出 value 图。
最新架构:DINOv3-H + 均匀化选簇 + 进度先验 + 远超罚 + 置信门控(无 proprio,纯视觉;kai0 缓存即纯视觉)。
v4 与 kai0 同任务(Task_A base),milestone 迁移。kai0 不重挖,直接用缓存特征重建 milestone。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, av
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from crave.config import REPO
from crave.encoders import load_encoder
from crave.data.kai0 import crop224
from crave.utils import L2, med, smooth_monotone
from crave.render import setup_mpl
from milestone_select import build_milestones_uniform
from transition_prior_fix import build_pen, viterbi_pen, visited_sequence
from cross_dataset_transition import emit_of, conf_hold, temporal_smooth, _smooth_block

K_SM = 3  # 3Hz 缓存 → 轻平滑(30Hz 的 9 帧 ≈ 3Hz 的 ~1 帧;取 3 帧 mild)
OUTD = REPO / "temp/crave_full_dinov3h"
V4 = Path("/transfer-shanghai/KAI0/Task_A/base/v4/2026-06-28-v4")

# 1+2) milestone 缓存(首次构建 ~90s, 之后秒载)——用最新均匀化选簇 on kai0 DINOv3-H 缓存特征
MC = OUTD / "milestones_uniform_dinov3h.npz"
if MC.exists():
    z = np.load(MC); C, Pord, sk, pen = z["C"], z["Pord"], z["sk"], z["pen"]; tau_lo, tau_hi = float(z["tau_lo"]), float(z["tau_hi"]); M = len(Pord)
    print(f"loaded cached milestones: M={M}", flush=True)
else:
    idx = np.load(OUTD / "index.npz", allow_pickle=True); E, Tv, N = idx["E"], idx["T"].astype(float), int(idx["n"])
    feat = np.zeros((N, 1280), np.float16)
    for sh in ["shard_0.npz", "shard_1.npz"]:
        s = np.load(OUTD / sh); feat[s["gidx"]] = s["feat"]
    F = L2(feat.astype(np.float32)); ne = len(np.unique(E))
    print(f"kai0 缓存: {ne} ep / {N} 帧 @3Hz, DINOv3-H 1280维(纯视觉)", flush=True)
    F = temporal_smooth(F, E, K_SM)
    cen, lab, order, Pord, M, nf = build_milestones_uniform(F, E, Tv, ne); C = cen[order]
    from sklearn.cluster import KMeans
    eps_s = sorted(np.unique(E).tolist())
    SP = np.concatenate([F[np.where(E == e)[0][np.argsort(Tv[np.where(E == e)[0]])][:2]] for e in eps_s[:400]])
    sk = KMeans(8, n_init=2, random_state=0).fit(SP).cluster_centers_
    am = np.linalg.norm(F[:, None] - C[None], axis=2).argmin(1); counts = np.zeros((M, M))
    for e in eps_s:
        seq = visited_sequence(am[np.where(E == e)[0][np.argsort(Tv[np.where(E == e)[0]])]])
        for i, j in zip(seq[:-1], seq[1:]): counts[i, j] += 1
    pen = build_pen(counts, Pord, 1.0, 0.4, 1.0)
    dM = np.linalg.norm(F[:, None] - C[None], axis=2).min(1); tau_lo, tau_hi = float(np.percentile(dM, 50)), float(np.percentile(dM, 82))
    np.savez(MC, C=C, Pord=Pord, sk=sk, pen=pen, tau_lo=tau_lo, tau_hi=tau_hi)
    print(f"milestone(均匀化选簇): M={M} (+{nf}补洞) → 已缓存 {MC.name}", flush=True)


def ms_of(Fq):
    base = np.linalg.norm(Fq[:, None] - C[None], axis=2)
    ms = viterbi_pen(emit_of(Fq, C, sk, Pord, 0.8, 0.15, 1.3, 0.25), pen)
    return conf_hold(ms, base, tau_lo, tau_hi, 0.5)


# 3) 提 v4 ep01(=ep0)@3Hz(stride10), crop224, DINOv3-H 编码
enc = load_encoder("dinov3-h")
mp4 = V4 / "videos/chunk-000/observation.images.top_head/episode_000000.mp4"
imgs = []; c = av.open(str(mp4))
for j, fr in enumerate(c.decode(video=0)):
    if j % 10 == 0: imgs.append(crop224(fr.to_ndarray(format="rgb24")))
c.close()
Fq = L2(enc.encode_pooled(imgs)); Fq = _smooth_block(Fq, K_SM); nq = len(Fq)
# 跨录制 domain shift: conf_hold 用目标(v4)自身 dmin 分布标定, 否则 kai0 in-domain tau 把 99% 帧门掉→塌缩
base_q = np.linalg.norm(Fq[:, None] - C[None], axis=2); dmin_q = base_q.min(1)
conf_in = np.mean(dmin_q < tau_hi) * 100  # 按 kai0 in-domain tau 算的"置信"占比(衡量 domain gap)
tau_lo, tau_hi = float(np.percentile(dmin_q, 50)), float(np.percentile(dmin_q, 82))  # 目标相对标定
ms = ms_of(Fq); val = smooth_monotone(Pord[ms].astype(float), fps=3.0)
mono = np.mean(np.diff(val) >= -1e-6) * 100
print(f"v4 ep01: {nq}帧@3Hz → end={val[-1]:.2f} max={val.max():.2f} 单调{mono:.0f}%", flush=True)
print(f"  domain gap: v4 dmin p50={np.percentile(dmin_q,2):.3f}.. 中位{np.median(dmin_q):.3f}; 按 kai0 in-domain tau 仅 {conf_in:.0f}% 帧置信", flush=True)

# 4) 出图:value 曲线 + milestone 台阶 + 代表帧
plt = setup_mpl(); fig = plt.figure(figsize=(13, 6)); gs = fig.add_gridspec(2, 5, height_ratios=[1.5, 1], hspace=0.35, wspace=0.3)
t = np.arange(nq) * 10 / 30.0
axv = fig.add_subplot(gs[0, :]); axv.plot(t, val, color="#2ca02c", lw=2.2, label="CRAVE value(kai0 DINOv3-H milestone 读出)")
axv.step(t, Pord[ms], where="post", color="#888", lw=0.9, alpha=.55, label="匹配 milestone 进度(原始)")
axv.plot([0, t[-1]], [0, 1], "k--", lw=1, alpha=.4, label="线性时间参考")
axv.set_xlim(0, t[-1]); axv.set_ylim(-.02, 1.02); axv.set_xlabel("时间 (s)"); axv.set_ylabel("value"); axv.grid(alpha=.25); axv.legend(fontsize=9, loc="upper left")
axv.set_title(f"v4/2026-06-28-v4 ep01 · 用已挖 kai0 DINOv3-H {M} milestone 读出 · {nq}帧≈{t[-1]:.0f}s · end={val[-1]:.2f} 单调{mono:.0f}%", fontsize=11)
axv.text(0.99, 0.04, f"⚠ 跨录制 domain shift: 按 kai0 in-domain 阈值仅 {conf_in:.0f}% 帧置信(v4 dmin 中位 {np.median(dmin_q):.2f} vs kai0 in-domain 0.44)\n"
                     f"→ 视觉匹配弱+起末混叠, 本曲线主要由进度先验稳出(≈线性时间), 非真实 milestone 复现匹配",
         transform=axv.transAxes, ha="right", va="bottom", fontsize=8, color="#b00", bbox=dict(boxstyle="round", fc="#fff3f3", ec="#e0a0a0"))
for i, fk in enumerate(np.linspace(0, nq - 1, 5).astype(int)):
    ax = fig.add_subplot(gs[1, i]); ax.imshow(imgs[fk]); ax.axis("off"); ax.set_title(f"t={fk*10/30:.0f}s\nvalue={val[fk]:.2f}", fontsize=8)
out = REPO / "crave/docs/visualization/cross_dataset/v4_ep01_value.png"
fig.savefig(out, dpi=130, bbox_inches="tight"); print("SAVED", out, flush=True)
