#!/usr/bin/env python
"""4 条短视频: 【所有上升段 / 所有平台段 / 所有退步段】+ 【全程混合 MIXED(按时间顺序三类交错)】。
每段: 中文标题卡(段序/帧范围/Δv) + 三栏对齐逐帧
   [相机(带 CRAVE/KAI0-AE 当前判档) | CRAVE&KAI0-AE 全程 value+分类条(游标) | 当前 value 对应的典型簇原型]。
读 crave_interp_clusters_ep808.py 产的 _cache.npz / _proto_*.npy。
输出: temp/crave_interp_ep808/crave_kai0ae_{POS,NORMAL,NEG}_all_ep808.mp4
"""
import os
import numpy as np, cv2, av

from crave.config import REPO
from crave.render import setup_mpl

plt = setup_mpl()

DS = REPO / os.environ.get("INTERP_DS", "kai0/data/Task_A/self_built/A_smooth800_dagger_all")
EP = int(os.environ.get("INTERP_EP", "808")); OUT = REPO / f"temp/crave_interp_ep{EP}{os.environ.get('INTERP_TAG','')}"; csDS = 1000
W = 50; MINLEN = 20; MAXCLIP = 90  # 段最短 20 帧(0.67s)纳入; 每段最多取 90 帧(长段抽帧)
SLOW = 2; HOLD = 14  # 每帧重复 SLOW 次=慢放; 典型簇切换时额外定格 HOLD 帧(看清状态/簇变化)
RGB = {1: (0.17, 0.63, 0.17), 0: (0.6, 0.6, 0.6), -1: (0.84, 0.15, 0.16)}
BGR = {1: (44, 160, 44), 0: (150, 150, 150), -1: (214, 39, 40)}; NAME = {1: "POSITIVE", 0: "NORMAL", -1: "NEGATIVE"}
CLS = {1: ("POS", "上升段 POSITIVE", (0.3, 0.85, 0.3)), 0: ("NORMAL", "平台段 NORMAL", (0.82, 0.82, 0.82)), -1: ("NEG", "退步段 NEGATIVE", (0.9, 0.4, 0.4))}

z = np.load(OUT / "_cache.npz")
cv_, ae_v, ccls, acls, nm30, Pord, n = z["cv"], z["ae_v"], z["ccls"].astype(int), z["acls"].astype(int), z["nm30"].astype(int), z["Pord"], int(z["n"])
HAVE_AE = bool(int(z["have_ae"])) if "have_ae" in z else True   # kai0_base 等无 KAI0-AE 标注 → 仅 CRAVE
proto_imgs = np.load(OUT / "_proto_imgs.npy"); proto_meta = np.load(OUT / "_proto_meta.npy")
fc = {c: float((ccls == c).mean()) for c in (1, 0, -1)}; fa = {c: float((acls == c).mean()) for c in (1, 0, -1)} if HAVE_AE else {1: 0, 0: 0, -1: 0}


def runs(cls):
    out = []; s = 0
    for i in range(1, len(cls) + 1):
        if i == len(cls) or cls[i] != cls[s]: out.append((s, i - 1, int(cls[s]))); s = i
    return out
ALLRUNS = runs(ccls)

# ---- 全程背景面板(有 AE: CRAVE+KAI0-AE 四格; 无 AE: 仅 CRAVE 两格) ----
x = np.arange(n) / 30.0
PFIG = plt.figure(figsize=(9.5, 6.2), dpi=100)
if HAVE_AE:
    gs = PFIG.add_gridspec(4, 1, height_ratios=[1, 0.22, 1, 0.22], hspace=0.5)
    axcv, axav = PFIG.add_subplot(gs[0]), PFIG.add_subplot(gs[2]); axcs, axas = PFIG.add_subplot(gs[1]), PFIG.add_subplot(gs[3])
else:
    gs = PFIG.add_gridspec(2, 1, height_ratios=[1, 0.22], hspace=0.45)
    axcv, axcs = PFIG.add_subplot(gs[0]), PFIG.add_subplot(gs[1])
axcv.plot(x, cv_, color="#333", lw=0.6, alpha=.35)
for c in (-1, 0, 1):
    m = ccls == c; axcv.scatter(x[m], cv_[m], s=4, c=[RGB[c]])
axcv.set_ylim(-.05, 1.05); axcv.set_xlim(0, n / 30); axcv.set_ylabel("CRAVE\nvalue", fontsize=8); axcv.tick_params(labelsize=7)
axcv.set_title(f"CRAVE(零训练): pos{fc[1]:.0%} / normal{fc[0]:.0%} / neg{fc[-1]:.0%}", fontsize=10); axcv.grid(alpha=.2)
axcs.imshow(np.array([RGB[c] for c in ccls])[None], aspect="auto", extent=[0, n / 30, 0, 1]); axcs.set_yticks([]); axcs.set_xlim(0, n / 30); axcs.tick_params(labelsize=7); axcs.set_ylabel("分类条", fontsize=7)
if HAVE_AE:
    axav.plot(x, ae_v, color="#333", lw=0.6, alpha=.35)
    for c in (-1, 0, 1):
        m = acls == c; axav.scatter(x[m], ae_v[m], s=4, c=[RGB[c]])
    axav.set_ylim(np.nanmin(ae_v) - .05, np.nanmax(ae_v) + .05); axav.set_xlim(0, n / 30); axav.set_ylabel("KAI0-AE\nvalue", fontsize=8); axav.tick_params(labelsize=7)
    axav.set_title(f"KAI0-AE(监督): pos{fa[1]:.0%} / normal{fa[0]:.0%} / neg{fa[-1]:.0%}  (成功episode却{fa[-1]:.0%}红=噪声)", fontsize=10); axav.grid(alpha=.2)
    axas.imshow(np.array([RGB[c] for c in acls])[None], aspect="auto", extent=[0, n / 30, 0, 1]); axas.set_yticks([]); axas.set_xlim(0, n / 30); axas.set_xlabel("秒", fontsize=8); axas.tick_params(labelsize=7); axas.set_ylabel("分类条", fontsize=7)
else:
    axcs.set_xlabel("秒", fontsize=8)
PFIG.suptitle(f"ep{EP} — CRAVE vs KAI0-AE 三档分类对齐" if HAVE_AE else f"ep{EP}(kai0_base)— CRAVE 三档分类(KAI0-AE 未在 kai0_base 标注)", fontsize=11)
PFIG.canvas.draw(); PANEL = np.asarray(PFIG.canvas.buffer_rgba())[..., :3].copy(); Hp, Wp = PANEL.shape[:2]


def pm(a):
    bb = a.get_position(); xl, xh = a.get_xlim(); yl, yh = a.get_ylim(); return bb.x0, bb.x1, bb.y0, bb.y1, xl, xh, yl, yh
def xpx(m, sec):
    x0, x1, *_, xl, xh, _, _ = m; return int(round((x0 + (sec - xl) / (xh - xl) * (x1 - x0)) * Wp))
def yp(m, val):
    x0, x1, y0, y1, xl, xh, yl, yh = m; return int(round((1 - (y0 + (val - yl) / (yh - yl) * (y1 - y0))) * Hp))
MV_ = pm(axcv); MA_ = pm(axav) if HAVE_AE else None
plt.close(PFIG)


def proto_card(k):
    img = proto_imgs[k]; pid, prog, pe = int(proto_meta[k][0]), proto_meta[k][1], int(proto_meta[k][2])
    f = plt.figure(figsize=(3.4, 6.2), dpi=100); a = f.add_axes([0.04, 0.30, 0.92, 0.46]); a.imshow(img); a.axis("off")
    f.text(0.5, 0.96, "当前 value 对应的\n典型簇(milestone)", ha="center", va="top", fontsize=12, color="#222")
    f.text(0.5, 0.25, f"典型簇 #{pid}", ha="center", fontsize=15, color="#3b6fb0", fontweight="bold")
    f.text(0.5, 0.17, f"挖掘进度 = {prog:.2f}", ha="center", fontsize=13, color="#1a9641")
    f.text(0.5, 0.09, f"= CRAVE value\n(原型来自 demo ep{pe})", ha="center", fontsize=10, color="#555")
    f.patch.set_facecolor("#f5f5f5"); f.canvas.draw()
    im = np.asarray(f.canvas.buffer_rgba())[..., :3].copy(); plt.close(f); return im
proto_cards = [proto_card(k) for k in range(len(proto_imgs))]
Wc = proto_cards[0].shape[1]

cam_h = Hp; cam_w = int(round(640 / 480 * cam_h)) // 2 * 2
Wt = (cam_w + Wp + Wc) // 2 * 2; Ht = Hp // 2 * 2


def title_card(lines, color):
    f = plt.figure(figsize=(Wt / 100, Ht / 100), dpi=100); f.patch.set_facecolor("#111")
    f.text(0.5, 0.5, lines, ha="center", va="center", fontsize=22, color=color, fontweight="bold", linespacing=1.6)
    f.canvas.draw(); im = np.asarray(f.canvas.buffer_rgba())[..., :3].copy(); plt.close(f); return im


def seg_idxs(s0, s1):
    idxs = list(range(s0, s1 + 1))
    if len(idxs) > MAXCLIP: idxs = idxs[:: max(1, len(idxs) // MAXCLIP)][:MAXCLIP]
    return idxs


def composite(fi, banner):
    mp4 = DS / f"videos/chunk-{EP//csDS:03d}/observation.images.top_head/episode_{EP:06d}.mp4"
    # 复用打开的 cap(外部传入更高效, 这里简化为函数内缓存)
    cap = composite._cap
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, fr = cap.read()
    cam = fr[:, :, ::-1] if ok else np.zeros((480, 640, 3), np.uint8)
    panel = PANEL.copy(); sec = fi / 30.0
    px = xpx(MV_, sec); cv2.line(panel, (px, 0), (px, Hp), (40, 40, 40), 1)
    cv2.circle(panel, (xpx(MV_, sec), yp(MV_, float(cv_[fi]))), 7, BGR[int(ccls[fi])], -1); cv2.circle(panel, (xpx(MV_, sec), yp(MV_, float(cv_[fi]))), 7, (0, 0, 0), 1)
    if HAVE_AE:
        cv2.circle(panel, (xpx(MA_, sec), yp(MA_, float(ae_v[fi]))), 7, BGR[int(acls[fi])], -1); cv2.circle(panel, (xpx(MA_, sec), yp(MA_, float(ae_v[fi]))), 7, (0, 0, 0), 1)
    cam2 = cv2.resize(np.ascontiguousarray(cam), (cam_w, cam_h))
    cv2.rectangle(cam2, (6, 6), (cam_w - 6, 104), (0, 0, 0), -1)
    cv2.putText(cam2, banner, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(cam2, f"ep{EP} f{fi}  v={cv_[fi]:.2f}", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(cam2, f"CRAVE: {NAME[int(ccls[fi])]}", (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.58, BGR[int(ccls[fi])][::-1], 2, cv2.LINE_AA)
    if HAVE_AE:
        cv2.putText(cam2, f"KAI0-AE: {NAME[int(acls[fi])]}", (12, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.58, BGR[int(acls[fi])][::-1], 2, cv2.LINE_AA)
    canv = np.zeros((Hp, cam_w + Wp + Wc, 3), np.uint8)
    canv[:, :cam_w] = cam2; canv[:, cam_w:cam_w + Wp] = panel; canv[:, cam_w + Wp:] = cv2.resize(proto_cards[int(nm30[fi])], (Wc, Hp))
    return canv


composite._cap = cv2.VideoCapture(str(DS / f"videos/chunk-{EP//csDS:03d}/observation.images.top_head/episode_{EP:06d}.mp4"))

# ---- 连续完整版: 无标题卡、无簇切换定格、无慢放, 正常速度逐帧走完全程 ----
if os.environ.get("INTERP_FULL_ONLY"):
    omp4 = str(OUT / f"crave_kai0ae_FULL_ep{EP}.mp4")
    oc = av.open(omp4, mode="w"); stv = oc.add_stream("libx264", rate=30); stv.width, stv.height, stv.pix_fmt = Wt, Ht, "yuv420p"
    stv.options = {"preset": "veryfast", "crf": "23"}
    for fi in range(n):
        base = composite(fi, f"ep{EP} FULL continuous  f{fi}")
        for pkt in stv.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(base[:Ht, :Wt]), format="rgb24")): oc.mux(pkt)
        if (fi + 1) % 500 == 0: print(f"  full {fi+1}/{n}", flush=True)
    for pkt in stv.encode(): oc.mux(pkt)
    oc.close(); composite._cap.release()
    print(f"SAVED {omp4}  {n}f {n/30:.0f}s (无标题卡/无定格/无慢放)", flush=True); print("INTERP_VIDEOS_DONE", flush=True)
    raise SystemExit


def render(segs, omp4, head_text, head_col):
    """segs = [(s0,s1,c)...](逐段带类别); 每段标题卡按该段类别着色, 逐帧慢放 + 簇切换定格。"""
    oc = av.open(omp4, mode="w"); stv = oc.add_stream("libx264", rate=30); stv.width, stv.height, stv.pix_fmt = Wt, Ht, "yuv420p"
    stv.options = {"preset": "veryfast", "crf": "23"}

    def emit(rgb):
        for pkt in stv.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb[:Ht, :Wt]), format="rgb24")): oc.mux(pkt)
    for _ in range(45): emit(title_card(head_text, head_col))
    ntot = 45
    for k, (s0, s1, c) in enumerate(segs):
        short, cn, col = CLS[c]; dv = float(cv_[s1] - cv_[s0])
        tc = title_card(f"{cn}  第 {k+1}/{len(segs)} 段\n\n帧 [{s0}–{s1}]  ({(s1-s0+1)/30:.1f}s)\nvalue {cv_[s0]:.2f} → {cv_[s1]:.2f}  (Δ={dv:+.2f})\n典型簇 #{nm30[s0]} → #{nm30[s1]}", col)
        for _ in range(20): emit(tc); ntot += 1
        banner = f"{short} seg {k+1}/{len(segs)}  f[{s0}-{s1}]"
        prev_nm = None
        for fi in seg_idxs(s0, s1):
            base = composite(fi, banner); cur_nm = int(nm30[fi])
            if prev_nm is not None and cur_nm != prev_nm:  # 典型簇切换 → 定格 + 黄框标注
                mk = base.copy()
                cv2.rectangle(mk, (2, 2), (mk.shape[1] - 3, mk.shape[0] - 3), (255, 230, 0), 6)
                cv2.rectangle(mk, (cam_w + Wp + 6, 6), (cam_w + Wp + Wc - 6, 52), (0, 0, 0), -1)
                cv2.putText(mk, f"CLUSTER #{prev_nm} -> #{cur_nm}", (cam_w + Wp + 12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 230, 0), 2, cv2.LINE_AA)
                for _ in range(HOLD): emit(mk); ntot += 1
            for _ in range(SLOW): emit(base); ntot += 1
            prev_nm = cur_nm
    for pkt in stv.encode(): oc.mux(pkt)
    oc.close()
    print(f"SAVED {omp4}  段数{len(segs)}  {ntot}f {ntot/30:.1f}s", flush=True)


# ---- 3 条 per-class(按类分组)----
for c in (1, 0, -1):
    short, cn, col = CLS[c]
    myruns = [(s0, s1, c) for (s0, s1, cc) in ALLRUNS if cc == c and (s1 - s0 + 1) >= MINLEN]
    dropped = sum(1 for (s0, s1, cc) in ALLRUNS if cc == c and (s1 - s0 + 1) < MINLEN)
    _cmp = f"\n\nCRAVE {fc[c]:.0%} vs KAI0-AE {fa[c]:.0%}" if HAVE_AE else f"\n\nCRAVE {fc[c]:.0%} (kai0_base 无 KAI0-AE)"
    render(myruns, str(OUT / f"crave_kai0ae_{short}_all_ep{EP}.mp4"),
           f"{cn}\n\nep{EP} 全部 {len(myruns)} 个{cn[:3]}\n(每段≥{MINLEN}帧; 另有{dropped}个更短段略去){_cmp}", col)

# ---- 1 条 MIXED(全程按时间顺序, 三类交错)----
mixed = [(s0, s1, c) for (s0, s1, c) in ALLRUNS if (s1 - s0 + 1) >= MINLEN]   # ALLRUNS 已按 s0 升序
mdrop = sum(1 for (s0, s1, c) in ALLRUNS if (s1 - s0 + 1) < MINLEN)
_cmpm = f"\n\nCRAVE pos{fc[1]:.0%}/normal{fc[0]:.0%}/neg{fc[-1]:.0%}" + ("" if HAVE_AE else "(kai0_base 无 KAI0-AE)")
render(mixed, str(OUT / f"crave_kai0ae_MIXED_all_ep{EP}.mp4"),
       f"全程混合(按时间顺序, 三类交错)\n\nep{EP} 共 {len(mixed)} 段\n(每段≥{MINLEN}帧; 另 {mdrop} 个更短略去){_cmpm}", (0.95, 0.85, 0.3))

composite._cap.release()
print("INTERP_VIDEOS_DONE", flush=True)
