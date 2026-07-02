#!/usr/bin/env python
"""ep2302: Coverage-gated jump control vs Raw argmin vs Viterbi — aligned video.

Reads cached CRAVE model + jump analysis data, renders a single full-speed mp4:
  Left:  camera frames
  Right: 3 value curves overlaid (raw / coverage-gated+hyst / Viterbi) with time cursor
No slow-mo, no title cards — clean continuous alignment.

Usage:
  /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_ep2302_covgate_video.py
Output: temp/coverage_gated_jump/ep2302_covgate_comparison.mp4
"""
import os, sys, numpy as np, cv2, av
from pathlib import Path
from sklearn.cluster import KMeans
from scipy.ndimage import median_filter

REPO = Path(os.environ.get("REPO", "/home/tim/workspace/deepdive_kai0"))
FC = REPO / "temp/crave_kai0bd/feat_cache"
OUT = REPO / "temp/coverage_gated_jump"; OUT.mkdir(exist_ok=True, parents=True)
DS = REPO / "kai0/data/Task_A/kai0_base"
EP = 2302; STRIDE = 10; import json
csDS = json.load(open(DS/"meta/info.json"))["chunks_size"]

# ===== Load model (same as previous experiments) =====
def load_ep(e):
    d = np.load(FC / f"ep{e}.npz"); a, r, s = d["armmask"], d["raw"], d["state"]
    n = min(len(a), len(r), len(s)); s = np.clip(np.nan_to_num(s[:n].astype(np.float64)), -10, 10)
    return a[:n], r[:n], s, n

all_eps = sorted(int(p.stem[2:]) for p in FC.glob("ep*.npz"))
mine_pool = [e for e in all_eps if e < 3055]
rng = np.random.RandomState(0); N_MINE = 200
mined = sorted(rng.permutation(mine_pool)[:min(N_MINE, len(mine_pool))].tolist())
if EP not in mined: mined = sorted(mined + [EP])

feature_list = []; E_mine = []
for e in mined:
    aa, rr, st, n = load_ep(e)
    an = aa/(np.linalg.norm(aa,axis=1,keepdims=True)+1e-8)
    rn = rr/(np.linalg.norm(rr,axis=1,keepdims=True)+1e-8)
    feature_list.append(np.concatenate([rn,an],1))
    E_mine.append(np.full(len(feature_list[-1]), e))
G_all = np.concatenate(feature_list); E_all = np.concatenate(E_mine)
N_ep = len(mined)

km = KMeans(96, n_init=2, random_state=0).fit(G_all)
lab = km.labels_; C_all = km.cluster_centers_
T_all = np.concatenate([np.arange(len(f))/max(1,len(f)-1) for f in feature_list])
tpos = np.array([T_all[lab==c].mean() if (lab==c).any() else 0.5 for c in range(96)])
covE = np.array([len(set(E_all[lab==c]))/N_ep if (lab==c).any() else 0.0 for c in range(96)])
TAU = float(np.quantile(covE, 0.5))
hi_cov = covE >= TAU
sel = sorted([c for c in range(96) if hi_cov[c]], key=lambda c: tpos[c])
C = C_all[sel]; Pord = tpos[sel]; cov_sel = covE[sel]; NM = len(sel)
print(f"[model] {NM} milestones, coverage τ={TAU:.2f}", flush=True)

# ===== ep2302 features =====
aa, rr, st, n_feat = load_ep(EP)
an = aa/(np.linalg.norm(aa,axis=1,keepdims=True)+1e-8)
rn = rr/(np.linalg.norm(rr,axis=1,keepdims=True)+1e-8)
Ge = np.concatenate([rn,an],1)
dist_all = np.linalg.norm(Ge[:,None]-C[None],axis=2)

# ---- Raw argmin ----
raw_idx = dist_all.argmin(1); raw_val = Pord[raw_idx]

# ---- Coverage-gated + hysteresis ----
cov_hyst_val = np.zeros(n_feat)
prev_k = None
for t in range(n_feat):
    k = dist_all[t].argmin()
    if cov_sel[k] >= TAU:
        prev_k = k; cov_hyst_val[t] = Pord[k]
    elif prev_k is not None:
        cov_hyst_val[t] = Pord[prev_k]
    else:
        hi_idx = np.where(cov_sel>=TAU)[0]; prev_k=hi_idx[dist_all[t][hi_idx].argmin()]; cov_hyst_val[t]=Pord[prev_k]

# ---- Viterbi ----
NB=21; b=np.linspace(0,1,NB); LAM=8.0; MEDW=9
em=np.full((n_feat,NB),1e3)
for ci in range(NM):
    bi=np.abs(b-Pord[ci]).argmin(); em[:,bi]=np.minimum(em[:,bi],dist_all[:,ci])
ds=dist_all[:,np.abs(Pord).argmin()]; de=dist_all[:,np.abs(Pord-1.0).argmin()]
tn=np.arange(n_feat)/n_feat
em[:,0]=np.minimum(em[:,0],np.where(tn<0.3,ds,ds+(tn-0.3)*6))
em[:,-1]=np.minimum(em[:,-1],np.where(tn>0.6,de,de+(0.6-tn)*6))
pen=LAM*np.abs(b[:,None]-b[None])
cost=np.full(NB,1e9); cost[0]=em[0,0]; bp_v=np.zeros((n_feat,NB),int)
for j in range(1,n_feat): tr=cost[None,:]+pen; kk=tr.argmin(1); cost=em[j]+tr[np.arange(NB),kk]; bp_v[j]=kk
cost[-1]-=2; path=np.zeros(n_feat,int); path[-1]=cost.argmin()
for j in range(n_feat-2,-1,-1): path[j]=bp_v[j+1,path[j+1]]
vit_val=median_filter(b[path],MEDW)

# ---- Upsample to 30Hz ----
import pandas as pd
NF = len(pd.read_parquet(DS/"data"/f"chunk-{EP//csDS:03d}"/f"episode_{EP:06d}.parquet",columns=["frame_index"]))

def upsample(v3hz, nf_target):
    v = np.repeat(v3hz, STRIDE)[:nf_target]
    if len(v) < nf_target: v = np.concatenate([v, np.full(nf_target-len(v), v[-1])])
    return v

raw_30 = upsample(raw_val, NF)
cov_30 = upsample(cov_hyst_val, NF)
vit_30 = upsample(vit_val, NF)
n = min(NF, len(raw_30), len(cov_30), len(vit_30))
raw_30, cov_30, vit_30 = raw_30[:n], cov_30[:n], vit_30[:n]

# ===== Build panel figure =====
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

x_sec = np.arange(n) / 30.0

def mono(v): return np.mean(np.diff(v)>=-0.001)

PFIG = plt.figure(figsize=(9.5, 5.5), dpi=100)
ax = PFIG.add_subplot(111)
ax.plot(x_sec, raw_30, 'gray', lw=0.5, alpha=0.5, label=f'raw argmin (mono={mono(raw_30):.3f})')
ax.plot(x_sec, cov_30, '#d62728', lw=1.5, alpha=0.9, label=f'coverage+hysteresis (mono={mono(cov_30):.3f})')
ax.plot(x_sec, vit_30, '#1a9641', lw=2.0, alpha=0.8, label=f'Viterbi (mono={mono(vit_30):.3f})')
ax.set_xlim(0, n/30); ax.set_ylim(-0.05, 1.08)
ax.set_xlabel('time (s)'); ax.set_ylabel('progress'); ax.set_title(f'ep{EP}: Raw argmin vs Coverage+Hysteresis vs Viterbi ({NM} milestones, τ={TAU:.2f})')
ax.legend(fontsize=9, loc='upper left'); ax.grid(alpha=0.25)
PFIG.canvas.draw(); PANEL = np.asarray(PFIG.canvas.buffer_rgba())[...,:3].copy()
Hp, Wp = PANEL.shape[:2]; plt.close(PFIG)

# Map functions
MP = ax  # saved panel axis
def xpx(sec): x0,x1=MP.get_position().x0,MP.get_position().x1; xl,xh=MP.get_xlim(); return int(round((x0+(sec-xl)/(xh-xl)*(x1-x0))*Wp))
def ypx(val): y0,y1=MP.get_position().y0,MP.get_position().y1; yl,yh=MP.get_ylim(); return int(round((1-(y0+(val-yl)/(yh-yl)*(y1-y0)))*Hp))

# ===== Render video =====
mp4_path = str(DS/f"videos/chunk-{EP//csDS:03d}/observation.images.top_head/episode_{EP:06d}.mp4")
cap = cv2.VideoCapture(mp4_path)

cam_h = Hp; cam_w = int(round(640/480*cam_h))//2*2
Wt = (cam_w + Wp)//2*2; Ht = Hp//2*2

omp4 = str(OUT/"ep2302_covgate_comparison.mp4")
oc = av.open(omp4, mode="w"); stv = oc.add_stream("libx264", rate=30)
stv.width, stv.height, stv.pix_fmt = Wt, Ht, "yuv420p"
stv.options = {"preset": "veryfast", "crf": "23"}

print(f"[render] {n} frames, {Wt}x{Ht} → {omp4}", flush=True)
for fi in range(n):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, fr = cap.read()
    cam = fr[:,:,::-1] if ok else np.zeros((480,640,3),np.uint8)
    cam2 = cv2.resize(np.ascontiguousarray(cam), (cam_w, cam_h))

    # Panel with cursor
    panel = PANEL.copy(); sec = fi/30.0; px = xpx(sec)
    cv2.line(panel, (px,0), (px,Hp), (40,40,40), 1)
    # Draw 3 dots at current values
    cv2.circle(panel, (xpx(sec), ypx(float(raw_30[fi]))), 5, (128,128,128), -1)
    cv2.circle(panel, (xpx(sec), ypx(float(cov_30[fi]))), 5, (40,40,214), -1)  # BGR for #d62728
    cv2.circle(panel, (xpx(sec), ypx(float(vit_30[fi]))), 5, (40,150,26), -1)  # BGR for #1a9641

    # Info overlay on camera
    cv2.rectangle(cam2, (6,6), (cam_w-6, 80), (0,0,0), -1)
    cv2.putText(cam2, f"ep{EP} f{fi}  v_raw={raw_30[fi]:.2f}  v_cov={cov_30[fi]:.2f}  v_vit={vit_30[fi]:.2f}",
               (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(cam2, "red=coverage+hyst  green=Viterbi  gray=raw argmin",
               (12,52), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1, cv2.LINE_AA)
    cv2.putText(cam2, f"coverage threshold tau={TAU:.2f}  ({NM} milestones)",
               (12,72), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1, cv2.LINE_AA)

    canv = np.zeros((Hp, cam_w+Wp, 3), np.uint8)
    canv[:,:cam_w] = cam2; canv[:,cam_w:] = panel

    for pkt in stv.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(canv[:Ht,:Wt]), format="rgb24")): oc.mux(pkt)
    if (fi+1)%500==0: print(f"  {fi+1}/{n}", flush=True)

for pkt in stv.encode(): oc.mux(pkt)
oc.close(); cap.release()
print(f"SAVED {omp4}  {n}f {n/30:.0f}s", flush=True)
print("COVGATE_VIDEO_DONE", flush=True)
