#!/usr/bin/env python
"""§5.3 最终版配图: kai 因果 GRU 蒸馏【去阶梯 polyline 双锚 Viterbi】teacher.

对齐 online_readout_route 单任务方案: 特征 = img PCA128(L2) ⊕ proprio位置14(标准化+L2), 1:1。
proprio 是最强相位锚 → milestone 从 img-only 的 M=3 拉回 M≈12, teacher 有真实阶段结构(非退化对角线)。
teacher = daw() 去阶梯 polyline; student = 142D 单向 GRU(严格因果 + 首帧 warmup)。
输出 temp/base_kai_gru.png → 覆盖 report assets/online_gru_heldout.png。
"""
import numpy as np, time, torch, torch.nn as nn, pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

REPO = Path('/vePFS/tim/workspace/deepdive_kai0'); DEV = 'cuda:0'
rng = np.random.RandomState(0); torch.manual_seed(0)
CAP = 1000; FPS = 30.; CSQ = 1000
KAI = REPO/'kai0/data/Task_A/kai0_base'   # proprio 源 (observation.state 14D)
def l2(x): return x/(np.linalg.norm(x, axis=-1, keepdims=True)+1e-9)
def cc(a, b): return np.corrcoef(a, b)[0, 1] if a.std() > 1e-6 and b.std() > 1e-6 else np.nan


def daw(F, C, P, lam):  # 双锚 Viterbi → polyline(去阶梯)
    sC = l2(F[:3].mean(0)[None])[0]; eC = l2(F[-3:].mean(0)[None])[0]
    C2 = np.vstack([C, sC, eC]); Pp = np.concatenate([P, [0.], [1.]])
    bins = np.unique(np.concatenate([[0.], Pp, [1.]])); nb = len(bins)
    cb = [int(np.searchsorted(bins, v)) for v in Pp]; pen = lam*np.abs(bins[:, None]-bins[None])
    de = np.linalg.norm(F[:, None]-C2[None], axis=2); em = np.full((len(F), nb), 1e3)
    for ti in range(len(Pp)): em[:, cb[ti]] = np.minimum(em[:, cb[ti]], de[:, ti])
    cost = np.full(nb, 1e9); cost[0] = em[0, 0]; BP = np.zeros((len(F), nb), int)
    for j in range(1, len(F)):
        tr = cost[None, :]+pen; kk = tr.argmin(1); cost = em[j]+tr[np.arange(nb), kk]; BP[j] = kk
    si = nb-1; path = np.zeros(len(F), int); path[-1] = si
    for j in range(len(F)-2, -1, -1): si = BP[j+1][si]; path[j] = si
    step = bins[path]; segs = []; a = 0
    for t in range(1, len(step)):
        if step[t] != step[t-1]: segs.append((a, t-1, step[t-1])); a = t
    segs.append((a, len(step)-1, step[-1])); reps = []
    for i0, i1, val in segs:
        cand = [ti for ti in range(len(Pp)) if abs(Pp[ti]-val) < 1e-9]; fr = np.arange(i0, i1+1); bd = 1e18; bf = i0
        for ti in cand:
            dd = np.linalg.norm(F[fr]-C2[ti], axis=1); k = int(dd.argmin())
            if dd[k] < bd: bd = dd[k]; bf = fr[k]
        reps.append((bf, float(val)))
    if reps[0][0] != 0: reps = [(0, float(step[0]))]+reps
    if reps[-1][0] != len(step)-1: reps = reps+[(len(step)-1, float(step[-1]))]
    rf = np.array([r[0] for r in reps]); rv = np.array([r[1] for r in reps]); keep = np.concatenate([[True], np.diff(rf) > 0])
    return np.interp(np.arange(len(step)), rf[keep], rv[keep]).astype(np.float32)


print('加载 kai base bank...', flush=True); t0 = time.time()
d = REPO/'lmvla/crave/data/kai_dinov3base'; idx = np.load(d/'index.npz'); E = idx['E']; FR = idx['FR']
feat = np.zeros((len(E), 768), np.float16)
for sh in sorted(d.glob('shard_*.npz')):
    s = np.load(sh); g = s['gidx']; v = s['valid'] if 'valid' in s else np.ones(len(g), bool); feat[g[v]] = s['feat'][v]
eps = sorted(np.unique(E).tolist())
if len(eps) > CAP: eps = [eps[i] for i in sorted(rng.choice(len(eps), CAP, replace=False))]
keep = np.isin(E, eps); E = E[keep]; FR = FR[keep]; feat = feat[keep]
print(f'  {len(eps)} eps {len(E)} frames; PCA...', flush=True)
pca = PCA(128, random_state=0).fit(l2(feat[rng.choice(len(feat), min(20000, len(feat)), replace=False)].astype(np.float32)))
IMG = l2((l2(feat.astype(np.float32))-pca.mean_.astype(np.float32))@pca.components_.astype(np.float32).T)

# ---- proprio: per-ep parquet observation.state[FR] → pos14 ----
print(f'  [{time.time()-t0:.0f}s] 读 proprio parquet...', flush=True)
POS = np.zeros((len(E), 14), np.float32)
for e in eps:
    m = np.where(E == e)[0]; o = m[np.argsort(FR[m])]; fr = FR[m][np.argsort(FR[m])]
    st = np.stack(pd.read_parquet(KAI/f'data/chunk-{e//CSQ:03d}/episode_{e:06d}.parquet',
                  columns=['observation.state'])['observation.state'].to_numpy()).astype(np.float32)
    POS[o] = st[np.minimum(fr, len(st)-1)]
SMU = POS.mean(0); SSD = POS.std(0)+1e-6
JOINT = np.concatenate([IMG, l2((POS-SMU)/SSD)], 1).astype(np.float32)  # 142D, img:pos 能量 1:1
D = JOINT.shape[1]; NC = len(eps)
T = np.zeros(len(E), np.float32)
for e in eps:
    m = np.where(E == e)[0]; o = m[np.argsort(FR[m])]; T[o] = np.linspace(0, 1, len(o))

print(f'  [{time.time()-t0:.0f}s] BayesianGMM on {D}D (img⊕proprio)...', flush=True)
bg = BayesianGaussianMixture(n_components=40, covariance_type='diag', weight_concentration_prior=1e-2, max_iter=120, random_state=0).fit(JOINT[rng.choice(len(JOINT), min(80000, len(JOINT)), replace=False)])
labs = bg.predict(JOINT); C = []; P = []
for k in range(40):
    m = labs == k
    if m.sum() < 20: continue
    if len(set(E[m].tolist()))/NC >= 0.5: C.append(JOINT[m].mean(0)); P.append(float(np.median(T[m])))
C = l2(np.array(C, np.float32)); P = np.array(P); lam = 16.*FPS/3.
print(f'  [{time.time()-t0:.0f}s] M={len(C)} milestones; polyline teacher...', flush=True)
DATA = []
for e in eps:
    m = np.where(E == e)[0]; o = m[np.argsort(FR[m])]; f = JOINT[o]; t = T[o]
    DATA.append((f, daw(f, C, P, lam), t))
tc = np.nanmean([cc(v, t) for f, v, t in DATA]); print(f'  teacher-vs-T corr={tc:.3f} ({time.time()-t0:.0f}s)', flush=True)
rng.shuffle(DATA); k = int(len(DATA)*0.85); TR = DATA[:k]; EV = DATA[k:]


class G(nn.Module):
    def __init__(s, h=256, L=2):
        super().__init__(); s.g = nn.GRU(D, h, L, batch_first=True)
        s.head = nn.Sequential(nn.Linear(h, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(s, x, ln):
        p = nn.utils.rnn.pack_padded_sequence(x, ln.cpu(), batch_first=True, enforce_sorted=False)
        o, _ = s.g(p); o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True)
        return torch.sigmoid(s.head(o)).squeeze(-1)


def batches(pool, bs=24):
    ix = sorted(range(len(pool)), key=lambda i: len(pool[i][0]))
    for kk in range(0, len(ix), bs):
        gp = [pool[i] for i in ix[kk:kk+bs]]; L = max(len(a[0]) for a in gp); B = len(gp)
        X = np.zeros((B, L, D), np.float32); Y = np.zeros((B, L), np.float32); M = np.zeros((B, L), np.float32); ln = np.zeros(B, int)
        for b, (f, v, t) in enumerate(gp): n = len(f); X[b, :n] = f; Y[b, :n] = v; M[b, :n] = 1; ln[b] = n
        yield torch.tensor(X, device=DEV), torch.tensor(Y, device=DEV), torch.tensor(M, device=DEV), torch.tensor(ln)


net = G().to(DEV); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 40)
print('训练因果 GRU(蒸馏 polyline teacher)...', flush=True)
for ep in range(40):
    net.train(); pool = TR[:]; rng.shuffle(pool)
    for X, Y, M, ln in batches(pool):
        pr = net(X, ln); loss = (((pr-Y)**2)*M).sum()/M.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    sch.step()
net.eval()


@torch.no_grad()
def pred(f, warm=20):
    fw = np.concatenate([f[:1].repeat(warm, 0), f])
    return net(torch.tensor(fw[None], device=DEV), torch.tensor([len(fw)]))[0].cpu().numpy()[warm:]


ev_corr = np.nanmean([cc(pred(f), v) for f, v, t in EV])
print(f'held-out corr(value model vs polyline teacher)={ev_corr:.3f} ({time.time()-t0:.0f}s)', flush=True)

order = sorted(range(len(EV)), key=lambda i: -len(EV[i][0]))[:12]
pick = [order[i] for i in np.linspace(0, len(order)-1, 6).astype(int)]
fig, axes = plt.subplots(2, 3, figsize=(14, 7)); axes = axes.flatten()
for ax, i in zip(axes, pick):
    f, v, t = EV[i]; p = pred(f)
    ax.plot(t, color='#e8830c', lw=1.2, alpha=.7, label='norm time')
    ax.plot(v, color='#2ca02c', lw=1.8, label='polyline teacher (de-staircased Viterbi)')
    ax.plot(p, color='#1f77ff', lw=2.1, label='value model (causal GRU, zero-future+warmup)')
    ax.set_title(f'kai held-out ep · corr={cc(p, v):.3f}', fontsize=9); ax.set_ylim(-.03, 1.03); ax.grid(alpha=.25)
axes[0].legend(fontsize=7, loc='lower right')
fig.suptitle(f'DINOv3-base img+proprio (M={len(C)}) · causal GRU distillation of DE-STAIRCASED polyline double-anchor Viterbi · kai held-out (mean corr {ev_corr:.3f})', fontsize=10)
fig.tight_layout()
outp = REPO/'lmvla/crave/temp/base_kai_gru.png'; outp.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(outp, dpi=115, bbox_inches='tight'); print('SAVED', outp, flush=True)
