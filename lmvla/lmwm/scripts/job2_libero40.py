"""Job2: LIBERO-40 全任务验证"尾巴负 lift"的普遍性 (P1 线, DINOv3-base grid)。"""
import numpy as np, torch, sys, json, pandas as pd, glob
sys.path.insert(0, "lmwm/scripts")
from p1_train_lmwm_libero import InverseEnc, MilestoneGenerator, MilestonePredictorGrid, load_grid, cosr, DIN
dev = "cuda"
ck = torch.load("lmwm/checkpoints/lmwm_libero_dinov3base/lmwm.pt", map_location="cpu", weights_only=False)
cd = ck["code_dim"]
gen = MilestoneGenerator(DIN, cd).to(dev); gen.load_state_dict(ck["gen"]); gen.eval()
prd = MilestonePredictorGrid(DIN, cd, 4).to(dev); prd.load_state_dict(ck["prd"]); prd.eval()
def deploy_code(G):
    logit, mu, ls = prd(G); return mu[torch.arange(len(G)), logit.argmax(1)]

ROOT = "/home/tim/workspace/deepdive_kai0/lmvla/lawam/dataset/libero_merged_no_noops_20hz"
t = pd.read_parquet(f"{ROOT}/meta/tasks.parquet"); desc = {int(v): k for k, v in zip(t.index, t["task_index"])}
P = np.load("lmwm/data/libero_milestone/pairs.npz")
NB, NEP = 10, 5
rows = []
for TASK in sorted(np.unique(P["pair_task"])):
    sel = P["pair_task"] == TASK
    trained = {}
    for e, fi in zip(P["cur_ep"][sel], P["cur_fi"][sel]):
        trained.setdefault(int(e), set()).add(int(fi))
    eps = np.unique(P["cur_ep"][sel])[:NEP]
    lift_b = [[] for _ in range(NB)]; cov_b = [[] for _ in range(NB)]
    cache = {}
    for e in eps:
        try: g = load_grid(cache, int(e))
        except Exception: continue
        G = torch.from_numpy(g).to(dev); N = len(G)
        last = G[-1:].expand(N, -1, -1, -1)
        with torch.no_grad():
            pd_ = gen(G, deploy_code(G)).flatten(1); lf = last.flatten(1)
            cp = cosr(G.flatten(1), lf).cpu().numpy(); cdp = cosr(pd_, lf).cpu().numpy()
        ts = trained.get(int(e), set())
        for i in range(N):
            b = min(int(i / N * NB), NB - 1)
            lift_b[b].append(cdp[i] - cp[i]); cov_b[b].append(1.0 if i in ts else 0.0)
        cache.clear()
    if not any(len(x) for x in lift_b): continue
    cov = np.array([np.mean(c) if c else 0 for c in cov_b])
    lift = np.array([np.mean(l) if l else np.nan for l in lift_b])
    untr = cov < 0.2
    rows.append(dict(task=int(TASK), desc=desc.get(int(TASK), "?")[:52],
                     tail_frac=float(untr.mean()),
                     tail_lift=float(np.nanmean(lift[untr])) if untr.any() else np.nan,
                     tr_lift=float(np.nanmean(lift[~untr])) if (~untr).any() else np.nan,
                     end_lift=float(lift[-1])))
    r = rows[-1]
    print(f"[{TASK:>2}] tail={r['tail_frac']:.2f} tail_lift={r['tail_lift']:+.4f} tr_lift={r['tr_lift']:+.4f} end={r['end_lift']:+.4f} | {r['desc']}", flush=True)

json.dump(rows, open("lmwm/outputs/job2_libero40_taillift.json", "w"), indent=1)
a = pd.DataFrame(rows)
print("\n" + "=" * 88)
print(f"任务数={len(a)}  尾巴 lift<0 的任务: {(a.tail_lift<0).sum()}/{len(a)}  ({(a.tail_lift<0).mean()*100:.0f}%)")
print(f"平均: tail_frac={a.tail_frac.mean():.2f}  tail_lift={a.tail_lift.mean():+.4f}  trained_lift={a.tr_lift.mean():+.4f}  end_lift={a.end_lift.mean():+.4f}")
print(f"Spearman(tail_frac, tail_lift) = {a.tail_frac.corr(a.tail_lift, method='spearman'):+.3f}")
print("\n最差 8 个(按 end_lift):")
for _, r in a.nsmallest(8, "end_lift").iterrows():
    print(f"  [{int(r.task):>2}] end_lift={r.end_lift:+.4f} tail={r.tail_frac:.2f} | {r.desc}")
