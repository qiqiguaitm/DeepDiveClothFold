#!/usr/bin/env python
"""Distribution plot: LMWM (ours, FINAL center_w=0.1) vs LaWM prediction time-lag. Overlaid histograms
of the per-frame model_lag (s), with means + design horizons marked, to visualize the difference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ours = np.load(REPO / "lmwm/outputs/lag_raw_cw01.npy")                        # FINAL center_w=0.1 (twomodel_final.pt)
lawm = np.load(REPO / "lmwm/outputs/lag_raw_lawm.npy")

O_HZ, L_HZ = 2.64, 1.66                                                      # dataset horizons
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))

bins = np.linspace(-2, 6, 49)
for a, title, xr in [(ax[0], "full range", (-2, 6)), (ax[1], "zoom [-1, 3]", (-1, 3))]:
    a.hist(ours, bins=bins, density=True, alpha=0.55, color="#7c3aed", label=f"LMWM final cw=0.1 (μ={ours.mean():.2f}s)")
    a.hist(lawm, bins=bins, density=True, alpha=0.55, color="#f59e0b", label=f"LaWM+predm (μ={lawm.mean():.2f}s)")
    a.axvline(ours.mean(), color="#7c3aed", ls="-", lw=1.6); a.axvline(lawm.mean(), color="#f59e0b", ls="-", lw=1.6)
    a.axvline(O_HZ, color="#7c3aed", ls="--", lw=1.2, alpha=.7); a.axvline(L_HZ, color="#f59e0b", ls="--", lw=1.2, alpha=.7)
    a.axvline(0, color="#888", ls=":", lw=1)
    a.set_xlim(*xr); a.set_xlabel("model prediction lag (s)"); a.set_ylabel("density"); a.set_title(title)
    a.legend(fontsize=9); a.grid(alpha=.2)

txt = (f"LMWM(ours): mean {ours.mean():.2f}s  median {np.median(ours):.2f}  "
       f">0 {(ours>0).mean()*100:.0f}%  <0 {(ours<0).mean()*100:.0f}%  horizon {O_HZ}s (dashed)\n"
       f"LaWM      : mean {lawm.mean():.2f}s  median {np.median(lawm):.2f}  "
       f">0 {(lawm>0).mean()*100:.0f}%  <0 {(lawm<0).mean()*100:.0f}%  horizon {L_HZ}s (dashed)  "
       f"| solid=mean reach, dashed=target horizon")
fig.suptitle("Prediction time-lag distribution: LMWM final (milestone 2.6s) vs LaWM (near-future 1.6s)", fontsize=12)
fig.text(0.5, -0.02, txt, ha="center", fontsize=8.5, family="monospace")
fig.tight_layout()
out = REPO / "lmwm/outputs/lag_distribution.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved {out}\nLMWM mean {ours.mean():.3f} median {np.median(ours):.3f} | LaWM mean {lawm.mean():.3f} median {np.median(lawm):.3f}")
