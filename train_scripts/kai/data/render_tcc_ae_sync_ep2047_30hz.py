"""ep2047 (kai0_base) value × 视频同步: 上=top_head 实时帧, 下=离散CRAVE/TCC连续/pi0-AE 三曲线+游标。
用 temp/_solve_ep2047_30hz.npz + kai0_base 视频。输出 temp/tcc_ae_kai0base_ep2047_30hz_sync.mp4
"""
import numpy as np, av, matplotlib, os
matplotlib.use("Agg")
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
from pathlib import Path
_sh = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/SimHei.ttf")
if os.path.exists(_sh): fm.fontManager.addfont(_sh)
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
z = np.load(REPO / "temp/_solve_ep2047_30hz.npz")
crave, tcc, ae, x = z["crave"], z["tcc"], z["ae"], z["x"]
NF = len(crave)
VID = REPO / "kai0/data/Task_A/kai0_base/videos/chunk-002/observation.images.top_head/episode_002047.mp4"
OUT = REPO / "temp/tcc_ae_kai0base_ep2047_30hz_sync.mp4"

MAXSIDE = 460
def stream(path):
    c = av.open(str(path))
    for f in c.decode(video=0):
        s = min(1.0, MAXSIDE / max(f.height, f.width))
        g = f.reformat(width=int(f.width*s)//2*2, height=int(f.height*s)//2*2, format="rgb24") if s < 1 else f
        yield g.to_ndarray(format="rgb24")
    c.close()
frame0 = next(stream(VID))

fig = plt.figure(figsize=(11, 8.6))
gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1.0], hspace=0.2)
axc = fig.add_subplot(gs[0]); axc.axis("off"); im = axc.imshow(frame0)
ttl = axc.set_title("", fontsize=11)
axv = fig.add_subplot(gs[1])
axv.step(x, crave, where="post", color="#1f77b4", lw=2.0, label=f"离散 CRAVE (end{crave[-1]:.2f})")
axv.plot(x, tcc, color="#2ca02c", lw=2.0, label=f"TCC 连续 (end{tcc[-1]:.2f}, 30Hz)")
axv.plot(x, ae, color="#d62728", lw=1.6, alpha=.85, label=f"pi0-AE 监督 (end{ae[-1]:.2f}, 30Hz)")
axv.axhline(1, color="#999", ls=":", lw=1)
cur = axv.axvline(0, color="k", lw=1.4)
dotc, = axv.plot([0], [crave[0]], "o", color="#1f77b4", ms=7, mec="k")
dott, = axv.plot([0], [tcc[0]], "o", color="#2ca02c", ms=7, mec="k")
dota, = axv.plot([0], [ae[0]], "o", color="#d62728", ms=7, mec="k")
axv.set_xlim(0, NF); axv.set_ylim(-0.05, 1.12); axv.set_xlabel("frame"); axv.set_ylabel("value")
axv.grid(alpha=.25); axv.legend(fontsize=9, loc="upper left")
fig.suptitle("kai0_base ep2047: 离散CRAVE vs TCC连续 vs pi0-AE监督 — value×视频同步", fontsize=12, y=0.97)

oc = av.open(str(OUT), mode="w"); st = oc.add_stream("libx264", rate=30); f0 = None
t = 0
for img in stream(VID):
    if t >= NF: break
    im.set_data(img)
    ttl.set_text(f"frame {t}/{NF}  CRAVE={crave[t]:.2f}  TCC={tcc[t]:.2f}  AE={ae[t]:.2f}")
    cur.set_xdata([t, t]); dotc.set_data([t], [crave[t]]); dott.set_data([t], [tcc[t]]); dota.set_data([t], [ae[t]])
    fig.canvas.draw()
    arr = np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    if f0 is None:
        H, W = arr.shape[:2]; H -= H % 2; W -= W % 2; st.width, st.height, st.pix_fmt = W, H, "yuv420p"; st.options = {"crf": "21"}; f0 = True
    for pkt in st.encode(av.VideoFrame.from_ndarray(arr[:H, :W], format="rgb24")): oc.mux(pkt)
    t += 1
    if t % 600 == 0: print(f"  {t}/{NF}", flush=True)
for pkt in st.encode(): oc.mux(pkt)
oc.close(); print(f"SAVED {OUT} {W}x{H} {t}f", flush=True)
