"""诊断 coffee ep0 milestone 0->5->1: ep0起始 vs ep0末帧 vs ep1末帧 大图对比。

Thin entrypoint over `crave`: `load_ep_native` comes from `crave.data`, the coffee
DatasetConfig from `crave.config.resolve_dataset`, REPO from `crave.config`.

跑法: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_coffee_ep0_diag.py
"""
from crave.config import REPO, resolve_dataset
from crave.data import load_ep_native
from crave.render import setup_mpl

cfg = resolve_dataset("coffee")
f0, _, _ = load_ep_native(cfg, 0); n0 = len(f0)
f1, _, _ = load_ep_native(cfg, 1); n1 = len(f1)
items = [(f0[int(n0*0.03)], "ep0 起始 t=0.03\n(milestone 0, value 0)"),
         (f0[int(n0*0.45)], "ep0 峰值 t=0.45\n(milestone 5, value 0.3)"),
         (f0[min(int(n0*0.97), n0-1)], "ep0 末帧 t=0.97\n(崩回 milestone 1, value 0.05)"),
         (f1[min(int(n1*0.97), n1-1)], "ep1 末帧 t=0.97\n(milestone 10, value 0.93 真完成)")]
plt = setup_mpl()
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
for ax, (im, t) in zip(axes, items):
    ax.imshow(im); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=11)
fig.suptitle("coffee ep0 末帧是否别名回起始态? (ep0末 vs ep0起 vs ep1末)", fontsize=13)
out = REPO / "temp/crave_align/coffee_ep0_frames_diag.png"
fig.tight_layout(); fig.savefig(out, dpi=115, bbox_inches="tight"); plt.close(fig); print("SAVED", out)
