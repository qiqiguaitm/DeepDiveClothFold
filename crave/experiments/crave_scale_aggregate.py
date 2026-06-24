"""聚合并行跑出的规模消融 per-config 工件 → 一张簇中心对比图(左侧清晰标注 编码器/解码器/数据量)+ 合并指标。
读 temp/crave_a1a2/scale_cfg_{A..F}.npz + scale_nearest.npz。
Run: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_scale_aggregate.py [--keys A,B,C,D,E,F]

Thin entrypoint over `crave`: REPO + viz_dir from crave.config, decoder param-count via
crave.decoding.make_decoder (legacy `Dec`), Agg via crave.render.
"""
import argparse, json
import numpy as np, cv2
from pathlib import Path

from crave.config import REPO, viz_dir
from crave.decoding import make_decoder
from crave.render import setup_mpl

plt = setup_mpl()

OUTV = viz_dir("centroid_decoder"); OUTJ = REPO / "temp/crave_a1a2"

# 每配置: 编码器(模型/维度/参数量) · 解码器(规模/维度) · 数据量
ENC = {"small": ("DINOv2-small", 384, "~22M"), "base": ("DINOv2-base", 768, "~86M"), "large": ("DINOv2-large", 1024, "~300M")}
SPEC = {  # key -> (role, enc_key, dec_size, data)
    "A": ("baseline", "small", "small", "9k"),
    "B": ("+data", "small", "small", "24k"),
    "C": ("+decoder", "small", "big", "9k"),
    "D": ("+encoder", "base", "small", "9k"),
    "E": ("+encoder", "large", "small", "9k"),
    "F": ("ALL maxed", "large", "big", "24k"),
    "G": ("+decoder+data", "small", "big", "24k"),
    "H": ("+decoderXL", "small", "xl", "9k"),
    "I": ("MAX-all", "large", "xl", "24k"),
    "J": ("sweet bigEnc+smallDec", "large", "small", "24k"),
    "K": ("sweet bigEnc+smallDec", "base", "small", "24k"),
    "L": ("dec-ladder", "large", "tiny", "9k"),
    "M": ("dec-ladder", "large", "medium", "9k"),
    "N": ("dec-ladder", "large", "big", "9k"),
    "O": ("dec-ladder", "large", "xl", "9k"),
}


def dec_params(din, dec):
    return sum(p.numel() for p in make_decoder(din, dec).parameters())


def lap(x): return float(cv2.Laplacian(cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--keys", default="A,B,C,D,E,F"); a = ap.parse_args()
    keys = [k.strip() for k in a.keys.split(",")]
    near = np.load(OUTJ / "scale_nearest.npz", allow_pickle=True)
    nearest = near["nearest"]; tpos_sel = near["tpos_sel"]; NS = len(nearest)
    rows = []; merged = {}
    for k in keys:
        f = OUTJ / f"scale_cfg_{k}.npz"
        if not f.exists(): print(f"  [skip] {f} 缺失"); continue
        z = np.load(f, allow_pickle=True); m = json.loads(str(z["metrics"]))
        role, enck, decsz, data = SPEC[k]; ename, edim, eparams = ENC[enck]
        dp = dec_params(edim, decsz)
        # 左侧清晰描述
        label = (f"{k}  {role}\n"
                 f"Enc: {ename} {edim}d ({eparams})\n"
                 f"Dec: {decsz}  {dp/1e6:.1f}M\n"
                 f"Data: {data} frames\n"
                 f"-> centroid={m['centroid_sharp']}  recon={m['recon_sharp']}")
        rows.append((label, z["cents"], m)); merged[f"{k} {role}"] = {**m, "enc": ename, "enc_dim": edim, "dec": decsz, "dec_params_M": round(dp / 1e6, 2), "data": data}
    merged["nearest_real_sharp"] = round(float(np.mean([lap(x) for x in nearest])), 1)
    json.dump(merged, open(OUTJ / "scale_ablation_merged.json", "w"), indent=2, ensure_ascii=False)
    print("MERGED METRICS", json.dumps(merged, ensure_ascii=False, indent=1))

    nrow = len(rows) + 1
    fig, axes = plt.subplots(nrow, NS, figsize=(1.55 * NS + 3.0, 1.75 * nrow))
    plt.subplots_adjust(left=0.26)
    for r, (label, cents, m) in enumerate(rows):
        for j in range(NS):
            ax = axes[r, j]; ax.imshow(cents[j]); ax.axis("off")
            if r == 0: ax.set_title(f"P={tpos_sel[j]:.2f}", fontsize=8)
        axes[r, 0].set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center", labelpad=6, linespacing=1.5)
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for sp in axes[r, 0].spines.values(): sp.set_visible(False)
    for j in range(NS):
        ax = axes[-1, j]; ax.imshow(nearest[j]); ax.axis("off")
    axes[-1, 0].set_ylabel(f"nearest real frame\n(current method)\n-> sharp={merged['nearest_real_sharp']}",
                           fontsize=8, rotation=0, ha="right", va="center", labelpad=6, linespacing=1.5, color="#1a7f37")
    axes[-1, 0].axis("on"); axes[-1, 0].set_xticks([]); axes[-1, 0].set_yticks([])
    for sp in axes[-1, 0].spines.values(): sp.set_visible(False)
    fig.suptitle("Scale levers on cluster CENTROID (grid-average) — each row: Encoder / Decoder / Data spelled out.  "
                 "Higher 'sharp' = crisper; only nearest-real is crisp; centroid stays soft regardless of scale.", fontsize=12)
    fig.savefig(OUTV / "crave_scale_ablation.png", dpi=125, bbox_inches="tight"); plt.close(fig)
    print("SAVED crave_scale_ablation.png")


if __name__ == "__main__":
    main()
