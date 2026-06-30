"""Research-style CRAVE pipeline figure for the Feishu 摘要 (encode→cluster→select→Viterbi→value + decoder branch)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import numpy as np
from crave.config import REPO
from crave.render import setup_mpl
plt = setup_mpl()
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(13.2, 5.4)); ax.set_xlim(0, 1200); ax.set_ylim(492, 0); ax.axis("off")
ax.text(600, 26, "CRAVE 全程零训练流程:encode → cluster → select → Viterbi-DP 读出 value",
        ha="center", fontsize=16, fontweight="bold", color="#22303f")
ax.text(600, 48, "frozen encoder + KMeans + 动态规划 —— 无梯度更新、无人工 stage 标注",
        ha="center", fontsize=11, color="#7c8794", style="italic")

BY, BH, BW, PITCH, X0 = 86, 116, 152, 198, 14
cy = BY + BH / 2
boxes = [
    ("① 输入 demo", ["多 episode 视频帧", "原生帧率 30 / 50Hz"], "#eef1f4", "#7a8794", "#243240"),
    ("② 编码器 (frozen)", ["frozen DINOv3-H", "整幅 pooled + proprio", "→ 逐帧特征 F"], "#cfe0f1", "#3a78b5", "#1c3a5e"),
    ("③ 聚类", ["KMeans  K0≈c·sqrt(N)", "每簇 = 跨 demo", "复现态"], "#eaeef2", "#7a8794", "#243240"),
    ("④ 选簇", ["覆盖率 + Otsu 阈值", "+ 均匀化补洞", "→ milestones C, Pord"], "#eaeef2", "#7a8794", "#243240"),
    ("⑤ Viterbi-DP 读出", ["emission |F−C|", "+ 双边进度先验", "+ 置信度门控"], "#dbe8f8", "#3f6fa6", "#1c3a5e"),
    ("⑥ value[t] = 0→1", ["单调进度", "advantage 信号"], "#d8efe0", "#2e9c5b", "#155e36"),
]
xs = [X0 + i * PITCH for i in range(6)]
for i, (title, subs, fill, stroke, tcol) in enumerate(boxes):
    x = xs[i]; xc = x + BW / 2
    ax.add_patch(FancyBboxPatch((x, BY), BW, BH, boxstyle="round,pad=0,rounding_size=10",
                                fc=fill, ec=stroke, lw=1.7))
    ax.text(xc, BY + 22, title, ha="center", fontsize=12.5, fontweight="bold", color=tcol)
    ax.plot([x + 14, x + BW - 14], [BY + 31, BY + 31], color=stroke, lw=0.8, alpha=0.5)
    for j, s in enumerate(subs):
        ax.text(xc, BY + 50 + j * 18, s, ha="center", fontsize=10.3, color="#46535f")
# mini value curve in output box
ox = xs[5]
cv = np.array([[ox+20,192],[ox+44,184],[ox+64,186],[ox+86,169],[ox+108,160],[ox+128,150],[ox+142,146]])
ax.plot(cv[:, 0], cv[:, 1], color="#2e9c5b", lw=2)

arr_lab = ["T 帧", "N×D 特征", "K0 候选簇", "M 个 milestone", "value[t]"]
for i in range(5):
    x1 = xs[i] + BW; x2 = xs[i + 1]; mid = (x1 + x2) / 2
    ax.annotate("", xy=(x2 - 3, cy), xytext=(x1 + 2, cy),
                arrowprops=dict(arrowstyle="-|>", color="#5d6b78", lw=1.8, shrinkA=0, shrinkB=0))
    ax.text(mid, cy - 9, arr_lab[i], ha="center", fontsize=9.3, style="italic", color="#8693a0")

# decoder branch (dashed)
dx, dy, dw, dh = 472, 320, 470, 120
ax.add_patch(FancyBboxPatch((dx, dy), dw, dh, boxstyle="round,pad=0,rounding_size=10",
                            fc="#f5efdc", ec="#c79a3e", lw=1.7, ls=(0, (6, 4))))
ax.text(dx + dw / 2, dy + 24, "解码器支路 · 可解释性(不参与 value 计算)", ha="center", fontsize=12, fontweight="bold", color="#7a5a16")
ax.text(dx + dw / 2, dy + 49, "每个簇心的 grid 特征 → 轻量解码器 train_dec(小 CNN, 55ep)", ha="center", fontsize=10.6, color="#5d5128")
ax.text(dx + dw / 2, dy + 71, "→ decoded centroid 原型图 → milestone 词表 / gallery(§4)", ha="center", fontsize=10.6, color="#5d5128")
ax.text(dx + dw / 2, dy + 95, '"像它,故是这个值" —— 每个 value 可指向一张原型,全程可检视', ha="center", fontsize=9.8, style="italic", color="#9a8347")
bx = xs[3] + BW / 2
ax.annotate("", xy=(bx, dy - 2), xytext=(bx, BY + BH),
            arrowprops=dict(arrowstyle="-|>", color="#c79a3e", lw=1.6, ls=(0, (6, 4)), shrinkA=0, shrinkB=0))
ax.text(bx + 8, (BY + BH + dy) / 2, "簇心 C", fontsize=9.3, style="italic", color="#b08a3e")

ax.text(14, 470, "（frozen = 零训练:无梯度更新、无人工标注  |  实线 = value 主路径  |  虚线 = 可视化支路）",
        fontsize=10.3, color="#5d6b78")
fig.tight_layout()
out = REPO / "crave/docs/visualization/cross_dataset/crave_pipeline_overview.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("SAVED", out)
