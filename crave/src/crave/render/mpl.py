"""Matplotlib setup shared by every CRAVE visualizer (headless Agg + SimHei CJK font)."""
from __future__ import annotations

import os


def setup_mpl():
    """Configure a headless Agg backend with Chinese-capable fonts. Returns pyplot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    sh = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/SimHei.ttf")
    if os.path.exists(sh):
        fm.fontManager.addfont(sh)
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt
