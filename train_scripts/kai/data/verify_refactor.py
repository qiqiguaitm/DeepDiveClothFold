"""验证重构: crave_value 模块在 ep2047 上重算 离散+连续 value, 与重构前保存的数组对比。
离散应 bit 级一致(确定性); 连续应高度一致(同 seed)。
"""
import sys, os, json
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crave_value import FeatureSpace, DiscreteValue, ContinuousValue, loadep, mono, adv_density
from hdf5_v24_eval import build_model  # 重构前的离散实现(对照)

R = Path("/vePFS/tim/workspace/deepdive_kai0")
FC = R / "temp/crave_kai0bd/feat_cache"; TEST = 2047
eps = sorted(int(p.stem[2:]) for p in FC.glob("ep*.npz"))
a, r, s, n = loadep(FC, TEST)

# ===== 离散: 新模块 select="fixed" vs 旧 build_model (必须 bit 级一致 → 硬断言锁定防回归) =====
# select="fixed" 是历史默认选择规则(每进度bin top-2 by coverage); 新增的 select="adaptive" 不得影响它。
fs = FeatureSpace(FC, eps)
dv = DiscreteValue(fs, eps, select="fixed", log=lambda *_: None)
new_disc = dv.value(a, r, s)
old_value, old_Pord = build_model(FC, eps, eps, log=lambda *_: None)
old_disc = old_value(a, r, s)
d_max = float(np.max(np.abs(new_disc - old_disc)))
pord_ok = bool(np.array_equal(dv.Pord, old_Pord))
print(f"[离散] select=fixed vs 旧build_model: max|Δ|={d_max:.2e}  Pord一致={pord_ok}  milestones={len(dv.order)}/{len(old_Pord)}  {'✅一致' if d_max < 1e-9 else '❌不一致'}")
assert len(dv.order) == len(old_Pord), f"REGRESSION: select='fixed' milestone 数 {len(dv.order)} ≠ 历史 {len(old_Pord)}"
assert pord_ok, "REGRESSION: select='fixed' 的 Pord 与历史 build_model 不再一致"
assert d_max < 1e-9, f"REGRESSION: select='fixed' 离散 value 与历史 build_model 不再 bit 级一致 (max|Δ|={d_max:.2e})"
# 同时确认 default 即 fixed(不显式传 select 也走历史路径)
assert np.array_equal(DiscreteValue(fs, eps, log=lambda *_: None).value(a, r, s), new_disc), \
    "REGRESSION: 默认 select 不再等于 'fixed'(默认行为被改动)"
print(f"[断言] ✅ 锁定: select='fixed'(=默认)与历史 build_model bit 级一致 ({len(dv.order)} milestones)")

# ===== adaptive 分支 smoke: 能跑、产合法 value、且确实是另一条路径(不污染 fixed) =====
dv_ada = DiscreteValue(fs, eps, select="adaptive", nbins=10, cap_pb=3, tau_q=0.5, log=lambda *_: None)
v_ada = dv_ada.value(a, r, s)
assert dv_ada.tau is not None and v_ada.shape == new_disc.shape and np.all((v_ada >= -1e-6) & (v_ada <= 1 + 1e-6)), "adaptive 分支输出非法"
assert np.array_equal(dv.value(a, r, s), new_disc), "REGRESSION: 构建 adaptive 后 fixed 结果发生变化(状态串扰)"
print(f"[smoke] ✅ select='adaptive' 可运行 (tau={dv_ada.tau:.2f}, {len(dv_ada.order)} milestones), 且不影响 fixed")

# ===== 离散: 新模块 vs 重构前保存的 30Hz crave (_solve_ep2047_30hz) =====
sav = np.load(R / "temp/_solve_ep2047_30hz.npz"); saved_crave = sav["crave"]
new_crave30 = np.repeat(new_disc, 10)[:len(saved_crave)]
if len(new_crave30) < len(saved_crave): new_crave30 = np.concatenate([new_crave30, np.full(len(saved_crave) - len(new_crave30), new_crave30[-1])])
m = min(len(new_crave30), len(saved_crave))
print(f"[离散30Hz] 新模块 vs 保存crave: max|Δ|={float(np.max(np.abs(new_crave30[:m]-saved_crave[:m]))):.2e}")

# ===== 连续: 新模块 vs 重构前保存的 TCC连续(从 _sim 重算的图53/3way口径) =====
# 重构前 ep2047 连续是从端到端 sim 场算的; 这里用 crave_kai0bd frozen-TCC 重算并自检稳定性(同 seed 两次)
cv = ContinuousValue(fs, eps, n_refs=30, seed=0).train_head(steps=1200, log=lambda *_: None)
new_cont = cv.value(a, r, s)
cv2 = ContinuousValue(fs, eps, n_refs=30, seed=0).train_head(steps=1200, log=lambda *_: None)
cont2 = cv2.value(a, r, s)
from scipy.stats import pearsonr
print(f"[连续] 同seed两次自洽: max|Δ|={float(np.max(np.abs(new_cont-cont2))):.2e}  corr={pearsonr(new_cont,cont2)[0]:.4f}")
print(f"[连续] 指标: end{new_cont[-1]:.2f} 单调{mono(new_cont):.0%} adv密度{adv_density(new_cont):.0%}")
np.savez(R / "temp/_verify_ep2047.npz", disc=new_disc, cont=new_cont)
print("DONE")
