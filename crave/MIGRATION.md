# CRAVE library — migration guide

The scattered `train_scripts/kai/data/crave_*.py` scripts are being rewritten onto the
`crave` package. Each rewritten script becomes a **thin entrypoint**: it imports shared
logic from `crave` and keeps only its own orchestration + plotting.

Reference exemplar: **`crave/scripts/generalize.py`** (rewrite of `crave_generalize.py`).

## Install / run
```bash
# editable install already done in the `srpo` env:
/home/tim/miniconda3/envs/srpo/bin/python -m pip install -e crave/
# run a script:
/home/tim/miniconda3/envs/srpo/bin/python crave/experiments/<name>.py [args]
```
Run everything with the **`srpo`** interpreter (torch 2.10 + transformers 4.57, DINOv3-capable).
Do NOT append the kai0 venv to `sys.path` (it shadows transformers).

## Package layout
```
crave/src/crave/
  config/    paths (REPO, TEMP, HF_HUB, out_dir), encoders (ENCODERS, resolve), datasets (DATASETS, resolve_dataset)
  encoders/  load_encoder(name) -> Encoder{.encode_pooled(imgs)->(N,dim), .encode_grid(imgs)->(N,dim,P,P)}
  data/      loadep, list_cache_eps (feat cache);  list_eps, load_ep, load_ep_native (raw datasets)
  clustering/ gpu_kmeans, cpu_kmeans, cluster_stats, first_arrival_matrix, precedence_order, runs
  value/     FeatureSpace, DiscreteValue, ContinuousValue
  decoding/  make_decoder, train_dec
  render/    setup_mpl() -> plt (Agg+SimHei),  VideoWriter
  utils/     L2, mkp, mkp_gap, med, advantage, mono, adv_density, otsu,
             viterbi, viterbi_forward, forward_penalty, smooth_monotone
```

## Old inline helper → new import (the mapping)
| old (inline / from crave_*)                    | new |
|------------------------------------------------|-----|
| `def mkp / mkp_gap / med / L2 / otsu`          | `from crave.utils import mkp, mkp_gap, med, L2, otsu` |
| `def viterbi(...)`                             | `from crave.utils import viterbi`  (returns `(values, path)`) |
| forward-biased DP (`PEN = where(dv>=0,3*dv,25*-dv)` + hard-start loop) | `from crave.utils import viterbi_forward, forward_penalty` |
| `from crave_readout import smooth_monotone`    | `from crave.utils import smooth_monotone` |
| `def gpu_kmeans(...)`                           | `from crave.clustering import gpu_kmeans` |
| `from crave_value import FeatureSpace, DiscreteValue, loadep` | `from crave.value import FeatureSpace, DiscreteValue` ; `from crave.data import loadep` |
| `from crave_decoder_scale_ablation import REPO, encode_grids, train_dec, Dec` | `from crave.config import REPO` ; `enc.encode_grid(imgs)` ; `from crave.decoding import train_dec, make_decoder` |
| `from crave_generalize import CFG, load_ep, load_ep_native, list_eps` | `from crave.config import resolve_dataset` ; `from crave.data import load_ep, load_ep_native, list_eps` |
| `make_enc()/enc_pooled()/encode_grids()` (DINOv2 only) | `enc = load_encoder("dinov2-large")` then `enc.encode_pooled(...)` / `enc.encode_grid(...)` |
| `matplotlib.use("Agg")` + SimHei boilerplate   | `from crave.render import setup_mpl; plt = setup_mpl()` |
| PyAV/ffmpeg video writing                      | `from crave.render import VideoWriter` (or keep the script's ffmpeg call) |
| `REPO = Path("/vePFS/.../deepdive_kai0")`      | `from crave.config import REPO` |
| `LARGE = ".../dinov2-large"; DIM=1024`         | `enc = load_encoder("dinov2-large"); enc.dim` |

Notes:
- `CFG[ds]` (a dict) is now `resolve_dataset(ds)` (a `DatasetConfig` dataclass): `cfg["root"]` → `cfg.root`.
- Encoders are selectable: `dinov2-{small,base,large}`, `dinov3-h`, `dinov3-7b`, `dinov3-7b-int8`, `wan-vae`.
  DINOv3 needs bf16 (handled internally) and feeds at res 256 → 16×16 grid (same as dinov2@224).
- `crave.utils.viterbi` returns `(values, path)`; the legacy `viterbi` in crave_value returned the same, the one in crave_generalize returned only values — adjust call sites.

## Rewrite recipe (per script)
1. Destination: `crave/experiments/<same_basename>.py`.
2. Delete inlined helper defs that now live in `crave.utils` / `crave.clustering` / etc.; import them instead.
3. Replace encoder-loading blocks with `load_encoder(...)`.
4. Replace hardcoded `REPO`/paths/`CFG` with `crave.config` symbols.
5. Keep the script's unique logic + plotting; wrap `__main__` with argparse if it took argv.
6. Preserve behavior exactly — same numbers, same outputs. Do not "improve" the algorithm.
7. Verify it at least imports: `python -c "import ast,py_compile; py_compile.compile('<path>', doraise=True)"`
   and, when feasible without heavy data, a dry import.

## Newly added library symbols (use these instead of re-inlining)
Wave-1 surfaced these gaps; they now exist in the library:

| need | new import |
|------|-----------|
| full-frame cluster→select→precedence milestones (`cl` dict) | `from crave.clustering import build_clusters, BINS` |
| per-episode readouts over a `cl` dict (production/direct/viterbi_ms) | `from crave.value import readout_production, readout_direct, readout_viterbi_ms` |
| kai0-family dataset (kai0_base, smooth800_dagger) | `from crave.config import resolve_dataset` → kind=="kai0" |
| kai0 raw-frame access (camp/crop224/grab_ep/decode_images/lpst) | `from crave.data import kai0` → `kai0.grab_ep(cfg,e,frames)`, `kai0.decode_images(cfg,idx,E,FR)`, `kai0.state_subsampled(cfg,e,n)`, `kai0.crop224(rgb)`, `kai0.chunks_size(cfg.root)` |
| kai0 3-path tcc cache reader (key "f") | `from crave.data import kai0` → `kai0.loadep_tcc(cfg, e)` → (armmask, raw, state, n) |
| full-scale dino shard cache (temp/crave_full) | `from crave.data import load_dino_shards` → `(feats, index)` |
| fp32 / hub-id encoder for byte-identical legacy scoring | `load_encoder("dinov2-small", dtype="fp32", path="facebook/dinov2-small")` |
| Wan VAE decode / raw latents | `enc = load_encoder("wan-vae")` → `enc.encode_latents(imgs)`, `enc.decode(latents)` |
| durable viz output dir under docs/ | `from crave.config import viz_dir` → `viz_dir("centroid_decoder")` |

The exemplar `crave/scripts/generalize.py` was verified to reproduce the legacy
`crave_generalize.py` byte-for-byte (M, value range identical), and `crave.value.DiscreteValue`
matches legacy `crave_value.DiscreteValue` exactly. So importing these is safe.
