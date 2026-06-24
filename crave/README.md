# CRAVE

**Cross-episode Recurrence As Value Estimation** — zero-training milestone discovery and
progress/value estimation from frozen visual features.

Pipeline: `frames → encoder (pluggable) → cluster → order milestones (precedence/isotonic)
→ readout value (Viterbi-DP)`.

## Layout (src-layout, mirrors kai0/)
```
crave/
├── pyproject.toml
├── MIGRATION.md            # how the legacy crave_*.py scripts map onto this library
├── src/crave/
│   ├── config/             # paths (REPO/TEMP/DOCS/out_dir/viz_dir), encoders registry, datasets registry
│   ├── encoders/           # load_encoder(name) -> Encoder{encode_pooled, encode_grid}  (dinov2/dinov3/wan)
│   ├── data/               # dataset loaders (lerobot2/hdf5/lerobotv3), kai0-family access, feature caches
│   ├── clustering/         # gpu/cpu kmeans, milestone selection/ordering, build_clusters
│   ├── value/              # FeatureSpace, DiscreteValue (V2.4), ContinuousValue (TCC), readout variants
│   ├── decoding/           # centroid decoder (grid -> image)
│   ├── render/             # matplotlib setup (Agg+SimHei), VideoWriter
│   └── utils/              # mkp/med/L2/otsu/viterbi/viterbi_forward/smooth_monotone
├── scripts/                # reference entrypoints (generalize.py)
└── experiments/            # the migrated one-off experiment/diagnostic scripts
```

## Install / run
```bash
# runs inside the `srpo` env (torch 2.10 + transformers 4.57, DINOv3-capable):
/home/tim/miniconda3/envs/srpo/bin/python -m pip install -e crave/
/home/tim/miniconda3/envs/srpo/bin/python crave/scripts/generalize.py coffee --encoder dinov3-h
```

## Encoders (pluggable)
`dinov2-small|base|large`, `dinov3-h`, `dinov3-7b`, `dinov3-7b-int8`, `wan-vae`.
Add one by adding a row to `crave.config.encoders.ENCODERS`. DINOv3 runs in bf16
(fp16 overflows) at res 256 → 16×16 grid (same as dinov2 @224); 1 CLS + 4 register
tokens are skipped before pooling.

```python
from crave.encoders import load_encoder
enc = load_encoder("dinov3-h")
pooled = enc.encode_pooled(frames)   # (N, dim)
grids  = enc.encode_grid(frames)     # (N, dim, 16, 16)
```

## Parity
`crave.value.DiscreteValue` is byte-identical to the legacy `crave_value.DiscreteValue`
(verified max|Δvalue|=0), and `crave/scripts/generalize.py` reproduces the legacy
`crave_generalize.py` exactly (same milestone count and value range). The legacy logic was
ported verbatim, not reimplemented.
