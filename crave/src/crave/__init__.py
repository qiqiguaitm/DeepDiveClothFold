"""CRAVE — Cross-episode Recurrence As Value Estimation.

Zero-training milestone discovery and progress/value estimation from frozen visual
features. The pipeline is: frames → encoder (pluggable) → cluster → order milestones
(precedence/isotonic) → readout value (Viterbi-DP).

Quick start:

    from crave.encoders import load_encoder
    from crave.clustering import gpu_kmeans
    from crave.value import FeatureSpace, DiscreteValue

    enc = load_encoder("dinov3-h")          # encoder-agnostic; see crave.config.ENCODERS
    pooled = enc.encode_pooled(frames)      # (N, dim)

Layers: config (paths/encoders/datasets) · encoders · data · clustering · value ·
decoding · render · utils.
"""
from __future__ import annotations

__version__ = "0.1.0"

from crave.encoders import load_encoder

__all__ = ["load_encoder", "__version__"]
