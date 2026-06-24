"""Centroid decoding: train a small CNN to render patch-grid centroids as prototype frames."""
from crave.decoding.decoder import make_decoder, train_dec

__all__ = ["make_decoder", "train_dec"]
