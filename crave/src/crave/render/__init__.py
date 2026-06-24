"""Rendering helpers: matplotlib setup + video writing. (Gallery/value-plot composers
live in the experiment scripts that need them and import from here.)"""
from crave.render.mpl import setup_mpl
from crave.render.video import VideoWriter

__all__ = ["setup_mpl", "VideoWriter"]
