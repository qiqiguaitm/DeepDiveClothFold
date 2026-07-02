"""Backward-compatibility shim.

Model definitions moved to :mod:`lmwm.models`. This module re-exports the two
names historically imported from ``lmwm.model`` so existing callers keep working.
Prefer importing from :mod:`lmwm.models` in new code.
"""

from __future__ import annotations

from lmwm.models import MLP, UnifiedLMWM

__all__ = ["MLP", "UnifiedLMWM"]
