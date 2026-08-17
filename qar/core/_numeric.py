"""Small numeric utilities shared across the package."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["scalarize", "safe_where"]


def scalarize(x: Any) -> Any:
    """Collapse a 0-d array back to a Python scalar.

    ``np.where`` and friends return 0-d arrays even for scalar inputs, so
    ``delta(...)`` would otherwise print as ``array(0.6368)``. Array inputs pass
    through untouched, and complex 0-d arrays come back as Python ``complex`` —
    which the complex-step validator relies on.
    """
    arr = np.asarray(x)
    if arr.ndim == 0:
        return arr.item()
    return arr


def safe_where(condition: Any, limit: Any, regular: Any) -> Any:
    """``np.where`` with floating-point warnings suppressed.

    Both branches of ``np.where`` are always evaluated, so the degenerate-limit
    pattern used throughout :mod:`qar.greeks` computes things like ``0 * inf``
    in the branch it is about to discard. That is harmless but noisy, hence the
    ``errstate``.
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.where(condition, limit, regular)
