"""Forecast evaluation: robust losses, significance tests, walk-forward splitting."""

from qar.evaluation.loss import LOSSES, loss_table, mse, qlike
from qar.evaluation.tests import (
    DieboldMarianoResult,
    MincerZarnowitzResult,
    diebold_mariano,
    mincer_zarnowitz,
    newey_west_variance,
)
from qar.evaluation.walkforward import (
    Split,
    WalkForwardResult,
    align_results,
    run_walk_forward,
    walk_forward_splits,
)

__all__ = [
    "LOSSES", "loss_table", "mse", "qlike",
    "DieboldMarianoResult", "MincerZarnowitzResult", "diebold_mariano",
    "mincer_zarnowitz", "newey_west_variance",
    "Split", "WalkForwardResult", "align_results", "run_walk_forward",
    "walk_forward_splits",
]
