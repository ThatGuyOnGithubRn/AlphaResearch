r"""Walk-forward evaluation — the only honest way to compare these models.

Fitting on the whole sample and reporting in-sample fit measures nothing: every
model can be made to fit history. The evaluator here refits at each step on data
strictly preceding the observation being predicted, so every reported number is
genuinely out of sample.

Three leaks this design is built to prevent, all of which are easy to introduce
and produce dramatically flattering results:

1.  **Look-ahead in the fit.** ``fit`` receives a slice ending at ``t``; the
    target is at ``t+1``. The slice is constructed here, not by the model.
2.  **Look-ahead in preprocessing.** Standardisation, detrending or outlier
    removal computed over the full sample leaks the future into the training
    window. Any such step belongs inside ``fit``.
3.  **Survivorship in model selection.** Choosing which models to report *after*
    seeing the out-of-sample results reintroduces the bias the whole exercise
    exists to avoid. Fix the model list before running.

Expanding windows use all history and suit stable processes; rolling windows
discard old data and adapt faster to regime change. Both are provided because
which one wins is an empirical question, and one that part 3 should answer
rather than assume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence

import numpy as np

from qar.models.base import Forecaster

__all__ = ["Split", "walk_forward_splits", "WalkForwardResult", "run_walk_forward"]


@dataclass(frozen=True)
class Split:
    """One train/predict step: fit on ``[start, end)``, predict at ``end``."""

    start: int
    end: int

    @property
    def train_length(self) -> int:
        return self.end - self.start


def walk_forward_splits(
    n_observations: int,
    min_train: int = 250,
    expanding: bool = True,
    window: int | None = None,
    step: int = 1,
) -> Iterator[Split]:
    """Generate train/predict splits.

    Parameters
    ----------
    min_train:
        Observations required before the first forecast. 250 is roughly a year
        of daily data — below that a GARCH fit is not meaningful.
    expanding:
        ``True`` grows the training window from the start of the sample;
        ``False`` rolls a fixed-length window.
    window:
        Length of the rolling window. Defaults to ``min_train``. Ignored when
        ``expanding`` is set.
    step:
        Observations between refits. ``1`` refits daily, which is the honest
        default; larger values are much faster and let the parameters go stale
        between refits.
    """
    if min_train >= n_observations:
        raise ValueError(
            f"min_train={min_train} leaves no room to forecast in "
            f"{n_observations} observations"
        )
    window = window or min_train

    for end in range(min_train, n_observations, step):
        start = 0 if expanding else max(0, end - window)
        yield Split(start=start, end=end)


@dataclass
class WalkForwardResult:
    """Out-of-sample forecasts and targets from one walk-forward run."""

    model_name: str
    forecasts: np.ndarray
    targets: np.ndarray
    indices: np.ndarray
    failures: list[tuple[int, str]] = field(default_factory=list)

    @property
    def n_forecasts(self) -> int:
        return int(self.forecasts.size)

    def __len__(self) -> int:
        return self.n_forecasts


def run_walk_forward(
    model_factory: Callable[[], Forecaster],
    returns: np.ndarray,
    realized_variance: np.ndarray,
    min_train: int = 250,
    expanding: bool = True,
    window: int | None = None,
    step: int = 1,
    horizon: int = 1,
) -> WalkForwardResult:
    """Refit and forecast across the sample, returning out-of-sample results.

    ``model_factory`` is a callable producing a *fresh* model, not a model
    instance — reusing one instance across splits would carry state from a
    later window into an earlier forecast, which is a leak that is very hard to
    see once the results look plausible.

    A model that raises during ``fit`` is recorded in ``failures`` and skipped
    rather than crashing the run, since optimiser failures on a handful of
    windows are normal and should not discard the other several hundred.
    """
    returns = np.asarray(returns, dtype=float)
    realized_variance = np.asarray(realized_variance, dtype=float)
    if returns.shape != realized_variance.shape:
        raise ValueError(
            f"returns {returns.shape} and realized_variance "
            f"{realized_variance.shape} must be aligned"
        )

    forecasts: list[float] = []
    targets: list[float] = []
    indices: list[int] = []
    failures: list[tuple[int, str]] = []
    name = model_factory().name

    for split in walk_forward_splits(
        returns.size - horizon + 1, min_train, expanding, window, step
    ):
        target_index = split.end + horizon - 1
        if target_index >= realized_variance.size:
            break

        try:
            model = model_factory()
            model.fit(
                returns[split.start : split.end],
                realized_variance[split.start : split.end],
            )
            prediction = model.forecast_variance(horizon)
        except Exception as exc:  # optimiser failures are expected occasionally
            failures.append((split.end, f"{type(exc).__name__}: {exc}"))
            continue

        if not np.isfinite(prediction) or prediction <= 0:
            failures.append((split.end, f"non-positive forecast {prediction!r}"))
            continue

        forecasts.append(float(prediction))
        targets.append(float(realized_variance[target_index]))
        indices.append(target_index)

    return WalkForwardResult(
        model_name=name,
        forecasts=np.asarray(forecasts),
        targets=np.asarray(targets),
        indices=np.asarray(indices, dtype=int),
        failures=failures,
    )


def align_results(results: Sequence[WalkForwardResult]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Restrict several runs to the observations they all produced a forecast for.

    Models fail on different windows, so their forecast series can differ in
    length. Comparing them without aligning would score each on a different
    sample — and a model that quietly failed on every crisis day would look
    excellent.
    """
    if not results:
        raise ValueError("no results to align")

    common = set(results[0].indices.tolist())
    for result in results[1:]:
        common &= set(result.indices.tolist())
    ordered = np.array(sorted(common), dtype=int)

    if ordered.size == 0:
        raise ValueError("models share no common forecast dates")

    aligned: dict[str, np.ndarray] = {}
    targets: np.ndarray | None = None
    for result in results:
        lookup = {int(i): position for position, i in enumerate(result.indices)}
        positions = [lookup[int(i)] for i in ordered]
        aligned[result.model_name] = result.forecasts[positions]
        if targets is None:
            targets = result.targets[positions]

    assert targets is not None
    return aligned, targets
