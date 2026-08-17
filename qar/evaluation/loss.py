r"""Loss functions for volatility forecasts.

The subtlety that governs this whole module: **the target is unobservable**.
Realised variance is a noisy proxy for the true conditional variance, and most
loss functions rank models differently depending on how noisy that proxy is.
Patton (2011) showed only a small family of losses is *robust* — meaning the
model that minimises expected loss against the noisy proxy is the same one that
would minimise it against the true variance.

Two members of that family are implemented:

*   **MSE**, :math:`(\hat\sigma^2 - RV)^2`. Robust, but dominated by the largest
    variance days, so a single crisis observation can decide the ranking.
*   **QLIKE**, :math:`RV/\hat\sigma^2 - \ln(RV/\hat\sigma^2) - 1`. Robust and
    scale-free, penalising proportional errors equally at every level. It is the
    default here for exactly that reason.

**Anything else is unsafe.** Mean absolute error, R², and mean absolute
percentage error are *not* robust to proxy noise: with a noisy proxy they can
rank a worse model above a better one, systematically rather than by chance.
They are deliberately not provided.
"""

from __future__ import annotations

import numpy as np

__all__ = ["mse", "qlike", "LOSSES", "loss_table"]

#: Variance forecasts are clipped to this floor before division. A forecast of
#: exactly zero would make QLIKE infinite and let one degenerate observation
#: decide the whole comparison.
_FLOOR = 1e-12


def mse(forecast: np.ndarray, realized: np.ndarray) -> np.ndarray:
    r"""Squared error on the variance scale, :math:`(\hat\sigma^2 - RV)^2`.

    Robust in Patton's sense, but heavily weighted toward high-variance days.
    Report it alongside QLIKE rather than instead of it — when the two disagree
    about the ranking, the disagreement is itself the finding, and usually says
    a model does well in calm markets and badly in crises or vice versa.
    """
    forecast = np.asarray(forecast, dtype=float)
    realized = np.asarray(realized, dtype=float)
    return (forecast - realized) ** 2


def qlike(forecast: np.ndarray, realized: np.ndarray) -> np.ndarray:
    r"""QLIKE loss, the default for volatility forecast evaluation.

    .. math::
        L(\hat\sigma^2, RV) = \frac{RV}{\hat\sigma^2}
            - \ln\frac{RV}{\hat\sigma^2} - 1

    Properties that make it the right default:

    *   **Zero if and only if the forecast is exact**, and positive otherwise.
    *   **Scale-free.** The loss depends only on the ratio, so a 10% error costs
        the same in a calm month as in a crisis. MSE does not have this property
        and is consequently decided by a handful of extreme days.
    *   **Asymmetric, in the useful direction.** Under-forecasting is penalised
        more heavily than over-forecasting, which matches the asymmetry of the
        underlying decision problem — being short gamma into a move that was not
        predicted costs more than carrying too much hedge.
    *   **Robust to proxy noise** (Patton 2011), so a noisy realised-variance
        estimator does not bias the ranking.
    """
    forecast = np.maximum(np.asarray(forecast, dtype=float), _FLOOR)
    realized = np.maximum(np.asarray(realized, dtype=float), _FLOOR)
    ratio = realized / forecast
    return ratio - np.log(ratio) - 1.0


#: Loss functions by name.
LOSSES = {"qlike": qlike, "mse": mse}


def loss_table(
    forecasts: dict[str, np.ndarray], realized: np.ndarray
) -> dict[str, dict[str, float]]:
    """Mean loss per model under every loss function.

    Returns ``{model_name: {loss_name: mean_loss}}``, ready to be sorted into a
    ranked comparison table.
    """
    realized = np.asarray(realized, dtype=float)
    table: dict[str, dict[str, float]] = {}
    for model_name, series in forecasts.items():
        series = np.asarray(series, dtype=float)
        if series.shape != realized.shape:
            raise ValueError(
                f"{model_name}: forecast shape {series.shape} does not match "
                f"realized shape {realized.shape}"
            )
        table[model_name] = {
            loss_name: float(np.mean(loss_fn(series, realized)))
            for loss_name, loss_fn in LOSSES.items()
        }
    return table
