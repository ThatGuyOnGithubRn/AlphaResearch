"""The forecaster interface every volatility model implements.

One interface, so the walk-forward evaluator can run a random walk and an LSTM
through identical code and the comparison stays honest. The three rules the
interface exists to enforce:

1.  ``fit`` sees only past data. The evaluator slices the window; the model must
    not reach outside it.
2.  ``forecast`` returns a *variance*, not a volatility, and always at the daily
    horizon. Mixing the two — or mixing daily and annualised — is the most
    common way a comparison table ends up meaningless.
3.  ``fit`` returns ``self``, so a walk-forward loop can refit in one line.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

__all__ = ["Forecaster", "FitResult"]


class FitResult(dict):
    """Diagnostics from a fit — parameters, log-likelihood, convergence flags.

    A plain dict so models can report whatever is meaningful for them, but with
    a name that documents intent at the call site.
    """


class Forecaster(ABC):
    """Base class for one-step-ahead daily variance forecasters.

    Subclasses implement :meth:`fit` and :meth:`forecast_variance`. Everything
    else — annualised volatility, multi-step aggregation — is derived here so
    that every model agrees on the conversions.
    """

    name: str = "unnamed"

    def __init__(self) -> None:
        self.fitted_: bool = False
        self.diagnostics_: FitResult = FitResult()

    @abstractmethod
    def fit(self, returns: np.ndarray, realized_variance: np.ndarray | None = None) -> "Forecaster":
        """Estimate parameters from a window of past data.

        Parameters
        ----------
        returns:
            Daily log returns, oldest first.
        realized_variance:
            Optional realised-variance series aligned with ``returns``. Models
            that target the proxy directly (HAR-RV, the neural nets) need it;
            GARCH-family models work from returns alone and ignore it.
        """

    @abstractmethod
    def forecast_variance(self, horizon: int = 1) -> float:
        """One-step (or ``horizon``-step) ahead daily variance forecast."""

    def forecast_volatility(self, horizon: int = 1, periods: float = 365.0) -> float:
        """Annualised volatility implied by the variance forecast."""
        return float(np.sqrt(self.forecast_variance(horizon) * periods))

    def _require_fit(self) -> None:
        if not self.fitted_:
            raise RuntimeError(f"{self.name} must be fitted before forecasting")

    def __repr__(self) -> str:
        state = "fitted" if self.fitted_ else "unfitted"
        return f"<{type(self).__name__} {self.name!r} ({state})>"
