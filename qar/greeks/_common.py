"""Shared machinery for the Greek implementations.

Every Greek follows the same three-step shape:

1.  pull the cached intermediates (``d1``, ``d2``, ``n(d1)``, discount factors)
    off the :class:`~qar.core.bsm.BSMInputs`;
2.  evaluate the closed form with a guarded denominator, so the degenerate
    branch never raises;
3.  overwrite the degenerate entries with their analytic limit.

:class:`Intermediates` does step 1 once so the Greek bodies read like the
formulas they implement rather than like plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from qar.core._numeric import safe_where, scalarize
from qar.core.bsm import BSMInputs, OptionKind, d1_d2
from qar.core.distributions import norm_cdf, norm_pdf

__all__ = ["Intermediates", "prepare", "resolve_kind", "finish"]


def resolve_kind(kind: Any) -> OptionKind:
    """Coerce ``"call"`` / ``"put"`` / :class:`OptionKind` to an enum member."""
    if isinstance(kind, OptionKind):
        return kind
    try:
        return OptionKind(str(kind).lower())
    except ValueError:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}") from None


@dataclass(frozen=True)
class Intermediates:
    """Everything the closed forms need, computed once."""

    inputs: BSMInputs
    d1: Any
    d2: Any
    n1: Any            # n(d1), the standard normal density at d1
    vol_sqrt_T: Any    # sigma * sqrt(T)
    sqrt_T: Any
    df_r: Any          # e^{-rT}
    df_carry: Any      # e^{(b-r)T}
    degenerate: Any    # where sigma*sqrt(T) is numerically zero
    safe_vol_sqrt_T: Any   # vol_sqrt_T with zeros replaced by 1
    safe_sqrt_T: Any       # sqrt(T) with zeros replaced by 1
    safe_T: Any            # T with zeros replaced by 1
    safe_sigma: Any        # sigma with zeros replaced by 1

    def N(self, x: Any) -> Any:
        """Standard normal CDF, spelled the way the formulas are written."""
        return norm_cdf(x)

    @property
    def S(self) -> Any:
        return self.inputs.S

    @property
    def K(self) -> Any:
        return self.inputs.K

    @property
    def T(self) -> Any:
        return self.inputs.T

    @property
    def r(self) -> Any:
        return self.inputs.r

    @property
    def b(self) -> Any:
        return self.inputs.b

    @property
    def sigma(self) -> Any:
        return self.inputs.sigma


def prepare(inputs: BSMInputs) -> Intermediates:
    """Compute the shared intermediates for one set of inputs."""
    d1, d2 = d1_d2(inputs)
    vol_sqrt_T = inputs.vol_sqrt_T
    degenerate = inputs.is_degenerate

    # Replace zeros in denominators with 1 so the regular branch is always
    # finite; the degenerate entries are overwritten afterwards anyway.
    if _complex(inputs):
        safe_vol_sqrt_T = vol_sqrt_T
        safe_sqrt_T = inputs.sqrt_T
        safe_T = inputs.T
        safe_sigma = inputs.sigma
    else:
        safe_vol_sqrt_T = safe_where(degenerate, 1.0, vol_sqrt_T)
        zero_T = np.asarray(inputs.T) <= 0
        safe_sqrt_T = safe_where(zero_T, 1.0, inputs.sqrt_T)
        safe_T = safe_where(zero_T, 1.0, inputs.T)
        safe_sigma = safe_where(np.asarray(inputs.sigma) <= 0, 1.0, inputs.sigma)

    return Intermediates(
        inputs=inputs,
        d1=d1,
        d2=d2,
        n1=norm_pdf(d1),
        vol_sqrt_T=vol_sqrt_T,
        sqrt_T=inputs.sqrt_T,
        df_r=inputs.df_r,
        df_carry=inputs.df_carry,
        degenerate=degenerate,
        safe_vol_sqrt_T=safe_vol_sqrt_T,
        safe_sqrt_T=safe_sqrt_T,
        safe_T=safe_T,
        safe_sigma=safe_sigma,
    )


def _complex(inputs: BSMInputs) -> bool:
    return any(
        np.iscomplexobj(v)
        for v in (inputs.S, inputs.K, inputs.T, inputs.r, inputs.sigma, inputs.b)
    )


def finish(state: Intermediates, regular: Any, limit: Any = 0.0) -> Any:
    """Select the degenerate limit where applicable and return a clean scalar.

    ``limit`` defaults to 0 because that is the correct :math:`\\sigma\\sqrt{T}
    \\to 0` limit for every Greek that involves ``n(d1)``: the density vanishes
    faster than the ``1/\\sigma\\sqrt{T}`` factors diverge. Gamma and Dual Gamma
    pass an explicit limit, since theirs is a Dirac spike at the forward.
    """
    if _complex(state.inputs):
        # No degenerate handling in the complex plane; see qar.core.bsm.
        return scalarize(regular)
    return scalarize(safe_where(state.degenerate, limit, regular))
