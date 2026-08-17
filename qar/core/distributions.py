"""Standard normal density, CDF and inverse CDF, implemented from scratch.

The whole suite rests on these three functions, so they are built here rather
than imported from ``scipy.stats``. Two consequences worth stating:

1.  ``norm_cdf`` is written in terms of the error function, which is exact to
    machine precision and, critically, is defined for *complex* arguments. That
    is what makes the complex-step differentiation in ``qar.validate`` possible
    (see :mod:`qar.validate.complex_step`).
2.  ``norm_ppf`` is a rational approximation refined by a Halley step, which
    takes it from ~1e-9 to full double precision. It is needed for implied
    volatility seeding and for simulating from the model.

Every function accepts scalars or array-likes. Scalars stay in pure Python
(``math.erf``); arrays route through ``scipy.special.erf``, which is a
vectorisation detail, not a borrowing of anybody's pricing code.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import special

__all__ = [
    "norm_pdf",
    "norm_cdf",
    "norm_ppf",
    "SQRT_2",
    "SQRT_2PI",
    "INV_SQRT_2PI",
]

SQRT_2 = math.sqrt(2.0)
SQRT_2PI = math.sqrt(2.0 * math.pi)
INV_SQRT_2PI = 1.0 / SQRT_2PI


def _is_scalar(x: Any) -> bool:
    """True when ``x`` is a single number we can keep in pure Python."""
    return isinstance(x, (int, float, np.floating, np.integer))


def norm_pdf(x: Any) -> Any:
    r"""Standard normal density.

    .. math::
        n(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2 / 2}

    Complex inputs are supported so that this can be differentiated by the
    complex-step method.
    """
    if _is_scalar(x):
        return INV_SQRT_2PI * math.exp(-0.5 * x * x)
    if isinstance(x, complex):
        # cmath rather than math: the complex-step validator lands here.
        import cmath

        return INV_SQRT_2PI * cmath.exp(-0.5 * x * x)
    arr = np.asarray(x)
    return INV_SQRT_2PI * np.exp(-0.5 * arr * arr)


def norm_cdf(x: Any) -> Any:
    r"""Standard normal cumulative distribution function.

    Derived from the error function by the substitution :math:`t = u/\sqrt{2}`:

    .. math::
        N(x) = \int_{-\infty}^{x} n(u)\,du
             = \tfrac{1}{2}\left[1 + \operatorname{erf}\!\left(
                   \frac{x}{\sqrt{2}}\right)\right]

    Using ``erf`` rather than a polynomial approximation (Hart, Abramowitz &
    Stegun 26.2.17, etc.) matters: those approximations top out around 1e-7 to
    1e-15 absolute error, which would swamp the finite-difference tolerances
    the validation harness is trying to assert. ``erf`` is correctly rounded.
    """
    if _is_scalar(x):
        return 0.5 * (1.0 + math.erf(x / SQRT_2))
    if isinstance(x, complex):
        # scipy.special.erf accepts complex arguments; math.erf does not.
        return 0.5 * (1.0 + complex(special.erf(x / SQRT_2)))
    arr = np.asarray(x)
    return 0.5 * (1.0 + special.erf(arr / SQRT_2))


# Acklam's rational approximation for the inverse normal CDF.
# Relative error < 1.15e-9 over the whole open interval (0, 1).
_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)

# Below/above these the central rational branch loses accuracy.
_P_LOW = 0.02425
_P_HIGH = 1.0 - _P_LOW


def _ppf_acklam(p: float) -> float:
    """Acklam's approximation, accurate to ~1.15e-9 relative."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf

    if p < _P_LOW:
        # Lower tail: expand in sqrt(-2 log p).
        q = math.sqrt(-2.0 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )

    if p > _P_HIGH:
        # Upper tail by symmetry.
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            ((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]
        ) / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)

    # Central branch.
    q = p - 0.5
    r = q * q
    return (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / (
        ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0
    )


def _ppf_refine(x: float, p: float) -> float:
    """One Halley step on ``N(x) - p = 0``, taking ~1e-9 to ~1e-16.

    Newton would use ``x - f/f'``. Halley adds the second-derivative term and
    converges cubically, so a single pass is enough:

    .. math::
        x \\leftarrow x - \\frac{f}{f'\\left(1 - \\tfrac{f f''}{2 f'^2}\\right)}

    With :math:`f = N(x) - p`, :math:`f' = n(x)` and :math:`f'' = -x\\,n(x)`
    this collapses to the expression below.
    """
    if not math.isfinite(x):
        return x
    err = norm_cdf(x) - p
    dens = norm_pdf(x)
    if dens < 1e-300:  # Deep in a tail; the approximation is all we have.
        return x
    u = err / dens
    return x - u / (1.0 + 0.5 * x * u)


def norm_ppf(p: Any) -> Any:
    r"""Inverse standard normal CDF, i.e. :math:`N^{-1}(p)`.

    Acklam's rational approximation followed by one Halley refinement, which
    brings the result to full double precision. Returns ``-inf``/``+inf`` at
    ``p = 0``/``p = 1`` and ``nan`` outside :math:`[0, 1]`.
    """
    if _is_scalar(p):
        pf = float(p)
        if pf < 0.0 or pf > 1.0:
            return math.nan
        return _ppf_refine(_ppf_acklam(pf), pf)

    arr = np.asarray(p, dtype=float)
    out = np.empty_like(arr)
    flat_in = arr.ravel()
    flat_out = out.ravel()
    for i, value in enumerate(flat_in):
        if value < 0.0 or value > 1.0:
            flat_out[i] = math.nan
        else:
            flat_out[i] = _ppf_refine(_ppf_acklam(float(value)), float(value))
    return out.reshape(arr.shape)
