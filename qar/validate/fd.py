r"""Finite-difference differentiation of the BSM price.

Used to confirm every analytical Greek against an independent numerical
derivative. Two things determine whether such a check is meaningful at all:

**Step size.** A central difference of order :math:`k` has truncation error
:math:`O(h^2)` and round-off error :math:`O(\varepsilon/h^k)`. Balancing them,

.. math:: h^\star \sim \varepsilon^{1/(k+2)},
    \qquad \text{error}^\star \sim \varepsilon^{2/(k+2)}

so the best achievable relative accuracy falls off fast with order:

===== ============== ==================
Order :math:`h^\star` Best accuracy
===== ============== ==================
1     6e-6           4e-11
2     1e-4           1e-8
3     5e-4           1e-6
===== ============== ==================

The tolerances in :mod:`qar.validate.harness` come from this table, not from
whatever made the suite go green.

**Richardson extrapolation.** Evaluating at :math:`h` and :math:`h/2` and
combining as :math:`(4D(h/2) - D(h))/3` cancels the leading :math:`h^2` term,
leaving :math:`O(h^4)`. That buys back roughly two digits at every order and is
what lifts the third-order checks into a range where they say something.

**The two parameter subtleties**, both of which silently produce plausible-
looking wrong answers if ignored:

*   Bumping ``q`` means bumping ``b`` the *other* way, since :math:`b = r - q`.
*   Bumping ``r`` must bump ``b`` alongside it when the carry tracks the rate
    (:math:`b = r - q` with :math:`q` fixed). Under Black-76 it must not. This
    is the same fork documented in :func:`qar.greeks.first.rho`, and getting it
    wrong here would "confirm" the wrong Rho.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from qar.core.bsm import BSMInputs


@dataclass(frozen=True)
class FDResult:
    r"""A numerical derivative together with an estimate of its own error.

    The error estimate is what makes an automated comparison honest. A fixed
    tolerance per derivative order assumes the truncation error depends only on
    the step size, but it also carries the magnitude of the next derivative in
    the Taylor series — and for a one-week deep-out-of-the-money option those
    higher derivatives are enormous. A tolerance tuned on comfortable
    parameters then fails in the corners, and one loosened to survive the
    corners stops testing anything in the middle.

    Richardson supplies the estimate for free. With
    :math:`D(h) = D + ch^2 + O(h^4)` and the extrapolant
    :math:`\hat{D} = (4D(h/2) - D(h))/3`,

    .. math::
        \hat{D} - D(h/2) = \frac{D(h/2) - D(h)}{3}

    so the spread between the two step sizes bounds the error in the
    extrapolant. The validator compares against that, and reports it.
    """

    value: float
    error_estimate: float

    def __float__(self) -> float:
        return self.value

__all__ = [
    "FDResult",
    "bump",
    "step_for",
    "nested_central",
    "richardson",
    "differentiate",
]

#: Machine epsilon for float64.
EPS = np.finfo(float).eps

#: Optimal relative step per total derivative order, from eps**(1/(k+2)).
_STEP_BY_ORDER = {
    1: EPS ** (1.0 / 3.0),
    2: EPS ** (1.0 / 4.0),
    3: EPS ** (1.0 / 5.0),
}


def bump(inputs: BSMInputs, var: str, amount: float) -> BSMInputs:
    """Return ``inputs`` with ``var`` shifted by ``amount``.

    Handles the two couplings that a naive ``replace(var=value + h)`` gets
    wrong; see the module docstring.
    """
    if var == "q":
        # b = r - q, so raising the dividend yield lowers the carry rate.
        return inputs.replace(b=inputs.b - amount)

    if var == "r":
        if inputs.carry_depends_on_r:
            # b = r - q with q held fixed: the carry rate moves with r.
            return inputs.replace(r=inputs.r + amount, b=inputs.b + amount)
        # Black-76: b is pinned (typically 0) and does not follow r.
        return inputs.replace(r=inputs.r + amount)

    if var not in ("S", "K", "T", "sigma", "b"):
        raise ValueError(f"cannot differentiate with respect to {var!r}")

    return inputs.replace(**{var: getattr(inputs, var) + amount})


def step_for(inputs: BSMInputs, var: str, order: int) -> float:
    """Absolute step size for ``var``, scaled to the magnitude of the value.

    A relative step is essential: ``sigma`` is O(0.2) while ``S`` may be
    O(50000) for a BTC option, and one absolute step cannot serve both.
    """
    base = _STEP_BY_ORDER.get(order, _STEP_BY_ORDER[3])
    current = {
        "q": inputs.r - inputs.b,
        "r": inputs.r,
    }.get(var, getattr(inputs, var, 1.0))
    scale = max(abs(float(np.asarray(current))), 1e-2)
    return base * scale


def nested_central(
    func: Callable[[BSMInputs], Any],
    inputs: BSMInputs,
    wrt: Sequence[str],
    steps: Sequence[float],
) -> float:
    """Apply a central difference once per entry in ``wrt``, recursively.

    Nesting is what makes this general: ``("S", "S", "sigma")`` becomes a
    central difference in ``sigma`` of a central difference in ``S`` of a
    central difference in ``S``, with no special-casing per Greek. Mixed and
    repeated partials fall out of the same code path.
    """
    if not wrt:
        return float(np.real(np.asarray(func(inputs))))

    var, rest = wrt[0], wrt[1:]
    h, rest_steps = steps[0], steps[1:]

    up = nested_central(func, bump(inputs, var, +h), rest, rest_steps)
    down = nested_central(func, bump(inputs, var, -h), rest, rest_steps)
    return (up - down) / (2.0 * h)


def richardson(
    func: Callable[[BSMInputs], Any],
    inputs: BSMInputs,
    wrt: Sequence[str],
    steps: Sequence[float],
) -> FDResult:
    r"""Richardson-extrapolated nested central difference, with error estimate.

    Central differences carry error :math:`D(h) = D + c h^2 + O(h^4)`. Halving
    the step and forming

    .. math:: \frac{4 D(h/2) - D(h)}{3}

    eliminates :math:`c h^2` exactly, leaving :math:`O(h^4)`. The disagreement
    between the two step sizes, divided by three, estimates what remains — see
    :class:`FDResult`.
    """
    coarse = nested_central(func, inputs, wrt, steps)
    fine = nested_central(func, inputs, wrt, [h / 2.0 for h in steps])
    extrapolated = (4.0 * fine - coarse) / 3.0
    return FDResult(extrapolated, abs(fine - coarse) / 3.0)


def differentiate(
    func: Callable[[BSMInputs], Any],
    inputs: BSMInputs,
    wrt: Sequence[str],
    order: int | None = None,
    extrapolate: bool = True,
) -> FDResult:
    """Numerically differentiate ``func`` at ``inputs`` with respect to ``wrt``.

    Parameters
    ----------
    func:
        Maps :class:`~qar.core.bsm.BSMInputs` to a price.
    wrt:
        Variables to differentiate against, with repetition, e.g.
        ``("S", "S", "sigma")`` for Zomma.
    order:
        Total derivative order; defaults to ``len(wrt)``. Sets the step size.
    extrapolate:
        Apply Richardson extrapolation. Leave on unless you are studying the
        raw truncation behaviour; without it no error estimate is available and
        the returned ``error_estimate`` is ``nan``.

    Returns
    -------
    FDResult
        The derivative and an estimate of its own numerical error.
    """
    total_order = order if order is not None else len(wrt)
    steps = [step_for(inputs, var, total_order) for var in wrt]
    if not extrapolate:
        return FDResult(nested_central(func, inputs, wrt, steps), float("nan"))
    return richardson(func, inputs, wrt, steps)
