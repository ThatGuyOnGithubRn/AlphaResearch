r"""Third-order Greeks.

========= ============================================== ==================
Greek     Derivative                                     Reads as
========= ============================================== ==================
Speed     :math:`\partial^3 V/\partial S^3`              Gamma's slope in spot
Zomma     :math:`\partial^3 V/\partial S^2\partial\sigma` Gamma's vega
Color     :math:`\partial^3 V/\partial S^2\partial t`     Gamma's decay
Ultima    :math:`\partial^3 V/\partial\sigma^3`           Volga's vega
========= ============================================== ==================

All four are call/put invariant. Put-call parity is linear in :math:`S` and
free of :math:`\sigma`, so any derivative of total order three that involves
:math:`S` at least twice, or :math:`\sigma` at all, annihilates it.

These are the formulas the validator earns its keep on. A third central
difference costs three subtractions of nearly equal numbers, so the achievable
accuracy is roughly :math:`\varepsilon^{1/2}\approx 10^{-8}` relative, not the
:math:`10^{-11}` the first-order checks reach. The tolerances in
:mod:`qar.validate.harness` are set from that error analysis rather than tuned
until the suite passed.
"""

from __future__ import annotations

from typing import Any

from qar.core.bsm import CALL, BSMInputs
from qar.greeks._common import finish, prepare, resolve_kind
from qar.greeks.second import gamma

__all__ = ["speed", "zomma", "color", "ultima"]


def speed(inputs: BSMInputs, kind: Any = CALL) -> Any:
    r"""Speed, :math:`\partial^3 V/\partial S^3 = \partial\Gamma/\partial S`.

    .. math::
        \text{Speed} = -\frac{\Gamma}{S}
            \left(\frac{d_1}{\sigma\sqrt{T}} + 1\right)

    Gamma is not flat across spot, so a gamma-neutral book re-breaks as soon as
    the underlying moves. Speed is the first-order size of that effect, and it
    is largest for short-dated near-the-money options — the same place Gamma
    itself peaks.
    """
    resolve_kind(kind)
    state = prepare(inputs)
    gamma_value = gamma(inputs, kind)
    regular = -(gamma_value / state.S) * (state.d1 / state.safe_vol_sqrt_T + 1.0)
    return finish(state, regular)


def zomma(inputs: BSMInputs, kind: Any = CALL) -> Any:
    r"""Zomma, :math:`\partial^3 V/\partial S^2\partial\sigma = \partial\Gamma/\partial\sigma`.

    .. math::
        \text{Zomma} = \Gamma\,\frac{d_1 d_2 - 1}{\sigma}

    Negative in the band around the forward where :math:`d_1 d_2 < 1`: a
    volatility spike *flattens* the gamma profile there, spreading it across
    strikes. Sign flips in the wings.
    """
    resolve_kind(kind)
    state = prepare(inputs)
    gamma_value = gamma(inputs, kind)
    regular = gamma_value * (state.d1 * state.d2 - 1.0) / state.safe_sigma
    return finish(state, regular)


def color(inputs: BSMInputs, kind: Any = CALL) -> Any:
    r"""Color, :math:`\partial^3 V/\partial S^2\partial t = \partial\Gamma/\partial t`.

    .. math::
        \text{Color} = \Gamma\left[(r - b)
            + \frac{b\,d_1}{\sigma\sqrt{T}}
            + \frac{1 - d_1 d_2}{2T}\right]

    Note the structural echo of :func:`~qar.greeks.second.veta`: same first two
    terms, and a final term that differs only by the sign on :math:`d_1 d_2`
    and on the whole bracket. That is not a coincidence — both descend from
    :math:`\partial n(d_1)/\partial t` — and it is a useful cross-check when
    transcribing them.

    Positive near the money: gamma piles up into expiry, which is what makes
    short-dated short-gamma books dangerous to leave unhedged.
    """
    resolve_kind(kind)
    state = prepare(inputs)
    gamma_value = gamma(inputs, kind)
    bracket = (
        (state.r - state.b)
        + state.b * state.d1 / state.safe_vol_sqrt_T
        + (1.0 - state.d1 * state.d2) / (2.0 * state.safe_T)
    )
    return finish(state, gamma_value * bracket)


def ultima(inputs: BSMInputs, kind: Any = CALL) -> Any:
    r"""Ultima, :math:`\partial^3 V/\partial\sigma^3 = \partial\text{Volga}/\partial\sigma`.

    .. math::
        \text{Ultima} = -\frac{\mathcal{V}}{\sigma^2}
            \left[d_1 d_2 (1 - d_1 d_2) + d_1^2 + d_2^2\right]

    The third-order term in a volatility Taylor expansion. It matters when
    marking a book through a large vol move, where Vega and Volga alone
    misestimate the P&L.
    """
    resolve_kind(kind)
    state = prepare(inputs)
    vega_value = state.S * state.df_carry * state.n1 * state.sqrt_T
    d1d2 = state.d1 * state.d2
    bracket = d1d2 * (1.0 - d1d2) + state.d1**2 + state.d2**2
    regular = -vega_value / (state.safe_sigma**2) * bracket
    return finish(state, regular)
