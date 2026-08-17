r"""Second-order Greeks: the seven second partial derivatives.

=========== ===================================================== ==============
Greek       Derivative                                            Call/put
=========== ===================================================== ==============
Gamma       :math:`\partial^2 V/\partial S^2`                     same
Vanna       :math:`\partial^2 V/\partial S\,\partial\sigma`        same
Volga       :math:`\partial^2 V/\partial\sigma^2`                  same
Charm       :math:`\partial^2 V/\partial S\,\partial t`            differs
Veta        :math:`\partial^2 V/\partial\sigma\,\partial t`        same
Vera        :math:`\partial^2 V/\partial\sigma\,\partial r`        same
DualGamma   :math:`\partial^2 V/\partial K^2`                      same
=========== ===================================================== ==============

Only Charm differs between calls and puts, and only through the carry term —
which is exactly what put-call parity predicts. Parity says

.. math:: c - p = S e^{(b-r)T} - K e^{-rT}

The right-hand side is independent of :math:`\sigma`, so every derivative taken
purely with respect to :math:`\sigma` is identical for the two. It is linear in
:math:`S`, so the *second* derivative in :math:`S` matches too. Only the mixed
:math:`S`-:math:`t` derivative picks up a residue, because the right-hand side
does depend on :math:`t`. That structure is asserted directly in the test suite.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qar.core._numeric import safe_where
from qar.core.bsm import CALL, BSMInputs
from qar.core.distributions import norm_pdf
from qar.greeks._common import finish, prepare, resolve_kind

__all__ = ["gamma", "vanna", "volga", "vomma", "charm", "veta", "vera", "dual_gamma"]


def _spike_limit(state: Any, at_the_forward_value: Any) -> Any:
    r"""Degenerate limit for Gamma-like Greeks.

    As :math:`\sigma\sqrt{T}\to 0` the payoff kink becomes a corner, and the
    second derivative becomes a Dirac spike: zero everywhere except at the
    forward, where it is unbounded. Returning ``inf`` there is the honest
    answer, and the alternative — silently returning 0 — would hide a real
    modelling discontinuity from anyone hedging near expiry.
    """
    at_forward = np.isclose(np.asarray(state.inputs.forward), np.asarray(state.K))
    return safe_where(at_forward, np.inf, 0.0) * np.ones_like(np.asarray(at_the_forward_value))


def gamma(inputs: BSMInputs, kind: Any = CALL) -> Any:
    r"""Gamma, :math:`\partial^2 V/\partial S^2`.

    Differentiating :math:`\Delta = \phi e^{(b-r)T} N(\phi d_1)` once more in
    :math:`S`, and using :math:`\partial d_1/\partial S = 1/(S\sigma\sqrt{T})`:

    .. math::
        \Gamma = \frac{e^{(b-r)T} n(d_1)}{S\sigma\sqrt{T}}

    The :math:`\phi^2 = 1` from the chain rule is what makes this identical for
    calls and puts. Always positive for a long option: the value function is
    convex in spot, which is the whole reason a delta-hedged long option makes
    money in a move of either sign.
    """
    resolve_kind(kind)
    state = prepare(inputs)
    regular = state.df_carry * state.n1 / (state.S * state.safe_vol_sqrt_T)
    return finish(state, regular, limit=_spike_limit(state, regular))


def vanna(inputs: BSMInputs, kind: Any = CALL) -> Any:
    r"""Vanna, :math:`\partial^2 V/\partial S\,\partial\sigma`.

    Equivalently :math:`\partial\Delta/\partial\sigma` or
    :math:`\partial\mathcal{V}/\partial S` — the mixed partial is symmetric,
    and the test suite checks both readings agree numerically.

    .. math::
        \text{Vanna} = -e^{(b-r)T} n(d_1)\frac{d_2}{\sigma}

    Sign flips with moneyness: it vanishes when :math:`d_2 = 0`, i.e. at the
    strike where the risk-neutral in-the-money probability is exactly one half.
    This is the Greek that makes a delta hedge leak when the smile moves.
    """
    resolve_kind(kind)
    state = prepare(inputs)
    regular = -state.df_carry * state.n1 * state.d2 / state.safe_sigma
    return finish(state, regular)


def volga(inputs: BSMInputs, kind: Any = CALL) -> Any:
    r"""Volga (Vomma), :math:`\partial^2 V/\partial\sigma^2`.

    Convexity in volatility — the vega of vega:

    .. math::
        \text{Volga} = \mathcal{V}\,\frac{d_1 d_2}{\sigma}

    Positive for wings, negative in a band around the forward where
    :math:`d_1 d_2 < 0`. That sign pattern is why long strangles are long
    volatility-of-volatility while straddles are close to flat.
    """
    resolve_kind(kind)
    state = prepare(inputs)
    vega_value = state.S * state.df_carry * state.n1 * state.sqrt_T
    regular = vega_value * state.d1 * state.d2 / state.safe_sigma
    return finish(state, regular)


#: ``vomma`` is the more common name on trading desks; ``volga`` in the literature.
vomma = volga


def charm(inputs: BSMInputs, kind: Any = CALL) -> Any:
    r"""Charm (delta decay), :math:`\partial^2 V/\partial S\,\partial t`.

    How Delta drifts purely from the passage of time, holding spot fixed:

    .. math::
        \text{Charm} = -e^{(b-r)T}\left[
            n(d_1)\left(\frac{b}{\sigma\sqrt{T}} - \frac{d_2}{2T}\right)
            + \phi (b-r) N(\phi d_1)\right]

    Measured per unit of *calendar* time, matching the Theta convention, so
    the finite-difference check negates the maturity derivative.

    This is the Greek that bites over a weekend: an untouched hedge on a
    near-expiry position drifts out of balance with no market move at all.
    """
    phi = resolve_kind(kind).sign
    state = prepare(inputs)
    density_term = state.n1 * (
        state.b / state.safe_vol_sqrt_T - state.d2 / (2.0 * state.safe_T)
    )
    carry_term = phi * (state.b - state.r) * state.N(phi * state.d1)
    regular = -state.df_carry * (density_term + carry_term)
    # At expiry only the carry term survives; the density term vanishes.
    return finish(state, regular, limit=-state.df_carry * carry_term)


def veta(inputs: BSMInputs, kind: Any = CALL) -> Any:
    r"""Veta, :math:`\partial^2 V/\partial\sigma\,\partial t`.

    The decay of Vega:

    .. math::
        \text{Veta} = \mathcal{V}\left[(r - b)
            + \frac{b\, d_1}{\sigma\sqrt{T}}
            - \frac{1 + d_1 d_2}{2T}\right]

    Normally negative — Vega shrinks as expiry approaches, roughly as
    :math:`\sqrt{T}`, which is why long-dated options carry the vega risk and
    short-dated ones carry the gamma.
    """
    resolve_kind(kind)
    state = prepare(inputs)
    vega_value = state.S * state.df_carry * state.n1 * state.sqrt_T
    bracket = (
        (state.r - state.b)
        + state.b * state.d1 / state.safe_vol_sqrt_T
        - (1.0 + state.d1 * state.d2) / (2.0 * state.safe_T)
    )
    return finish(state, vega_value * bracket)


def vera(inputs: BSMInputs, kind: Any = CALL) -> Any:
    r"""Vera (rhova), :math:`\partial^2 V/\partial\sigma\,\partial r`.

    How Vega responds to rates. Like :func:`~qar.greeks.first.rho` this splits
    on whether the carry rate tracks :math:`r`.

    *Carry moves with the rate.* Then :math:`e^{(b-r)T}` is rate-free and
    :math:`r` reaches Vega only through :math:`d_1`, with
    :math:`\partial d_1/\partial r = \sqrt{T}/\sigma`. Since
    :math:`n'(d_1) = -d_1 n(d_1)`:

    .. math::
        \text{Vera} = -S e^{(b-r)T} n(d_1)\,\frac{T d_1}{\sigma}

    *Carry fixed* (futures). Then :math:`r` appears only in the overall
    discount, so Vega scales as :math:`e^{-rT}` and
    :math:`\text{Vera} = -T\mathcal{V}`.
    """
    resolve_kind(kind)
    state = prepare(inputs)
    vega_value = state.S * state.df_carry * state.n1 * state.sqrt_T

    if not inputs.carry_depends_on_r:
        return finish(state, -inputs.T * vega_value)

    regular = -state.S * state.df_carry * state.n1 * inputs.T * state.d1 / state.safe_sigma
    return finish(state, regular)


def dual_gamma(inputs: BSMInputs, kind: Any = CALL) -> Any:
    r"""Dual Gamma, :math:`\partial^2 V/\partial K^2`.

    .. math::
        \text{DualGamma} = \frac{e^{-rT} n(d_2)}{K\sigma\sqrt{T}}

    Worth more than its obscurity suggests: by Breeden-Litzenberger this *is*
    the risk-neutral density of :math:`S_T` at :math:`K`, discounted. A negative
    value on a fitted surface means the implied density goes negative, i.e. a
    butterfly arbitrage — which is precisely the check
    :func:`qar.arb.bounds.check_butterfly` performs on market quotes.
    """
    resolve_kind(kind)
    state = prepare(inputs)
    regular = state.df_r * norm_pdf(state.d2) / (state.K * state.safe_vol_sqrt_T)
    return finish(state, regular, limit=_spike_limit(state, regular))
