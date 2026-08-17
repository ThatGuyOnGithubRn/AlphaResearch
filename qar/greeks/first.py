r"""First-order Greeks: the six first partial derivatives of the BSM price.

======== ============================ ===============================
Greek    Derivative                   With respect to
======== ============================ ===============================
Delta    :math:`\partial V/\partial S`   Spot
Vega     :math:`\partial V/\partial\sigma` Volatility
Theta    :math:`\partial V/\partial t`   Calendar time (not maturity)
Rho      :math:`\partial V/\partial r`   Risk-free rate
Epsilon  :math:`\partial V/\partial q`   Dividend yield
DualDelta :math:`\partial V/\partial K`  Strike
======== ============================ ===============================

Two sign conventions are fixed here and used consistently throughout:

*   **Theta is per unit of calendar time**, so :math:`\Theta = -\partial
    V/\partial T`. It is negative for most long options — they decay. The
    finite-difference validator negates accordingly.
*   **Vega is per unit of volatility**, not per volatility point. Multiply by
    0.01 for the "per 1% move" figure desks quote.

The recurring identity behind the unified call/put forms is

.. math:: N(-x) = 1 - N(x)

which lets :math:`\phi = \pm 1` absorb the entire call/put distinction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qar.core.bsm import CALL, BSMInputs, price
from qar.greeks._common import finish, prepare, resolve_kind

__all__ = ["delta", "vega", "theta", "rho", "epsilon", "dual_delta"]


def delta(inputs: BSMInputs, kind: Any = CALL) -> Any:
    r"""Delta, :math:`\partial V/\partial S`.

    Differentiating :math:`V = \phi[S e^{(b-r)T} N(\phi d_1) - K e^{-rT}
    N(\phi d_2)]` with respect to :math:`S`, the two density terms cancel
    exactly — this is the classic BSM simplification:

    .. math::
        S e^{(b-r)T} n(d_1) \frac{\partial d_1}{\partial S}
        = K e^{-rT} n(d_2) \frac{\partial d_2}{\partial S}

    because :math:`S e^{bT} n(d_1) = K n(d_2)`. What survives is

    .. math:: \Delta = \phi\, e^{(b-r)T} N(\phi d_1)

    Range: :math:`(0, e^{(b-r)T})` for a call, :math:`(-e^{(b-r)T}, 0)` for a
    put. Under plain BSM that is the familiar :math:`(0, 1)` and :math:`(-1, 0)`.
    """
    phi = resolve_kind(kind).sign
    state = prepare(inputs)
    regular = phi * state.df_carry * state.N(phi * state.d1)
    # Delta survives the degenerate limit unchanged: d1 -> +-inf sends N(d1) to
    # 1 or 0, giving the forward-intrinsic delta with no division involved.
    return finish(state, regular, limit=regular)


def vega(inputs: BSMInputs, kind: Any = CALL) -> Any:
    r"""Vega, :math:`\partial V/\partial\sigma`.

    .. math:: \mathcal{V} = S e^{(b-r)T} n(d_1)\sqrt{T}

    Identical for calls and puts, which follows immediately from put-call
    parity: the parity relation contains no :math:`\sigma`, so differentiating
    it gives :math:`\mathcal{V}_c - \mathcal{V}_p = 0`. The ``kind`` argument is
    accepted only so every Greek shares one signature.

    Always non-negative — more volatility cannot make an option worth less.
    """
    resolve_kind(kind)
    state = prepare(inputs)
    regular = state.S * state.df_carry * state.n1 * state.sqrt_T
    return finish(state, regular)


def theta(inputs: BSMInputs, kind: Any = CALL) -> Any:
    r"""Theta, :math:`\partial V/\partial t = -\partial V/\partial T`.

    Three effects, in the order they appear below:

    1.  :math:`-\dfrac{S e^{(b-r)T} n(d_1)\sigma}{2\sqrt{T}}` — loss of optionality
        as the remaining variance shrinks. Always negative, and the dominant
        term near expiry.
    2.  :math:`-\phi (b-r) S e^{(b-r)T} N(\phi d_1)` — carry on the spot leg.
    3.  :math:`-\phi\, r K e^{-rT} N(\phi d_2)` — unwinding of the discount on
        the strike leg. This is the term that can flip a deep-ITM European put's
        theta positive.

    .. math::
        \Theta = -\frac{S e^{(b-r)T} n(d_1)\sigma}{2\sqrt{T}}
                 - \phi (b-r) S e^{(b-r)T} N(\phi d_1)
                 - \phi\, r K e^{-rT} N(\phi d_2)
    """
    phi = resolve_kind(kind).sign
    state = prepare(inputs)
    decay = -(state.S * state.df_carry * state.n1 * state.sigma) / (2.0 * state.safe_sqrt_T)
    carry = -phi * (state.b - state.r) * state.S * state.df_carry * state.N(phi * state.d1)
    discount = -phi * state.r * state.K * state.df_r * state.N(phi * state.d2)
    regular = decay + carry + discount
    # At T = 0 the decay term vanishes (n(d1) -> 0) and the remaining terms are
    # already finite, so the limit is carry + discount evaluated there.
    return finish(state, regular, limit=carry + discount)


def rho(inputs: BSMInputs, kind: Any = CALL) -> Any:
    r"""Rho, :math:`\partial V/\partial r`.

    **This Greek is ambiguous unless you say what happens to the carry rate**,
    and getting it wrong is silent — the number looks plausible either way. Two
    regimes, selected by ``inputs.carry_depends_on_r``:

    *Carry moves with the rate* (``True``; BSM where :math:`b = r`, or Merton
    where :math:`b = r - q` with :math:`q` held fixed). Then :math:`e^{(b-r)T}`
    is free of :math:`r` entirely, and only the strike discount responds:

    .. math:: \rho = \phi\, T K e^{-rT} N(\phi d_2)

    *Carry is fixed* (``False``; Black-76 on futures, where :math:`b = 0`
    regardless of rates). Now :math:`r` enters only through the overall
    discount factor, so the price scales as :math:`e^{-rT}` and

    .. math:: \rho = -T V

    The two disagree in both magnitude and sign for a put, which is why the
    flag is explicit rather than inferred.
    """
    option = resolve_kind(kind)
    phi = option.sign
    state = prepare(inputs)

    if not inputs.carry_depends_on_r:
        fixed_carry = -inputs.T * price(inputs, option)
        return finish(state, fixed_carry, limit=fixed_carry)

    regular = phi * inputs.T * state.K * state.df_r * state.N(phi * state.d2)
    return finish(state, regular, limit=regular)


def epsilon(inputs: BSMInputs, kind: Any = CALL) -> Any:
    r"""Epsilon (also psi), :math:`\partial V/\partial q`.

    Sensitivity to the continuous dividend yield, where :math:`b = r - q`, so
    :math:`\partial/\partial q = -\partial/\partial b`. As with Delta the
    density terms cancel, leaving

    .. math:: \epsilon = -\phi\, S T e^{(b-r)T} N(\phi d_1)

    Negative for calls: dividends leak value out of the underlying that the
    call holder never receives. Positive for puts, for the same reason.
    """
    phi = resolve_kind(kind).sign
    state = prepare(inputs)
    regular = -phi * state.S * inputs.T * state.df_carry * state.N(phi * state.d1)
    return finish(state, regular, limit=regular)


def dual_delta(inputs: BSMInputs, kind: Any = CALL) -> Any:
    r"""Dual Delta, :math:`\partial V/\partial K`.

    The strike analogue of Delta, and — up to the discount factor — the
    risk-neutral probability of finishing in the money:

    .. math:: \frac{\partial V}{\partial K} = -\phi\, e^{-rT} N(\phi d_2)

    For a call this is :math:`-e^{-rT}\,\mathbb{Q}(S_T > K)`. It is the
    quantity a strike-monotonicity arbitrage check is really testing, which is
    why :mod:`qar.arb.bounds` leans on it.
    """
    phi = resolve_kind(kind).sign
    state = prepare(inputs)
    regular = -phi * state.df_r * state.N(phi * state.d2)
    return finish(state, regular, limit=regular)
