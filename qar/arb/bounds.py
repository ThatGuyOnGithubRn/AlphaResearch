r"""Static no-arbitrage constraints on a set of option quotes.

Put-call parity relates a call to *its own* put. These conditions constrain
prices across strikes and across maturities, and together they are what a
volatility surface must satisfy before it is worth calibrating anything to.

All four are model-free — each follows from a portfolio whose payoff is
non-negative in every state of the world, so a negative cost would be free
money.

============================ ====================================================
Condition                    Statement (calls, discounted forward :math:`F_d`)
============================ ====================================================
Bounds                       :math:`(F_d - K_d)^+ \le c \le F_d`
Strike monotonicity          :math:`K_1 < K_2 \Rightarrow c(K_1) \ge c(K_2)`
Butterfly convexity          :math:`c(K_1) - 2c(K_2) + c(K_3) \ge 0`
Calendar                     :math:`T_1 < T_2 \Rightarrow c(T_1) \le c(T_2)`
============================ ====================================================

Butterfly convexity is the sharpest of the four. By Breeden-Litzenberger the
second derivative of the call price in strike *is* the discounted risk-neutral
density, so a butterfly priced below zero says the market-implied density is
negative somewhere — which is not a mispricing to be smoothed away but a
statement that the quotes cannot all be right. It is also the condition that
:func:`qar.greeks.second.dual_gamma` computes in closed form.

The calendar condition as stated assumes a non-negative carry over the interval;
under a strongly negative carry (a deeply backwardated future, or a high
dividend yield) a longer-dated European call can legitimately be worth less, so
:func:`check_calendar` takes the carry into account rather than applying the
naive rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from qar.arb.violation import (
    ArbitrageViolation,
    Trade,
    ViolationKind,
    cash_leg,
    option_leg,
    underlying_leg,
)

__all__ = [
    "Quote",
    "check_bounds",
    "check_strike_monotonicity",
    "check_butterfly",
    "check_calendar",
    "scan",
]


@dataclass(frozen=True)
class Quote:
    """One observed option price, with the market data needed to test it."""

    price: float
    S: float
    K: float
    T: float
    r: float
    kind: str = "call"
    b: float | None = None

    @property
    def carry(self) -> float:
        return self.r if self.b is None else self.b

    @property
    def discounted_forward(self) -> float:
        r""":math:`S e^{(b-r)T}` — the spot leg's present value."""
        return self.S * math.exp((self.carry - self.r) * self.T)

    @property
    def discounted_strike(self) -> float:
        r""":math:`K e^{-rT}`."""
        return self.K * math.exp(-self.r * self.T)


def check_bounds(quote: Quote, tol: float = 1e-8) -> list[ArbitrageViolation]:
    r"""Test that a quote lies inside its model-free price bounds.

    For a call, :math:`\max(F_d - K_d, 0) \le c \le F_d`:

    *   **Lower.** A call is worth at least the forward less the discounted
        strike, and never less than zero. Below that, buy the call, short the
        underlying and lend the proceeds: the payoff at expiry is
        :math:`\max(S_T - K, 0) - S_T + K = \max(K - S_T, 0) \ge 0`, for a
        negative cost today.
    *   **Upper.** A call can never be worth more than the underlying it
        delivers, since its payoff is capped by :math:`S_T`.

    Puts are the mirror image: :math:`\max(K_d - F_d, 0) \le p \le K_d`.
    """
    violations: list[ArbitrageViolation] = []
    fwd, strike_pv = quote.discounted_forward, quote.discounted_strike
    units = math.exp((quote.carry - quote.r) * quote.T)
    # See underlying_leg: income reinvestment carries the holding to one unit.
    growth = math.exp((quote.r - quote.carry) * quote.T)

    if quote.kind == "call":
        lower, upper = max(fwd - strike_pv, 0.0), fwd
        lower_legs = (
            option_leg(True, 1.0, "call", quote.K, quote.price),
            underlying_leg(False, units, quote.S, growth),
            cash_leg(True, strike_pv, quote.r, quote.T),
        )
        lower_rationale = (
            "Call trades below intrinsic-forward. Buy the call, short the "
            "underlying, lend the discounted strike; the residual payoff is a "
            "free put."
        )
        upper_legs = (
            option_leg(False, 1.0, "call", quote.K, quote.price),
            underlying_leg(True, units, quote.S, growth),
        )
        upper_rationale = (
            "Call trades above the underlying. Sell the call and buy the "
            "underlying; the call can never claim more than what you hold."
        )
    else:
        lower, upper = max(strike_pv - fwd, 0.0), strike_pv
        lower_legs = (
            option_leg(True, 1.0, "put", quote.K, quote.price),
            underlying_leg(True, units, quote.S, growth),
            cash_leg(False, strike_pv, quote.r, quote.T),
        )
        lower_rationale = (
            "Put trades below intrinsic-forward. Buy the put, buy the "
            "underlying, borrow the discounted strike; the residual payoff is a "
            "free call."
        )
        upper_legs = (
            option_leg(False, 1.0, "put", quote.K, quote.price),
            cash_leg(True, strike_pv, quote.r, quote.T),
        )
        upper_rationale = (
            "Put trades above the discounted strike. Sell the put and lend the "
            "discounted strike; the put can never claim more than K."
        )

    if quote.price < lower - tol:
        violations.append(
            ArbitrageViolation(
                kind=ViolationKind.LOWER_BOUND,
                magnitude=lower - quote.price,
                detail=f"{quote.kind} at {quote.price:.6g} < lower bound {lower:.6g}",
                trade=Trade(lower_legs, lower_rationale),
            )
        )

    if quote.price > upper + tol:
        violations.append(
            ArbitrageViolation(
                kind=ViolationKind.UPPER_BOUND,
                magnitude=quote.price - upper,
                detail=f"{quote.kind} at {quote.price:.6g} > upper bound {upper:.6g}",
                trade=Trade(upper_legs, upper_rationale),
            )
        )

    return violations


def check_strike_monotonicity(
    low: Quote, high: Quote, tol: float = 1e-8
) -> ArbitrageViolation | None:
    r"""Calls must not increase in strike (puts must not decrease).

    The vertical spread — long the low strike, short the high — has payoff
    :math:`\min(\max(S_T - K_1, 0), K_2 - K_1) \ge 0`. It can never be worth
    less than nothing, so it cannot be entered for a credit.
    """
    if low.K >= high.K:
        raise ValueError("`low` must have the smaller strike")

    if low.kind == "call":
        breach = high.price - low.price
        legs = (
            option_leg(True, 1.0, "call", low.K, low.price),
            option_leg(False, 1.0, "call", high.K, high.price),
        )
        rationale = (
            f"Call at K={high.K:g} costs more than at K={low.K:g}. Buy the low "
            "strike, sell the high strike: a credit today for a payoff that is "
            "never negative."
        )
    else:
        breach = low.price - high.price
        legs = (
            option_leg(True, 1.0, "put", high.K, high.price),
            option_leg(False, 1.0, "put", low.K, low.price),
        )
        rationale = (
            f"Put at K={low.K:g} costs more than at K={high.K:g}. Buy the high "
            "strike, sell the low strike."
        )

    if breach <= tol:
        return None

    return ArbitrageViolation(
        kind=ViolationKind.STRIKE_MONOTONICITY,
        magnitude=breach,
        detail=(
            f"{low.kind} K={low.K:g} at {low.price:.6g} vs "
            f"K={high.K:g} at {high.price:.6g}"
        ),
        trade=Trade(legs, rationale),
    )


def check_butterfly(
    low: Quote, mid: Quote, high: Quote, tol: float = 1e-8
) -> ArbitrageViolation | None:
    r"""Option prices must be convex in strike.

    For equally spaced strikes the butterfly :math:`c(K_1) - 2c(K_2) + c(K_3)`
    has a non-negative payoff — a tent peaking at :math:`K_2` — so it cannot
    cost less than zero. With unequal spacing the weights adjust to
    :math:`\lambda = (K_3 - K_2)/(K_3 - K_1)` on the low strike and
    :math:`1 - \lambda` on the high.

    A breach means the implied risk-neutral density is negative between the
    strikes, by Breeden-Litzenberger. That is the strongest statement in this
    module: the quotes are not merely aggressive, they are jointly impossible.
    """
    if not (low.K < mid.K < high.K):
        raise ValueError("strikes must be strictly increasing")

    weight_low = (high.K - mid.K) / (high.K - low.K)
    weight_high = 1.0 - weight_low
    cost = weight_low * low.price + weight_high * high.price - mid.price

    if cost >= -tol:
        return None

    legs = (
        option_leg(True, weight_low, low.kind, low.K, low.price),
        option_leg(False, 1.0, mid.kind, mid.K, mid.price),
        option_leg(True, weight_high, high.kind, high.K, high.price),
    )
    rationale = (
        f"Butterfly {low.K:g}/{mid.K:g}/{high.K:g} can be bought for a credit. "
        "Its payoff is a non-negative tent, so the implied density is negative "
        "between these strikes."
    )

    return ArbitrageViolation(
        kind=ViolationKind.BUTTERFLY,
        magnitude=-cost,
        detail=(
            f"{weight_low:.4g}*{low.price:.6g} + {weight_high:.4g}*{high.price:.6g} "
            f"- {mid.price:.6g} = {cost:.6g} < 0"
        ),
        trade=Trade(legs, rationale),
    )


def check_calendar(
    near: Quote, far: Quote, tol: float = 1e-8
) -> ArbitrageViolation | None:
    r"""A near-dated call must not cost more than a carry-scaled far-dated one.

    The familiar rule — a longer-dated option is always worth more — is only
    true for a non-income underlying, and applying it blindly to a chain with a
    dividend yield manufactures arbitrages that are not there. Here is the
    trade that actually justifies a calendar comparison, with
    :math:`\tau = T_2 - T_1`.

    Sell one near call, buy :math:`\lambda = e^{(r-b)\tau}` far calls. At
    :math:`T_1` the far calls are worth at least their own intrinsic-forward
    bound,

    .. math::
        \lambda\left[S e^{(b-r)\tau} - K e^{-r\tau}\right]^+
        = \left[S - K e^{-b\tau}\right]^+

    — the :math:`\lambda` cancels the carry discount exactly — while the short
    near call owes :math:`[S - K]^+`. Since

    .. math:: \left[S - K e^{-b\tau}\right]^+ \ge [S - K]^+
        \quad\text{whenever } b \ge 0,

    the position never loses. So for :math:`b \ge 0` the no-arbitrage condition
    is

    .. math:: c(T_1) \le e^{(r-b)\tau} c(T_2)

    which collapses to the textbook :math:`c(T_1) \le c(T_2)` at :math:`b = r`,
    and correctly loosens for futures (:math:`b = 0`).

    **When the carry is negative** the inequality above reverses and no
    quantity of far calls dominates the near one at every spot — a long-dated
    call on a heavily backwardated underlying is genuinely allowed to be
    cheaper. There is no valid same-strike test in that regime, so this returns
    ``None`` rather than inventing one. The correct check there compares equal
    *forward moneyness* across expiries, which a same-strike chain does not
    provide.
    """
    if near.T >= far.T:
        raise ValueError("`near` must have the shorter maturity")
    if abs(near.K - far.K) > 1e-12:
        raise ValueError("calendar comparison requires the same strike")
    if near.kind != "call" or far.kind != "call":
        # The mirrored argument for puts needs b <= r rather than b >= 0; not
        # implemented, and silently reusing the call rule would be wrong.
        return None

    tau = far.T - near.T
    carry = far.carry
    if carry < 0:
        return None

    scale = math.exp((far.r - carry) * tau)
    breach = near.price - scale * far.price

    if breach <= tol:
        return None

    legs = (
        option_leg(False, 1.0, "call", near.K, near.price),
        option_leg(True, scale, "call", far.K, far.price),
    )
    rationale = (
        f"The T={near.T:g} call costs more than {scale:.4f} of the T={far.T:g} "
        "call, which dominates it at every spot. Sell the near, buy the "
        "carry-scaled far."
    )

    return ArbitrageViolation(
        kind=ViolationKind.CALENDAR,
        magnitude=breach,
        detail=(
            f"near T={near.T:g} at {near.price:.6g} > {scale:.6g} x far "
            f"T={far.T:g} at {far.price:.6g} = {scale * far.price:.6g}"
        ),
        trade=Trade(legs, rationale),
    )


def scan(quotes: Sequence[Quote], tol: float = 1e-8) -> list[ArbitrageViolation]:
    """Run every applicable check over a set of quotes.

    Bounds are tested on each quote individually; monotonicity and convexity on
    each same-expiry, same-kind strike slice; calendars on each same-strike,
    same-kind maturity chain.
    """
    violations: list[ArbitrageViolation] = []

    for quote in quotes:
        violations.extend(check_bounds(quote, tol))

    # Strike slices: fixed expiry and kind, sorted by strike.
    slices: dict[tuple[float, str], list[Quote]] = {}
    for quote in quotes:
        slices.setdefault((quote.T, quote.kind), []).append(quote)

    for slice_quotes in slices.values():
        ordered = sorted(slice_quotes, key=lambda q: q.K)
        for first, second in zip(ordered, ordered[1:]):
            found = check_strike_monotonicity(first, second, tol)
            if found is not None:
                violations.append(found)
        for a, b, c in zip(ordered, ordered[1:], ordered[2:]):
            found = check_butterfly(a, b, c, tol)
            if found is not None:
                violations.append(found)

    # Maturity chains: fixed strike and kind, sorted by expiry.
    chains: dict[tuple[float, str], list[Quote]] = {}
    for quote in quotes:
        chains.setdefault((quote.K, quote.kind), []).append(quote)

    for chain_quotes in chains.values():
        ordered = sorted(chain_quotes, key=lambda q: q.T)
        for near, far in zip(ordered, ordered[1:]):
            found = check_calendar(near, far, tol)
            if found is not None:
                violations.append(found)

    return violations
