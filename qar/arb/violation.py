r"""Arbitrage violations, and the trades that monetise them.

A checker that only reports "put-call parity is violated by 0.18" is hard to
trust and impossible to act on. Every violation here carries the actual trade:
which legs, in what direction, at what price, with the cash flow today and the
cash flow at expiry.

That also makes the module self-verifying. Each leg knows its own payoff as a
function of the terminal spot, so :meth:`Trade.worst_case` can sweep a range of
terminal prices and confirm the position never loses. An arbitrage that fails
that sweep is not an arbitrage, and the test suite asserts it for every
violation type the package can raise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence

import numpy as np

__all__ = ["ViolationKind", "Leg", "Trade", "ArbitrageViolation"]


class ViolationKind(str, Enum):
    """Which no-arbitrage condition was breached."""

    PARITY = "put-call parity"
    LOWER_BOUND = "lower bound"
    UPPER_BOUND = "upper bound"
    STRIKE_MONOTONICITY = "strike monotonicity"
    BUTTERFLY = "butterfly convexity"
    CALENDAR = "calendar spread"


@dataclass(frozen=True)
class Leg:
    """One position in an arbitrage portfolio.

    Attributes
    ----------
    action:
        ``BUY``, ``SELL``, ``LEND`` or ``BORROW``. Presentation only; the
        arithmetic lives in ``cashflow_now`` and ``payoff``.
    quantity:
        Number of units, always positive — direction is carried by ``action``.
    instrument:
        Human-readable name, e.g. ``"call K=100"``.
    unit_price:
        Price paid or received per unit today.
    cashflow_now:
        Signed cash at inception. Negative means money leaves your pocket.
    payoff:
        Cash from this leg at expiry as a function of terminal spot.
    """

    action: str
    quantity: float
    instrument: str
    unit_price: float
    cashflow_now: float
    payoff: Callable[[float], float] = field(repr=False)

    def describe(self) -> str:
        if self.action in ("LEND", "BORROW"):
            return f"{self.action:<6} {self.unit_price:>10.4f}  {self.instrument}"
        return (
            f"{self.action:<6} {self.quantity:>10.4f}  {self.instrument} "
            f"@ {self.unit_price:.4f}"
        )


@dataclass(frozen=True)
class Trade:
    """A self-financing portfolio that captures a specific mispricing."""

    legs: tuple[Leg, ...]
    rationale: str

    @property
    def cashflow_now(self) -> float:
        """Net cash received today. Positive is the arbitrage profit."""
        return sum(leg.cashflow_now for leg in self.legs)

    def cashflow_at_expiry(self, terminal_spot: float) -> float:
        """Net cash at expiry for a given terminal underlying price."""
        return sum(leg.payoff(terminal_spot) for leg in self.legs)

    def worst_case(self, spots: Sequence[float] | None = None) -> float:
        """Minimum expiry cash flow over a range of terminal prices.

        The proof that the trade is riskless. For a genuine static arbitrage
        this is zero (parity, which locks exactly) or positive (the bound and
        convexity trades, which keep an option on top of the locked profit).
        A negative value means the trade is not an arbitrage and the checker
        that produced it is wrong.

        **Single-horizon only.** Every leg is evaluated at one terminal spot,
        which assumes all legs expire together. That covers parity, bounds,
        monotonicity and butterflies. It does *not* describe a calendar spread,
        whose legs expire at different dates — the justification there is the
        dominance argument in :func:`qar.arb.bounds.check_calendar`, not this
        sweep.
        """
        if spots is None:
            reference = max(
                (leg.unit_price for leg in self.legs if leg.unit_price > 0),
                default=100.0,
            )
            spots = np.linspace(1e-6, 5.0 * reference, 2001)
        return float(min(self.cashflow_at_expiry(float(s)) for s in spots))

    def describe(self) -> str:
        lines = [self.rationale, ""]
        lines.extend("  " + leg.describe() for leg in self.legs)
        lines.append("")
        lines.append(f"  cashflow now:     {self.cashflow_now:+.4f}")
        worst = self.worst_case()
        locked = "locked" if abs(worst) < 1e-9 else "plus upside"
        lines.append(f"  cashflow at T:    {worst:+.4f}  (worst case, {locked})")
        return "\n".join(lines)


@dataclass(frozen=True)
class ArbitrageViolation:
    """A breached no-arbitrage condition, with the trade that exploits it."""

    kind: ViolationKind
    magnitude: float
    detail: str
    trade: Trade

    @property
    def profit(self) -> float:
        """Riskless profit today, per unit of the trade."""
        return self.trade.cashflow_now

    def __str__(self) -> str:
        return (
            f"{self.kind.value}: {self.detail} "
            f"(magnitude {self.magnitude:.6g}, riskless profit {self.profit:.6g})"
        )

    def report(self) -> str:
        """Full multi-line description including the trade."""
        return f"{self}\n\n{self.trade.describe()}"


# --------------------------------------------------------------------------
# Leg constructors — used by parity.py and bounds.py
# --------------------------------------------------------------------------


def option_leg(
    buy: bool, quantity: float, kind: str, strike: float, premium: float
) -> Leg:
    """A long or short European option position expiring at the horizon."""
    sign = 1.0 if buy else -1.0

    def payoff(terminal_spot: float, k: float = strike, o: str = kind) -> float:
        intrinsic = (
            max(terminal_spot - k, 0.0) if o == "call" else max(k - terminal_spot, 0.0)
        )
        return sign * quantity * intrinsic

    return Leg(
        action="BUY" if buy else "SELL",
        quantity=quantity,
        instrument=f"{kind} K={strike:g}",
        unit_price=premium,
        cashflow_now=-sign * quantity * premium,
        payoff=payoff,
    )


def underlying_leg(
    buy: bool,
    quantity: float,
    spot: float,
    growth: float = 1.0,
    label: str = "underlying",
) -> Leg:
    r"""A long or short position in the underlying, held to the horizon.

    ``growth`` is the factor by which the *holding* grows over the life of the
    trade through continuous income reinvestment — :math:`e^{qT}` for a
    dividend yield :math:`q`, equivalently :math:`e^{(r-b)T}`. Buying
    :math:`e^{-qT}` shares today therefore delivers exactly one share at
    expiry.

    Modelling this explicitly is not a nicety. Omitting it leaves a conversion
    short by :math:`(1 - e^{-qT})` of a share, which is worth nothing at
    today's spot but scales linearly with the terminal price — so the position
    is quietly short the underlying and the "arbitrage" loses without bound.
    The default of 1.0 covers a non-income underlying, where spot and holding
    coincide.
    """
    sign = 1.0 if buy else -1.0
    suffix = "" if abs(growth - 1.0) < 1e-12 else f" (grows to {quantity * growth:.4f})"
    return Leg(
        action="BUY" if buy else "SELL",
        quantity=quantity,
        instrument=label + suffix,
        unit_price=spot,
        cashflow_now=-sign * quantity * spot,
        payoff=lambda terminal_spot: sign * quantity * growth * terminal_spot,
    )


def cash_leg(lend: bool, present_value: float, rate: float, maturity: float) -> Leg:
    """Lend or borrow ``present_value`` today, settling at the horizon.

    Lending costs cash now and returns ``present_value * exp(r*T)`` at expiry;
    borrowing is the mirror image. This is the leg that makes the discounting
    in the parity relation explicit rather than implied.
    """
    sign = 1.0 if lend else -1.0
    future_value = present_value * math.exp(rate * maturity)
    return Leg(
        action="LEND" if lend else "BORROW",
        quantity=1.0,
        instrument=f"cash at r={rate:.4g} to T={maturity:g} (repays {future_value:.4f})",
        unit_price=present_value,
        cashflow_now=-sign * present_value,
        payoff=lambda _terminal_spot: sign * future_value,
    )
