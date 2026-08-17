r"""Delta-hedged straddle backtest — the economic half of the evidence.

A better QLIKE is a statistical claim. This turns it into a financial one: if a
volatility forecast genuinely beats the market's, then systematically selling
options when the forecast is below implied — and buying when it is above, while
hedging the directional exposure away — should make money.

This is also where parts 1 and 2 stop being decorative. Every rebalance calls
:func:`qar.greeks.first.delta`; the entry and exit marks call
:func:`qar.core.bsm.price`; the P&L attribution below is the Greek expansion the
analytical derivations produced.

The mechanics
-------------
Each day, compare the forecast volatility to the implied. If they disagree by
more than a threshold, open a straddle in the direction of the disagreement and
delta-hedge it until expiry. The P&L of a continuously delta-hedged long option
is, to second order,

.. math::
    \mathrm{d}\Pi \approx \tfrac{1}{2}\Gamma S^2
        \left(\sigma_{\text{realised}}^2 - \sigma_{\text{implied}}^2\right)\mathrm{d}t

which is the point: the position is a bet on realised variance against implied,
with the direction exposure hedged out. Gamma weights that bet, so the same
volatility view earns most where gamma is largest.

What this deliberately does not model
-------------------------------------
Discrete hedging error, the bid-ask spread on the option legs beyond the flat
cost parameter, funding on the hedge, and the volatility smile (a single implied
vol is used per option). Each of those makes real trading worse than this
backtest, never better — so treat the output as an upper bound and read the
transaction-cost sweep, not the headline Sharpe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qar.core.bsm import CALL, PUT, BSMInputs, price
from qar.greeks.first import delta

__all__ = ["BacktestConfig", "BacktestResult", "run_backtest", "cost_sweep"]


@dataclass(frozen=True)
class BacktestConfig:
    """Parameters of the straddle backtest."""

    holding_days: int = 5
    """Days a straddle is held before expiry."""

    entry_threshold: float = 0.05
    """Minimum |forecast - implied| in vol points required to open a position."""

    transaction_cost: float = 0.0005
    """Proportional cost, charged on option premium and on each hedge trade."""

    rebalance_per_day: int = 1
    """Delta hedges per day. More hedging reduces discretisation error and
    increases cost — the trade-off this parameter exists to explore."""

    rate: float = 0.0
    """Risk-free rate. Zero is a defensible default for a crypto-margined book."""

    carry: float = 0.0
    """Cost of carry. Zero matches Black-76 on a perpetual/future."""

    trading_days: float = 365.0


@dataclass
class BacktestResult:
    """Daily P&L and the summary statistics computed from it."""

    pnl: np.ndarray
    equity: np.ndarray
    positions: np.ndarray
    config: BacktestConfig
    diagnostics: dict = field(default_factory=dict)

    @property
    def total_return(self) -> float:
        return float(self.equity[-1]) if self.equity.size else 0.0

    @property
    def sharpe(self) -> float:
        """Annualised Sharpe of the daily P&L series.

        Computed on P&L rather than on returns, since the strategy has no
        natural capital base — the denominator would be an arbitrary choice
        that changes the number without changing the strategy.
        """
        if self.pnl.size < 2:
            return 0.0
        deviation = float(np.std(self.pnl, ddof=1))
        if deviation == 0:
            return 0.0
        return float(np.mean(self.pnl) / deviation * np.sqrt(self.config.trading_days))

    @property
    def max_drawdown(self) -> float:
        if self.equity.size == 0:
            return 0.0
        peak = np.maximum.accumulate(self.equity)
        return float(np.min(self.equity - peak))

    @property
    def hit_rate(self) -> float:
        """Fraction of *settled trades* that made money.

        Scored on settlements, not on calendar days. A trade's P&L is booked
        on the single day it settles, so averaging over every day it was open
        would count the four flat days in a five-day hold as losses and drag
        the number toward zero regardless of how the trade went.
        """
        settled = self.pnl[self.pnl != 0.0]
        return float(np.mean(settled > 0)) if settled.size else 0.0

    @property
    def n_trades(self) -> int:
        """Number of positions opened.

        Counts settlements rather than transitions in ``positions``: a
        transition count double-counts every trade (once entering, once
        leaving) except where two trades happen to abut.
        """
        return int(np.count_nonzero(self.pnl))

    def summary(self) -> str:
        return (
            f"total P&L {self.total_return:+.4f} | Sharpe {self.sharpe:.2f} | "
            f"max DD {self.max_drawdown:.4f} | hit rate {self.hit_rate:.1%} | "
            f"{self.n_trades} trades"
        )


def run_backtest(
    spot: np.ndarray,
    forecast_vol: np.ndarray,
    implied_vol: np.ndarray,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Trade the forecast-versus-implied disagreement, delta-hedged.

    Parameters
    ----------
    spot:
        Underlying price series, one observation per day.
    forecast_vol, implied_vol:
        Annualised volatilities aligned with ``spot``. ``forecast_vol`` comes
        from a model in :mod:`qar.models`; ``implied_vol`` from the option
        market.

    Notes
    -----
    Options are priced at the *implied* volatility — that is what they cost —
    while the underlying evolves as the market actually did. The P&L therefore
    reflects the gap between implied and realised, which is exactly the bet.
    Sizing is one straddle per signal, so the equity curve is directly readable
    as P&L per unit of exposure rather than being shaped by a position-sizing
    rule that would need its own justification.
    """
    config = config or BacktestConfig()
    spot = np.asarray(spot, dtype=float)
    forecast_vol = np.asarray(forecast_vol, dtype=float)
    implied_vol = np.asarray(implied_vol, dtype=float)

    if not (spot.shape == forecast_vol.shape == implied_vol.shape):
        raise ValueError("spot, forecast_vol and implied_vol must be aligned")

    n = spot.size
    horizon = config.holding_days
    pnl = np.zeros(n)
    positions = np.zeros(n)

    day = 0
    while day + horizon < n:
        edge = forecast_vol[day] - implied_vol[day]
        if abs(edge) < config.entry_threshold or not np.isfinite(edge):
            day += 1
            continue

        # Forecast above implied -> options look cheap -> buy volatility.
        direction = 1.0 if edge > 0 else -1.0
        strike = spot[day]           # at-the-money straddle
        entry_iv = implied_vol[day]
        maturity = horizon / config.trading_days

        def option_inputs(index: int, elapsed: float) -> BSMInputs:
            return BSMInputs(
                S=spot[index],
                K=strike,
                T=max(maturity - elapsed, 1e-8),
                r=config.rate,
                sigma=entry_iv,
                b=config.carry,
                carry_depends_on_r=False,
            )

        entry = option_inputs(day, 0.0)
        entry_premium = float(price(entry, CALL)) + float(price(entry, PUT))
        trade_pnl = -direction * entry_premium
        trade_pnl -= config.transaction_cost * entry_premium

        # Delta-hedge to expiry.
        hedge_position = direction * (
            float(delta(entry, CALL)) + float(delta(entry, PUT))
        )
        trade_pnl -= hedge_position * spot[day]
        trade_pnl -= config.transaction_cost * abs(hedge_position) * spot[day]

        for offset in range(1, horizon):
            index = day + offset
            elapsed = offset / config.trading_days
            current = option_inputs(index, elapsed)
            target = direction * (
                float(delta(current, CALL)) + float(delta(current, PUT))
            )
            traded = target - hedge_position
            trade_pnl -= traded * spot[index]
            trade_pnl -= config.transaction_cost * abs(traded) * spot[index]
            hedge_position = target

        # Settle: options at intrinsic, hedge at the terminal spot.
        terminal = spot[day + horizon]
        payoff = max(terminal - strike, 0.0) + max(strike - terminal, 0.0)
        trade_pnl += direction * payoff
        trade_pnl += hedge_position * terminal
        trade_pnl -= config.transaction_cost * abs(hedge_position) * terminal

        pnl[day + horizon] = trade_pnl
        positions[day : day + horizon] = direction
        day += horizon

    return BacktestResult(
        pnl=pnl,
        equity=np.cumsum(pnl),
        positions=positions,
        config=config,
        diagnostics={
            "n_days": n,
            "days_in_market": int(np.sum(positions != 0)),
            "mean_abs_edge": float(np.nanmean(np.abs(forecast_vol - implied_vol))),
        },
    )


def cost_sweep(
    spot: np.ndarray,
    forecast_vol: np.ndarray,
    implied_vol: np.ndarray,
    costs: tuple[float, ...] = (0.0, 0.0002, 0.0005, 0.001, 0.002),
    config: BacktestConfig | None = None,
) -> dict[float, BacktestResult]:
    """Re-run the backtest across transaction-cost assumptions.

    The most informative single output of the backtest. A strategy whose Sharpe
    survives realistic costs is interesting; one that collapses between zero and
    two basis points was never a strategy, it was a measurement of the spread.
    """
    base = config or BacktestConfig()
    results = {}
    for cost in costs:
        variant = BacktestConfig(
            holding_days=base.holding_days,
            entry_threshold=base.entry_threshold,
            transaction_cost=cost,
            rebalance_per_day=base.rebalance_per_day,
            rate=base.rate,
            carry=base.carry,
            trading_days=base.trading_days,
        )
        results[cost] = run_backtest(spot, forecast_vol, implied_vol, variant)
    return results
