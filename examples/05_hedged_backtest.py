"""Economic evaluation: trade the forecast against implied, delta-hedged.

The statistical table in example 04 says which model predicts variance best.
This asks the harder question — whether that edge is worth money once you have
to hedge it and pay to trade.

This is also the example where parts 1 and 2 are load-bearing: every rebalance
calls the analytical Delta, and every mark calls the BSM price.

Run:  .venv/bin/python examples/05_hedged_backtest.py
"""

from __future__ import annotations

import sys

import numpy as np

from qar.backtest import BacktestConfig, cost_sweep, run_backtest
from qar.data import (
    DeribitClient,
    annualize,
    log_returns,
    realized_variance,
    trailing_mean,
)
from qar.evaluation import run_walk_forward
from qar.models import EWMAVariance, GARCH11


def build_implied_proxy(variance: np.ndarray, window: int = 30) -> np.ndarray:
    r"""Stand-in for the implied-vol series, built from **strictly past** data.

    A proper study reads implied vol from the option chain at each date, which
    needs a historical chain snapshot Deribit's public API does not serve. As a
    substitute this uses trailing realised volatility scaled by 1.1, encoding
    the well-documented variance risk premium — implied normally sits above
    subsequent realised.

    The window ends at ``i-1``, not ``i``. That detail is the whole ballgame:
    an average that includes day ``i``'s own realised volatility lets the
    strategy see part of the very quantity it is betting on, and because
    realised volatility is strongly persistent the leak propagates into the
    days after it too. The first version of this function got that wrong and
    produced a Sharpe near 3.

    **This is still a proxy, and the numbers below inherit its assumptions.**
    It exercises the machinery and sizes the transaction-cost effect. It is not
    evidence that the strategy makes money. Replace it with a stored chain
    before drawing any conclusion.
    """
    volatility = np.asarray(annualize(variance), dtype=float)
    return trailing_mean(volatility, window) * 1.1


def main() -> None:
    client = DeribitClient()
    print("Fetching BTC daily bars...")
    try:
        bars = client.ohlc("BTC-PERPETUAL", resolution="1D", days=730)
    except Exception as exc:
        print(f"could not reach Deribit ({exc}).")
        return

    returns = log_returns(bars["close"])
    variance = realized_variance(bars, "garman_klass")
    spot = bars["close"][1:]

    print(f"{returns.size} daily observations, "
          f"median realised vol {np.median(annualize(variance)):.1%}\n")

    print("Generating out-of-sample forecasts (GARCH(1,1))...")
    result = run_walk_forward(GARCH11, returns, variance, min_train=250, step=1)
    print(f"  {len(result)} forecasts\n")

    indices = result.indices
    forecast_vol = annualize(result.forecasts)
    implied_vol = build_implied_proxy(variance)[indices]
    aligned_spot = spot[indices]

    print("=" * 72)
    print("SIGNAL")
    print("=" * 72)
    edge = forecast_vol - implied_vol
    print(f"forecast vol : median {np.median(forecast_vol):.1%}")
    print(f"implied proxy: median {np.median(implied_vol):.1%}")
    print(f"edge         : mean {np.mean(edge):+.2%}, "
          f"std {np.std(edge):.2%}, |edge|>5% on "
          f"{np.mean(np.abs(edge) > 0.05):.0%} of days")

    print("\n" + "=" * 72)
    print("TRANSACTION-COST SWEEP")
    print("=" * 72)
    print(f"{'cost':>10}{'total P&L':>14}{'Sharpe':>10}{'max DD':>12}"
          f"{'hit rate':>11}{'trades':>9}")
    print("-" * 72)

    results = cost_sweep(
        aligned_spot, forecast_vol, implied_vol,
        config=BacktestConfig(holding_days=5, entry_threshold=0.05),
    )
    for cost in sorted(results):
        r = results[cost]
        print(
            f"{cost:>10.4%}{r.total_return:>14.2f}{r.sharpe:>10.2f}"
            f"{r.max_drawdown:>12.2f}{r.hit_rate:>11.1%}{r.n_trades:>9}"
        )

    print(
        "\nRead the sweep, not the headline. A Sharpe that collapses between 0\n"
        "and 2bp of cost was never a strategy — it was a measurement of the\n"
        "bid-ask spread."
    )

    print("\n" + "=" * 72)
    print("CAVEATS")
    print("=" * 72)
    print(
        "  * implied vol is a trailing-realised proxy, not market data — see\n"
        "    build_implied_proxy above. Replace it before believing any number.\n"
        "  * hedging is daily; real discrete-hedging error is larger.\n"
        "  * one implied vol per option: no smile, so the straddle is priced\n"
        "    at a single vol rather than strike by strike.\n"
        "  * no funding on the hedge, no margin, no slippage beyond the flat cost.\n\n"
        "Every one of these makes live trading worse than this backtest, never\n"
        "better. Treat the output as an upper bound."
    )


if __name__ == "__main__":
    main()
