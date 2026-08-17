"""Walk-forward comparison of the baseline volatility forecasters on BTC.

The part-3 harness running end to end: fetch history, build a realised-variance
target, refit each model on strictly past data, then score with QLIKE and MSE
and test the differences for significance.

Everything reported here is out of sample. The models are the *baselines* — the
bar that GJR-GARCH, Markov switching, HAR-RV and the neural net in
:mod:`qar.models.research` have to clear.

Run:  .venv/bin/python examples/04_baseline_eval.py
      (add --refresh to bypass the cache)
"""

from __future__ import annotations

import sys

import numpy as np

from qar.data import DeribitClient, annualize, log_returns, realized_variance
from qar.evaluation import (
    align_results,
    diebold_mariano,
    loss_table,
    mincer_zarnowitz,
    qlike,
    run_walk_forward,
)
from qar.models import EWMAVariance, GARCH11, RandomWalkVariance

MODELS = (RandomWalkVariance, EWMAVariance, GARCH11)
ESTIMATOR = "garman_klass"
MIN_TRAIN = 250
STEP = 1


def main(refresh: bool = False) -> None:
    client = DeribitClient()
    print("Fetching BTC daily bars from Deribit...")
    try:
        bars = client.ohlc("BTC-PERPETUAL", resolution="1D", days=730, refresh=refresh)
    except Exception as exc:
        print(f"could not reach Deribit ({exc}); using cache if present.")
        return

    returns = log_returns(bars["close"])
    variance = realized_variance(bars, ESTIMATOR)

    print(f"{returns.size} daily observations")
    print(f"realised vol: median {np.median(annualize(variance)):.1%}, "
          f"range {np.min(annualize(variance)):.1%} to {np.max(annualize(variance)):.1%}")
    print(f"target: {ESTIMATOR}, refit every {STEP} day(s) after {MIN_TRAIN} of history\n")

    # -- walk forward ----------------------------------------------------
    results = []
    for factory in MODELS:
        result = run_walk_forward(
            factory, returns, variance, min_train=MIN_TRAIN, step=STEP
        )
        note = f", {len(result.failures)} failures" if result.failures else ""
        print(f"  {result.model_name:<14} {len(result)} out-of-sample forecasts{note}")
        results.append(result)

    forecasts, targets = align_results(results)
    print(f"\naligned on {targets.size} common dates\n")

    # -- statistical evaluation ------------------------------------------
    print("=" * 74)
    print("STATISTICAL ACCURACY (lower is better)")
    print("=" * 74)

    table = loss_table(forecasts, targets)
    ranked = sorted(table.items(), key=lambda item: item[1]["qlike"])

    print(f"{'model':<16}{'QLIKE':>12}{'MSE':>14}{'MZ slope':>12}{'MZ R2':>10}{'bias':>12}")
    print("-" * 74)
    for name, losses in ranked:
        mz = mincer_zarnowitz(forecasts[name], targets)
        verdict = "unbiased" if mz.unbiased else "biased"
        print(
            f"{name:<16}{losses['qlike']:>12.5f}{losses['mse']:>14.4e}"
            f"{mz.slope:>12.3f}{mz.r_squared:>10.3f}{verdict:>12}"
        )

    best = ranked[0][0]
    print(f"\nbest by QLIKE: {best}")

    # -- significance ----------------------------------------------------
    print("\n" + "=" * 74)
    print("DIEBOLD-MARIANO — is the difference real, or sampling noise?")
    print("=" * 74)
    print(f"{'comparison':<38}{'DM stat':>10}{'p-value':>10}  verdict")
    print("-" * 74)

    best_loss = qlike(forecasts[best], targets)
    for name in forecasts:
        if name == best:
            continue
        test = diebold_mariano(best_loss, qlike(forecasts[name], targets))
        label = f"{best} vs {name}"
        print(f"{label:<38}{test.statistic:>10.3f}{test.p_value:>10.4f}  {test.verdict()}")

    print(
        "\nThe Harvey-Leybourne-Newbold correction is on: without it this test\n"
        "over-rejects badly at a few hundred observations, which is exactly the\n"
        "sample size available here."
    )

    # -- where to go next ------------------------------------------------
    print("\n" + "=" * 74)
    print("NEXT")
    print("=" * 74)
    print(
        "These are the baselines. Implement the models in qar/models/research.py\n"
        "and they will appear in this table automatically — each already\n"
        "conforms to the Forecaster interface, so adding one to MODELS above is\n"
        "the only change needed.\n\n"
        "  GJRGarch                 asymmetric response to negative returns\n"
        "  MarkovSwitchingVariance  regime shifts GARCH adapts to too slowly\n"
        "  HARRV                    long memory from three OLS coefficients\n"
        "  NeuralVolatility         nonlinear interactions, if the data supports it\n\n"
        "Beat GARCH(1,1) on QLIKE with a Diebold-Mariano p below 0.05, then run\n"
        "examples/05 to check the edge survives transaction costs."
    )


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
