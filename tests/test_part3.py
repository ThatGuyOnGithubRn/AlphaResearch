"""Part-3 scaffold: realised variance, baselines, evaluation, backtest.

The two tests that matter most here are less obvious than they look:

*   :class:`TestGarchRecoversTruth` simulates from a *known* GARCH process and
    checks the estimator finds the parameters back. Without it, a fitted model
    has nothing to be checked against — real data has no answer key.
*   :class:`TestNoLookAhead` instruments a model to record exactly which
    observations it was shown, and asserts it never saw the one it was asked to
    predict. Look-ahead is the failure mode that makes a forecasting result
    worthless while leaving every number looking plausible.
"""

import numpy as np
import pytest

from qar.backtest import BacktestConfig, cost_sweep, run_backtest
from qar.data.realized import (
    annualize,
    close_to_close,
    garman_klass,
    log_returns,
    parkinson,
    realized_variance,
    rogers_satchell,
    trailing_mean,
)
from qar.evaluation import (
    align_results,
    diebold_mariano,
    loss_table,
    mincer_zarnowitz,
    mse,
    newey_west_variance,
    qlike,
    run_walk_forward,
    walk_forward_splits,
)
from qar.models import EWMAVariance, GARCH11, RandomWalkVariance
from qar.models.base import Forecaster
from qar.models.research import GJRGarch, HARRV, MarkovSwitchingVariance, NeuralVolatility

RNG = np.random.default_rng(20260817)


def simulate_garch(n, omega=2e-6, alpha=0.08, beta=0.90, seed=7):
    """Simulate a GARCH(1,1) path with known parameters."""
    rng = np.random.default_rng(seed)
    variance = omega / (1 - alpha - beta)
    returns = np.empty(n)
    variances = np.empty(n)
    for t in range(n):
        variances[t] = variance
        shock = rng.standard_normal() * np.sqrt(variance)
        returns[t] = shock
        variance = omega + alpha * shock**2 + beta * variance
    return returns, variances


def synthetic_bars(n=600, daily_vol=0.02, drift=0.0, steps_per_day=400, seed=3):
    r"""OHLC bars aggregated from a simulated intraday path.

    The range estimators are derived for the running maximum and minimum of a
    *continuous* diffusion. Fabricating ``high = max(open, close)`` instead — the
    obvious shortcut — silently breaks that assumption: Rogers-Satchell becomes
    identically zero, because one factor of each product is then always
    :math:`\ln 1`. Simulating the path and taking its true extremes is what
    makes any test of these estimators meaningful.

    Discretisation still biases the observed extremes inward, so range
    estimators read slightly low; 400 steps a day keeps that under a percent.
    """
    rng = np.random.default_rng(seed)
    step_vol = daily_vol / np.sqrt(steps_per_day)
    step_drift = drift / steps_per_day

    increments = rng.standard_normal((n, steps_per_day)) * step_vol + step_drift
    log_path = np.log(100.0) + np.cumsum(increments.ravel()).reshape(n, steps_per_day)
    prices = np.exp(log_path)

    close = prices[:, -1]
    open_ = np.concatenate([[100.0], close[:-1]])
    return {
        "open": open_,
        "high": np.maximum(prices.max(axis=1), np.maximum(open_, close)),
        "low": np.minimum(prices.min(axis=1), np.minimum(open_, close)),
        "close": close,
    }


class TestRealizedVariance:
    def test_log_returns_are_additive(self):
        close = np.array([100.0, 110.0, 99.0])
        returns = log_returns(close)
        assert returns.sum() == pytest.approx(np.log(99.0 / 100.0), rel=1e-14)

    @pytest.mark.parametrize(
        "estimator", ["close_to_close", "parkinson", "garman_klass", "rogers_satchell"]
    )
    def test_estimators_are_non_negative_and_aligned(self, estimator):
        bars = synthetic_bars()
        values = realized_variance(bars, estimator)
        assert values.size == bars["close"].size - 1
        assert np.all(values >= 0)

    def test_range_estimators_are_less_noisy_than_close_to_close(self):
        """The efficiency claim in the module docstring, tested rather than asserted.

        On a constant-volatility path every estimator here is (near) unbiased,
        so the one with the smaller sampling variance is the better proxy.
        Theory puts Parkinson's variance around a fifth of close-to-close's; the
        assertion below only demands a factor of two, leaving room for
        discretisation.
        """
        true_variance = 0.02**2
        bars = synthetic_bars(n=4000, daily_vol=0.02, seed=11)

        ctc = close_to_close(bars["close"])
        park = parkinson(bars["high"], bars["low"])[1:]

        # Both must be estimating the same thing before comparing their noise.
        assert np.mean(ctc) == pytest.approx(true_variance, rel=0.15)
        assert np.mean(park) == pytest.approx(true_variance, rel=0.15)

        assert np.std(park) < 0.5 * np.std(ctc)

    def test_garman_klass_never_returns_negative(self):
        """The clipping matters: a raw GK can go negative on wide-body bars."""
        bars = {
            "open": np.array([100.0]),
            "high": np.array([100.5]),
            "low": np.array([99.5]),
            "close": np.array([100.5]),
        }
        assert garman_klass(bars["open"], bars["high"], bars["low"], bars["close"])[0] >= 0

    def test_rogers_satchell_is_drift_robust(self):
        """Under a strong drift, Parkinson reads high while Rogers-Satchell does not.

        A trending path spends its range travelling rather than oscillating, so
        an estimator that attributes the whole range to volatility overstates
        it. Rogers-Satchell's cross terms cancel the drift; Parkinson's do not.
        """
        vol = 0.01
        truth = vol**2
        drift = 2.0 * vol       # a strong but not degenerate trend

        def estimates(with_drift: float) -> dict[str, float]:
            bars = synthetic_bars(n=6000, daily_vol=vol, drift=with_drift, seed=5)
            return {
                "rs": float(np.mean(rogers_satchell(
                    bars["open"], bars["high"], bars["low"], bars["close"]))),
                "parkinson": float(np.mean(parkinson(bars["high"], bars["low"]))),
                "close_to_close": float(np.mean(close_to_close(bars["close"]))),
            }

        flat = estimates(0.0)
        trend = estimates(drift)

        # Sampling 400 times a day misses a little of each true continuous
        # extreme, so the range estimators read ~9% low even with no drift.
        # That is a property of the simulation, not of the estimators, which is
        # why each is compared against its own driftless value.
        assert flat["rs"] == pytest.approx(truth, rel=0.15)
        assert flat["parkinson"] == pytest.approx(truth, rel=0.15)

        inflation = {k: trend[k] / flat[k] - 1.0 for k in flat}

        # Close-to-close is worst: the drift enters the return directly and is
        # then squared, so it contributes drift^2 outright.
        assert inflation["close_to_close"] > 3.0
        # Parkinson attributes the whole trended range to volatility.
        assert inflation["parkinson"] > 1.0
        # Rogers-Satchell's cross terms cancel the drift and it barely moves.
        assert abs(inflation["rs"]) < 0.10
        assert abs(inflation["rs"]) < inflation["parkinson"]

    def test_annualize(self):
        assert annualize(0.0001, periods=365) == pytest.approx(np.sqrt(0.0365))


class TestBaselines:
    def test_random_walk_carries_the_last_value(self):
        rv = np.array([1e-4, 2e-4, 3e-4])
        model = RandomWalkVariance().fit(np.zeros(3), rv)
        assert model.forecast_variance() == pytest.approx(3e-4)

    def test_ewma_recursion_matches_the_formula(self):
        returns = RNG.standard_normal(500) * 0.02
        model = EWMAVariance(lam=0.94).fit(returns)

        seed_length = max(2, returns.size // 10)
        expected = float(np.var(returns[:seed_length]))
        for r in returns[seed_length:]:
            expected = 0.94 * expected + 0.06 * r * r
        assert model.forecast_variance() == pytest.approx(expected, rel=1e-12)

    def test_ewma_is_flat_across_horizons(self):
        """Persistence is exactly 1, so there is no mean reversion to forecast."""
        model = EWMAVariance().fit(RNG.standard_normal(400) * 0.02)
        assert model.forecast_variance(1) == pytest.approx(model.forecast_variance(20))

    def test_unfitted_model_refuses_to_forecast(self):
        with pytest.raises(RuntimeError):
            RandomWalkVariance().forecast_variance()

    def test_ewma_rejects_bad_lambda(self):
        with pytest.raises(ValueError):
            EWMAVariance(lam=1.0)


class TestGarchRecoversTruth:
    """Simulate from known parameters, then check the estimator finds them.

    This is the only place in part 3 with an answer key, which makes it the
    load-bearing test for the whole forecasting layer.
    """

    def test_parameters_are_recovered(self):
        omega, alpha, beta = 3e-6, 0.09, 0.88
        returns, _ = simulate_garch(6000, omega, alpha, beta, seed=42)
        model = GARCH11().fit(returns)

        assert model.alpha_ == pytest.approx(alpha, abs=0.04)
        assert model.beta_ == pytest.approx(beta, abs=0.05)
        # Persistence is the best-identified combination and should be tight.
        assert model.alpha_ + model.beta_ == pytest.approx(alpha + beta, abs=0.03)

    def test_long_run_variance_is_recovered(self):
        omega, alpha, beta = 3e-6, 0.09, 0.88
        returns, _ = simulate_garch(6000, omega, alpha, beta, seed=43)
        model = GARCH11().fit(returns)
        truth = omega / (1 - alpha - beta)
        assert model.long_run_variance == pytest.approx(truth, rel=0.35)

    def test_reparameterisation_enforces_stationarity(self):
        """alpha + beta < 1 must hold by construction, not by luck."""
        for seed in range(6):
            returns, _ = simulate_garch(400, seed=seed)
            model = GARCH11().fit(returns)
            assert 0 < model.alpha_ < 1
            assert 0 < model.beta_ < 1
            assert model.alpha_ + model.beta_ < 1
            assert model.omega_ > 0

    def test_forecast_mean_reverts_toward_long_run(self):
        returns, _ = simulate_garch(2000, seed=9)
        model = GARCH11().fit(returns)
        near, far = model.forecast_variance(1), model.forecast_variance(250)
        assert abs(far - model.long_run_variance) < abs(near - model.long_run_variance)

    def test_beats_random_walk_on_its_own_process(self):
        """Fitted GARCH must beat a naive carry-forward on GARCH-generated data."""
        returns, variances = simulate_garch(3000, seed=17)
        rv = returns**2

        garch = run_walk_forward(GARCH11, returns, rv, min_train=500, step=20)
        walk = run_walk_forward(RandomWalkVariance, returns, rv, min_train=500, step=20)
        aligned, targets = align_results([garch, walk])

        table = loss_table(aligned, targets)
        assert table["garch(1,1)"]["qlike"] < table["random-walk"]["qlike"]

    def test_rejects_short_samples(self):
        with pytest.raises(ValueError):
            GARCH11().fit(RNG.standard_normal(20) * 0.02)


class TestLosses:
    def test_qlike_is_zero_only_when_exact(self):
        rv = np.array([1e-4, 4e-4, 9e-4])
        assert np.allclose(qlike(rv, rv), 0.0, atol=1e-15)
        assert np.all(qlike(rv * 1.5, rv) > 0)
        assert np.all(qlike(rv * 0.5, rv) > 0)

    def test_qlike_is_scale_free(self):
        """The property that makes QLIKE the right default: only the ratio matters."""
        rv = np.array([1e-4, 5e-4])
        assert np.allclose(qlike(2e-4, 1e-4), qlike(2e-2, 1e-2), rtol=1e-14)

    def test_qlike_penalises_under_forecasting_more(self):
        """Asymmetric in the direction that matches the trading consequence."""
        truth = 1e-4
        assert qlike(truth / 2, truth) > qlike(truth * 2, truth)

    def test_mse_is_symmetric_on_the_variance_scale(self):
        truth = 1e-4
        assert mse(truth + 1e-5, truth) == pytest.approx(mse(truth - 1e-5, truth))

    def test_loss_table_rejects_misaligned_series(self):
        with pytest.raises(ValueError):
            loss_table({"m": np.ones(5)}, np.ones(6))


class TestSignificanceTests:
    def test_diebold_mariano_does_not_reject_identical_forecasts(self):
        losses = np.abs(RNG.standard_normal(300))
        result = diebold_mariano(losses, losses.copy())
        assert result.p_value == pytest.approx(1.0, abs=1e-9)

    def test_diebold_mariano_detects_a_real_difference(self):
        better = np.abs(RNG.standard_normal(400)) * 0.5
        worse = better + 0.4 + np.abs(RNG.standard_normal(400)) * 0.05
        result = diebold_mariano(better, worse)
        assert result.p_value < 0.01
        assert result.favours == "first"
        assert "first" in result.verdict()

    def test_size_is_controlled_under_the_null(self):
        """Rejection rate near 5% when the two models are genuinely equal.

        An uncorrected DM test over-rejects on samples this size; this is what
        the Harvey-Leybourne-Newbold correction is for.
        """
        rejections = 0
        trials = 200
        for seed in range(trials):
            rng = np.random.default_rng(1000 + seed)
            a = np.abs(rng.standard_normal(120))
            b = np.abs(rng.standard_normal(120))
            if diebold_mariano(a, b).p_value < 0.05:
                rejections += 1
        assert rejections / trials < 0.12   # generous, but catches gross over-rejection

    def test_newey_west_handles_autocorrelation(self):
        """A persistent series has a long-run variance above its sample variance."""
        rng = np.random.default_rng(4)
        series = np.zeros(2000)
        for t in range(1, 2000):
            series[t] = 0.8 * series[t - 1] + rng.standard_normal()
        assert newey_west_variance(series) > np.var(series)

    def test_newey_west_is_non_negative(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            assert newey_west_variance(rng.standard_normal(200)) > 0

    def test_diebold_mariano_rejects_tiny_samples(self):
        with pytest.raises(ValueError):
            diebold_mariano(np.ones(4), np.ones(4))

    def test_mincer_zarnowitz_accepts_a_perfect_forecast(self):
        rv = np.abs(RNG.standard_normal(300)) * 1e-4 + 1e-5
        result = mincer_zarnowitz(rv, rv)
        assert result.slope == pytest.approx(1.0, abs=1e-9)
        assert result.intercept == pytest.approx(0.0, abs=1e-12)
        assert result.unbiased

    def test_mincer_zarnowitz_detects_a_scaled_forecast(self):
        rng = np.random.default_rng(8)
        rv = np.abs(rng.standard_normal(400)) * 1e-4 + 1e-5
        inflated = rv * 2.0                      # systematically twice too high
        result = mincer_zarnowitz(inflated, rv)
        assert not result.unbiased
        assert result.slope == pytest.approx(0.5, abs=0.05)


class TestWalkForward:
    def test_splits_never_overlap_the_target(self):
        splits = list(walk_forward_splits(100, min_train=30))
        assert all(split.end <= 99 for split in splits)
        assert all(split.start < split.end for split in splits)

    def test_expanding_grows_and_rolling_does_not(self):
        expanding = list(walk_forward_splits(200, min_train=50, expanding=True))
        rolling = list(walk_forward_splits(200, min_train=50, expanding=False, window=50))
        assert expanding[-1].train_length > expanding[0].train_length
        assert all(split.train_length == 50 for split in rolling)

    def test_requires_room_to_forecast(self):
        with pytest.raises(ValueError):
            list(walk_forward_splits(50, min_train=50))

    def test_failures_are_recorded_not_raised(self):
        class AlwaysFails(Forecaster):
            name = "always-fails"

            def fit(self, returns, realized_variance=None):
                raise RuntimeError("optimiser gave up")

            def forecast_variance(self, horizon=1):
                return 1.0

        returns = RNG.standard_normal(300) * 0.02
        result = run_walk_forward(AlwaysFails, returns, returns**2, min_train=100, step=10)
        assert len(result) == 0
        assert result.failures
        assert "optimiser gave up" in result.failures[0][1]

    def test_non_positive_forecasts_are_rejected(self):
        class ForecastsZero(Forecaster):
            name = "zero"

            def fit(self, returns, realized_variance=None):
                self.fitted_ = True
                return self

            def forecast_variance(self, horizon=1):
                return 0.0

        returns = RNG.standard_normal(300) * 0.02
        result = run_walk_forward(ForecastsZero, returns, returns**2, min_train=100, step=10)
        assert len(result) == 0
        assert all("non-positive" in message for _, message in result.failures)

    def test_align_results_restricts_to_common_dates(self):
        returns = RNG.standard_normal(400) * 0.02
        rv = returns**2
        good = run_walk_forward(RandomWalkVariance, returns, rv, min_train=100, step=10)
        also = run_walk_forward(EWMAVariance, returns, rv, min_train=150, step=10)
        aligned, targets = align_results([good, also])
        assert len(aligned) == 2
        assert all(series.size == targets.size for series in aligned.values())
        assert targets.size == min(len(good), len(also))


class TestNoLookAhead:
    """The forecast must never be computed from data that includes its target."""

    def test_model_only_sees_the_past(self):
        seen: list[tuple[int, int]] = []

        class Spy(Forecaster):
            name = "spy"

            def fit(self, returns, realized_variance=None):
                seen.append((returns.size, int(np.argmax(returns))))
                self.fitted_ = True
                self.value_ = float(np.var(returns))
                return self

            def forecast_variance(self, horizon=1):
                return self.value_

        # A distinctive spike late in the sample: if any training window
        # contains it before its own index, the slicing is wrong.
        returns = RNG.standard_normal(400) * 0.01
        spike_index = 350
        returns[spike_index] = 100.0

        result = run_walk_forward(Spy, returns, returns**2, min_train=100, step=1)

        # Every fit call must have seen strictly fewer observations than the
        # index of the target it produced.
        for (train_size, _), target_index in zip(seen, result.indices):
            assert train_size <= target_index

    def test_target_is_strictly_after_the_training_window(self):
        returns = np.arange(300, dtype=float) * 1e-3
        rv = returns**2 + 1e-8

        class Echo(Forecaster):
            name = "echo"

            def fit(self, returns, realized_variance=None):
                self.last_ = float(returns[-1])
                self.fitted_ = True
                return self

            def forecast_variance(self, horizon=1):
                return abs(self.last_) + 1e-9

        result = run_walk_forward(Echo, returns, rv, min_train=100, step=1)
        # Forecast n was built from returns[n-1], so it must be strictly below
        # the return at the target index.
        for forecast, index in zip(result.forecasts, result.indices):
            assert forecast < returns[index] + 1e-9


class TestBacktest:
    def _series(self, n=400, vol=0.02, seed=1):
        rng = np.random.default_rng(seed)
        return 100 * np.exp(np.cumsum(rng.standard_normal(n) * vol))

    def test_no_signal_means_no_trades(self):
        spot = self._series()
        vol = np.full(spot.size, 0.35)
        result = run_backtest(spot, vol, vol)     # forecast equals implied
        assert result.n_trades == 0
        assert result.total_return == 0.0

    def test_a_correct_view_makes_money(self):
        """Realised volatility well above implied should pay a long straddle.

        The strategy buys volatility when the forecast exceeds implied; here the
        forecast is right and the option is genuinely underpriced, so a
        delta-hedged long position must profit on average.
        """
        daily_vol = 0.04
        spot = self._series(n=600, vol=daily_vol, seed=2)
        realised_annual = daily_vol * np.sqrt(365)
        implied = np.full(spot.size, realised_annual * 0.5)   # market too cheap
        forecast = np.full(spot.size, realised_annual)        # we are right

        result = run_backtest(
            spot, forecast, implied,
            BacktestConfig(holding_days=5, transaction_cost=0.0),
        )
        assert result.n_trades > 0
        assert result.total_return > 0

    def test_a_wrong_view_loses_money(self):
        """The mirror image, and the guard against a backtest that always wins."""
        daily_vol = 0.04
        spot = self._series(n=600, vol=daily_vol, seed=2)
        realised_annual = daily_vol * np.sqrt(365)
        implied = np.full(spot.size, realised_annual * 2.0)   # market too rich
        forecast = np.full(spot.size, realised_annual * 4.0)  # we are more wrong

        result = run_backtest(
            spot, forecast, implied,
            BacktestConfig(holding_days=5, transaction_cost=0.0),
        )
        assert result.n_trades > 0
        assert result.total_return < 0

    def test_costs_only_ever_hurt(self):
        spot = self._series(n=600, vol=0.04, seed=2)
        annual = 0.04 * np.sqrt(365)
        results = cost_sweep(
            spot, np.full(spot.size, annual), np.full(spot.size, annual * 0.5)
        )
        totals = [results[c].total_return for c in sorted(results)]
        assert all(later <= earlier + 1e-9 for earlier, later in zip(totals, totals[1:]))

    def test_rejects_misaligned_inputs(self):
        with pytest.raises(ValueError):
            run_backtest(np.ones(10), np.ones(10), np.ones(9))

    def test_hit_rate_and_trade_count_are_scored_on_settlements(self):
        """Regression: both were computed from the wrong series.

        P&L is booked on the one day a trade settles, but ``positions`` is set
        for every day the trade is open. Averaging P&L over open days counted
        the flat days of each hold as losses, which produced the contradictory
        combination of a Sharpe near 3 alongside a 10% hit rate. Counting
        transitions in ``positions`` double-counted every trade for the same
        structural reason.
        """
        spot = self._series(n=600, vol=0.04, seed=2)
        annual = 0.04 * np.sqrt(365)
        result = run_backtest(
            spot, np.full(spot.size, annual), np.full(spot.size, annual * 0.5),
            BacktestConfig(holding_days=5, transaction_cost=0.0),
        )

        settled = result.pnl[result.pnl != 0.0]
        assert result.n_trades == settled.size
        assert result.hit_rate == pytest.approx(float(np.mean(settled > 0)))

        # A profitable strategy cannot have a hit rate near zero; the two
        # numbers have to tell a consistent story.
        assert result.total_return > 0
        assert result.hit_rate > 0.5

        # Days in market must exceed settlements, since each hold spans days.
        assert result.diagnostics["days_in_market"] > result.n_trades

    def test_summary_reports_the_headline_numbers(self):
        spot = self._series(n=400, vol=0.04, seed=6)
        annual = 0.04 * np.sqrt(365)
        result = run_backtest(
            spot, np.full(spot.size, annual), np.full(spot.size, annual * 0.6)
        )
        text = result.summary()
        for field in ("Sharpe", "max DD", "hit rate", "trades"):
            assert field in text


class TestTrailingMean:
    """Guards the proxy-construction mistake that inflated example 05.

    A window ending at ``i`` rather than ``i-1`` leaks the current
    observation into a signal that is about to trade on it.
    """

    def test_excludes_the_current_observation(self):
        series = np.arange(10, dtype=float)
        result = trailing_mean(series, window=3)
        assert result[5] == pytest.approx(np.mean(series[2:5]))
        assert result[9] == pytest.approx(np.mean(series[6:9]))

    def test_a_spike_never_appears_before_its_own_index(self):
        series = np.zeros(50)
        series[30] = 1000.0
        result = trailing_mean(series, window=5)
        assert np.all(result[:31] == 0.0)   # nothing up to and including day 30
        assert result[31] > 0.0             # visible from the next day on

    def test_leading_values_use_what_history_exists(self):
        series = np.array([4.0, 6.0, 8.0, 10.0])
        result = trailing_mean(series, window=10)
        assert result[0] == pytest.approx(4.0)       # no history: fall back to first
        assert result[1] == pytest.approx(4.0)
        assert result[2] == pytest.approx(5.0)       # mean of [4, 6]
        assert result[3] == pytest.approx(6.0)       # mean of [4, 6, 8]

    def test_matches_a_naive_loop(self):
        rng = np.random.default_rng(21)
        series = rng.standard_normal(200)
        result = trailing_mean(series, window=7)
        for i in range(1, series.size):
            expected = np.mean(series[max(0, i - 7) : i])
            assert result[i] == pytest.approx(expected)

class TestResearchStubs:
    """Unimplemented models must fail loudly, never return a plausible number."""

    @pytest.mark.parametrize(
        "model_class", [GJRGarch, MarkovSwitchingVariance, HARRV, NeuralVolatility]
    )
    def test_stubs_raise_rather_than_guess(self, model_class):
        with pytest.raises(NotImplementedError):
            model_class().fit(RNG.standard_normal(500) * 0.02)

    @pytest.mark.parametrize(
        "model_class", [GJRGarch, MarkovSwitchingVariance, HARRV, NeuralVolatility]
    )
    def test_stubs_conform_to_the_interface(self, model_class):
        """They must already be usable by the evaluator once filled in."""
        model = model_class()
        assert isinstance(model, Forecaster)
        assert model.name

    @pytest.mark.parametrize(
        "model_class", [GJRGarch, MarkovSwitchingVariance, HARRV, NeuralVolatility]
    )
    def test_stubs_document_their_specification(self, model_class):
        """The docstring is the spec; an empty one makes the stub useless."""
        assert model_class.__doc__ is not None
        assert len(model_class.__doc__) > 400
