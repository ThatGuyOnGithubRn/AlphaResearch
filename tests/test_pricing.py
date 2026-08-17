"""BSM pricing: published reference values, parity, carry regimes, edge cases."""

import math

import numpy as np
import pytest

from qar.core.bsm import (
    CALL,
    PUT,
    BSMInputs,
    carry_dividend,
    carry_futures,
    carry_fx,
    price,
)


class TestPublishedValues:
    """Against figures from the standard references, not self-generated."""

    def test_hull_example(self):
        """Hull, *Options, Futures and Other Derivatives*: S=42 K=40 T=0.5 r=10% sig=20%.

        Hull quotes 4.76 and 0.81 to two decimals, so the tolerance matches the
        precision of the published figure rather than the precision of our own
        arithmetic. Parity below pins the pair together far more tightly than
        the reference values can.
        """
        inputs = BSMInputs(S=42, K=40, T=0.5, r=0.10, sigma=0.20)
        assert float(price(inputs, CALL)) == pytest.approx(4.76, abs=5e-3)
        assert float(price(inputs, PUT)) == pytest.approx(0.81, abs=5e-3)

    def test_black76_on_futures(self):
        """b = 0 must reduce to Black-76: c = e^{-rT}[F N(d1) - K N(d2)]."""
        F, K, T, r, sigma = 100.0, 95.0, 0.75, 0.04, 0.30
        inputs = BSMInputs(
            S=F, K=K, T=T, r=r, sigma=sigma, b=carry_futures(),
            carry_depends_on_r=False,
        )
        d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        from qar.core.distributions import norm_cdf

        expected = math.exp(-r * T) * (F * norm_cdf(d1) - K * norm_cdf(d2))
        assert float(price(inputs, CALL)) == pytest.approx(expected, rel=1e-14)

    def test_garman_kohlhagen_fx(self):
        """FX call = domestic-discounted, foreign-discounted spot form."""
        S, K, T, r_d, r_f, sigma = 1.20, 1.25, 1.0, 0.04, 0.01, 0.12
        inputs = BSMInputs(S=S, K=K, T=T, r=r_d, sigma=sigma, b=carry_fx(r_d, r_f))
        from qar.core.distributions import norm_cdf

        d1 = (math.log(S / K) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        expected = S * math.exp(-r_f * T) * norm_cdf(d1) - K * math.exp(-r_d * T) * norm_cdf(d2)
        assert float(price(inputs, CALL)) == pytest.approx(expected, rel=1e-14)


class TestPutCallParity:
    """Parity is model-free, so it must hold exactly at every parameter point."""

    @pytest.mark.parametrize("S", [50.0, 100.0, 180.0])
    @pytest.mark.parametrize("T", [1 / 365, 0.5, 3.0])
    @pytest.mark.parametrize("sigma", [0.05, 0.4, 1.5])
    @pytest.mark.parametrize("b_offset", [0.0, -0.03, 0.06])
    def test_parity_holds(self, S, T, sigma, b_offset):
        r = 0.045
        inputs = BSMInputs(S=S, K=100.0, T=T, r=r, sigma=sigma, b=r + b_offset)
        left = float(price(inputs, CALL)) - float(price(inputs, PUT))
        right = float(inputs.S * inputs.df_carry - inputs.K * inputs.df_r)
        assert left == pytest.approx(right, abs=1e-11)


class TestBounds:
    @pytest.mark.parametrize("K", [60.0, 100.0, 150.0])
    def test_prices_respect_model_free_bounds(self, K):
        inputs = BSMInputs(S=100.0, K=K, T=0.6, r=0.03, sigma=0.35, b=0.01)
        call, put = float(price(inputs, CALL)), float(price(inputs, PUT))
        forward_pv = float(inputs.S * inputs.df_carry)
        strike_pv = float(inputs.K * inputs.df_r)

        assert max(forward_pv - strike_pv, 0.0) - 1e-12 <= call <= forward_pv + 1e-12
        assert max(strike_pv - forward_pv, 0.0) - 1e-12 <= put <= strike_pv + 1e-12

    def test_call_decreasing_in_strike(self):
        inputs = BSMInputs(S=100.0, K=80.0, T=1.0, r=0.03, sigma=0.3)
        prices = [float(price(inputs.replace(K=k), CALL)) for k in np.linspace(60, 160, 60)]
        assert all(later <= earlier + 1e-12 for earlier, later in zip(prices, prices[1:]))

    def test_call_convex_in_strike(self):
        """Convexity in strike, i.e. a non-negative implied density."""
        inputs = BSMInputs(S=100.0, K=100.0, T=1.0, r=0.03, sigma=0.3)
        strikes = np.linspace(60, 160, 60)
        prices = np.array([float(price(inputs.replace(K=k), CALL)) for k in strikes])
        second_difference = prices[:-2] - 2 * prices[1:-1] + prices[2:]
        assert np.all(second_difference >= -1e-12)


class TestDegenerateLimits:
    """Zero time and zero volatility must return limits, never nan."""

    @pytest.mark.parametrize("S,K,expected_call", [(120.0, 100.0, 20.0), (80.0, 100.0, 0.0)])
    def test_zero_maturity_returns_intrinsic(self, S, K, expected_call):
        inputs = BSMInputs(S=S, K=K, T=0.0, r=0.05, sigma=0.3)
        assert float(price(inputs, CALL)) == pytest.approx(expected_call, abs=1e-12)
        assert float(price(inputs, PUT)) == pytest.approx(max(K - S, 0.0), abs=1e-12)

    def test_zero_volatility_returns_discounted_forward_intrinsic(self):
        S, K, T, r, b = 105.0, 100.0, 2.0, 0.05, 0.02
        inputs = BSMInputs(S=S, K=K, T=T, r=r, sigma=0.0, b=b)
        expected = math.exp(-r * T) * max(S * math.exp(b * T) - K, 0.0)
        assert float(price(inputs, CALL)) == pytest.approx(expected, rel=1e-13)

    def test_at_the_forward_zero_vol_is_zero(self):
        """The knife edge: forward exactly at the strike, no volatility."""
        r, T, b = 0.05, 1.0, 0.05
        S = 100.0 * math.exp(-b * T)
        inputs = BSMInputs(S=S, K=100.0, T=T, r=r, sigma=0.0, b=b)
        assert float(price(inputs, CALL)) == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize("S", [1e-6, 1e6])
    def test_extreme_moneyness_stays_finite(self, S):
        inputs = BSMInputs(S=S, K=100.0, T=1.0, r=0.03, sigma=0.4)
        for kind in (CALL, PUT):
            value = float(price(inputs, kind))
            assert math.isfinite(value)
            assert value >= -1e-12


class TestInputValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"S": -1.0}, {"S": 0.0}, {"K": -5.0}, {"T": -0.1}, {"sigma": -0.2},
        ],
    )
    def test_rejects_impossible_inputs(self, kwargs):
        base = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.03, "sigma": 0.2}
        base.update(kwargs)
        with pytest.raises(ValueError):
            BSMInputs(**base)

    def test_carry_defaults_to_rate(self):
        assert BSMInputs(S=100, K=100, T=1, r=0.07, sigma=0.2).b == 0.07

    def test_rejects_unknown_kind(self):
        inputs = BSMInputs(S=100, K=100, T=1, r=0.03, sigma=0.2)
        with pytest.raises(ValueError):
            price(inputs, "straddle")


class TestVectorisation:
    def test_array_inputs_match_scalar_loop(self):
        strikes = np.array([80.0, 100.0, 125.0])
        vector = price(BSMInputs(S=100.0, K=strikes, T=1.0, r=0.03, sigma=0.25), CALL)
        scalar = [
            float(price(BSMInputs(S=100.0, K=float(k), T=1.0, r=0.03, sigma=0.25), CALL))
            for k in strikes
        ]
        assert np.allclose(vector, scalar, rtol=1e-15)

    def test_scalar_input_returns_python_float(self):
        value = price(BSMInputs(S=100, K=100, T=1, r=0.03, sigma=0.2), CALL)
        assert isinstance(value, float)


def test_dividend_carry_lowers_call_raises_put():
    """A dividend yield transfers value from calls to puts."""
    base = BSMInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.3, b=0.05)
    with_dividend = base.replace(b=carry_dividend(0.05, 0.04))
    assert float(price(with_dividend, CALL)) < float(price(base, CALL))
    assert float(price(with_dividend, PUT)) > float(price(base, PUT))
