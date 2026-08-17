"""Implied volatility: round-trip accuracy, wing robustness, honest failure."""

import pytest

import numpy as np

from qar.core.bsm import CALL, PUT, BSMInputs, price
from qar.core.iv import NoImpliedVolError, implied_vol


def fair(S, K, T, r, sigma, b=None, kind=CALL):
    return float(price(BSMInputs(S=S, K=K, T=T, r=r, sigma=sigma, b=b), kind))


class TestRoundTrip:
    @pytest.mark.parametrize("sigma", [0.05, 0.15, 0.30, 0.75, 1.50])
    @pytest.mark.parametrize("K", [70.0, 100.0, 140.0])
    @pytest.mark.parametrize("kind", [CALL, PUT])
    def test_reprices_the_input(self, sigma, K, kind):
        """The solver's actual contract: the returned vol reproduces the price."""
        S, T, r = 100.0, 0.75, 0.04
        target = fair(S, K, T, r, sigma, kind=kind)
        result = implied_vol(target, S=S, K=K, T=T, r=r, kind=kind)
        assert result.converged
        assert fair(S, K, T, r, result.sigma, kind=kind) == pytest.approx(
            target, abs=1e-9
        )

    @pytest.mark.parametrize("sigma", [0.05, 0.15, 0.30, 0.75, 1.50])
    @pytest.mark.parametrize("K", [70.0, 100.0, 140.0])
    @pytest.mark.parametrize("kind", [CALL, PUT])
    def test_recovers_the_input_volatility_where_vega_permits(self, sigma, K, kind):
        r"""Volatility is only *identifiable* where the price responds to it.

        A 5%-vol call struck at 70 with spot at 100 is worth its forward
        intrinsic to nine decimal places; its Vega is around 1e-9, so a whole
        range of volatilities reprices it within any sane tolerance. Demanding
        that the solver return the exact input there tests nothing about the
        solver and everything about floating point.

        The condition below is the honest one: recovery is required precisely
        where a 1e-9 price tolerance pins sigma to better than 1e-6, i.e. where
        :math:`\mathcal{V} \gtrsim 10^{-3}`.
        """
        from qar.greeks import vega

        S, T, r = 100.0, 0.75, 0.04
        inputs = BSMInputs(S=S, K=K, T=T, r=r, sigma=sigma)
        if float(vega(inputs)) < 1e-3:
            pytest.skip("vega too small for sigma to be identifiable")

        target = fair(S, K, T, r, sigma, kind=kind)
        result = implied_vol(target, S=S, K=K, T=T, r=r, kind=kind)
        assert result.sigma == pytest.approx(sigma, rel=1e-6)

    @pytest.mark.parametrize("b", [0.0, 0.01, -0.05])
    def test_works_across_carry_regimes(self, b):
        S, K, T, r, sigma = 100.0, 105.0, 1.0, 0.03, 0.28
        target = fair(S, K, T, r, sigma, b=b)
        result = implied_vol(target, S=S, K=K, T=T, r=r, b=b)
        assert result.sigma == pytest.approx(sigma, rel=1e-6)

    def test_short_dated_atm(self):
        """One day to expiry — where Newton is most likely to misbehave."""
        S, K, T, r, sigma = 100.0, 100.0, 1 / 365, 0.05, 0.40
        target = fair(S, K, T, r, sigma)
        result = implied_vol(target, S=S, K=K, T=T, r=r)
        assert result.sigma == pytest.approx(sigma, rel=1e-6)


class TestWings:
    @pytest.mark.parametrize("K", [40.0, 250.0])
    def test_deep_wings_still_converge(self, K):
        """Vega is near zero here, so Newton alone would stall or diverge."""
        S, T, r, sigma = 100.0, 0.5, 0.03, 0.45
        target = fair(S, K, T, r, sigma)
        result = implied_vol(target, S=S, K=K, T=T, r=r)
        assert result.converged
        assert result.sigma == pytest.approx(sigma, rel=1e-5)

    def test_bisection_fallback_is_reachable(self):
        """Confirm the fallback is live code, not an untested branch."""
        S, K, T, r, sigma = 100.0, 300.0, 0.05, 0.03, 0.9
        target = fair(S, K, T, r, sigma)
        result = implied_vol(target, S=S, K=K, T=T, r=r)
        assert result.converged
        assert result.method in ("bisection", "mixed")


class TestFailureModes:
    def test_below_intrinsic_raises(self):
        with pytest.raises(NoImpliedVolError, match="below"):
            implied_vol(0.5, S=100.0, K=70.0, T=0.5, r=0.05, kind=CALL)

    def test_above_upper_bound_raises(self):
        """A call worth more than the underlying has no implied volatility.

        Note it takes a price above spot, not merely a large one: at 1000% vol
        the call is already worth ~99.9% of spot, so 99.0 is perfectly
        attainable and rejecting it would be a false alarm.
        """
        with pytest.raises(NoImpliedVolError):
            implied_vol(101.0, S=100.0, K=100.0, T=0.5, r=0.05, kind=CALL)

    def test_quote_at_intrinsic_reprices_even_though_sigma_is_unidentified(self):
        """At the zero-volatility bound, sigma is not recoverable — and that is correct.

        A deep in-the-money call is worth its forward intrinsic for every
        volatility up to a few percent, so the solver returns *a* root rather
        than *the* root. What it must still guarantee is that the root it
        returns reproduces the quoted price.
        """
        S, K, T, r = 100.0, 90.0, 0.5, 0.05
        floor = fair(S, K, T, r, 1e-9)
        result = implied_vol(floor, S=S, K=K, T=T, r=r)
        assert result.converged
        assert fair(S, K, T, r, result.sigma) == pytest.approx(floor, abs=1e-9)


def test_result_converts_to_float():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.03, 0.25
    result = implied_vol(fair(S, K, T, r, sigma), S=S, K=K, T=T, r=r)
    assert float(result) == pytest.approx(sigma, rel=1e-6)


def test_converges_in_few_iterations_when_vega_is_healthy():
    """Newton should dominate near the money; a slow solve signals a bad seed."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.03, 0.25
    result = implied_vol(fair(S, K, T, r, sigma), S=S, K=K, T=T, r=r)
    assert result.iterations <= 8
    assert result.method == "newton"
