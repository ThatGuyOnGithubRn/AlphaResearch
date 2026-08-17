r"""Structural identities the Greeks must satisfy, proved from put-call parity.

These tests need no numerical differentiation at all, which makes them the
sharpest diagnostics in the suite: a failure points at an algebra error in one
specific formula rather than at a tolerance.

Everything here descends from differentiating

.. math:: c - p = S e^{(b-r)T} - K e^{-rT}

term by term. The right-hand side is linear in :math:`S`, free of
:math:`\sigma`, and exponential in :math:`T` and :math:`r`, so each derivative
of it is either a constant, zero, or a known closed form.
"""

import math

import numpy as np
import pytest

from qar.core.bsm import CALL, PUT, BSMInputs
from qar.greeks import REGISTRY, delta, dual_delta, epsilon, gamma, rho, theta, vega

GRID = [
    BSMInputs(S=S, K=K, T=T, r=r, sigma=sigma, b=b, carry_depends_on_r=tracks)
    for S in (80.0, 100.0, 130.0)
    for K in (90.0, 100.0, 115.0)
    for T in (0.08, 1.0, 2.5)
    for r, b, tracks in ((0.05, 0.05, True), (0.05, 0.01, True), (0.03, 0.0, False))
    for sigma in (0.15, 0.45)
]


@pytest.mark.parametrize("inputs", GRID, ids=lambda i: f"S{i.S:g}K{i.K:g}T{i.T:g}b{i.b:g}")
class TestParityDerivedIdentities:
    def test_delta_difference_is_carry_discount(self, inputs):
        r"""$\partial_S$ of parity: $\Delta_c - \Delta_p = e^{(b-r)T}$."""
        difference = float(delta(inputs, CALL)) - float(delta(inputs, PUT))
        assert difference == pytest.approx(float(inputs.df_carry), rel=1e-14)

    def test_dual_delta_difference_is_strike_discount(self, inputs):
        r"""$\partial_K$ of parity: $\partial_K c - \partial_K p = -e^{-rT}$."""
        difference = float(dual_delta(inputs, CALL)) - float(dual_delta(inputs, PUT))
        assert difference == pytest.approx(-float(inputs.df_r), rel=1e-14)

    def test_rho_difference(self, inputs):
        r"""$\partial_r$ of parity.

        When the carry tracks the rate, :math:`e^{(b-r)T}` drops out and only
        the strike term contributes, giving :math:`T K e^{-rT}`. When the carry
        is pinned, the spot leg contributes as well.
        """
        difference = float(rho(inputs, CALL)) - float(rho(inputs, PUT))
        if inputs.carry_depends_on_r:
            expected = inputs.T * inputs.K * float(inputs.df_r)
        else:
            expected = float(
                inputs.T * (inputs.K * inputs.df_r - inputs.S * inputs.df_carry)
            )
        assert difference == pytest.approx(expected, rel=1e-12)

    def test_epsilon_difference(self, inputs):
        r"""$\partial_q$ of parity: only the spot leg carries $q$."""
        difference = float(epsilon(inputs, CALL)) - float(epsilon(inputs, PUT))
        expected = -float(inputs.S * inputs.T * inputs.df_carry)
        assert difference == pytest.approx(expected, rel=1e-13)

    def test_theta_difference(self, inputs):
        r"""$\partial_t$ of parity, i.e. minus $\partial_T$ of the right-hand side."""
        difference = float(theta(inputs, CALL)) - float(theta(inputs, PUT))
        expected = -float(
            inputs.S * (inputs.b - inputs.r) * inputs.df_carry
            + inputs.r * inputs.K * inputs.df_r
        )
        assert difference == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("inputs", GRID, ids=lambda i: f"S{i.S:g}K{i.K:g}T{i.T:g}b{i.b:g}")
@pytest.mark.parametrize("spec", [s for s in REGISTRY if s.call_put_invariant], ids=lambda s: s.name)
def test_call_put_invariance(inputs, spec):
    """Greeks parity says are kind-independent must be bit-for-bit identical.

    Parity is linear in ``S`` and free of ``sigma``, so any derivative taken
    twice in ``S``, or at all in ``sigma``, annihilates the difference between
    the call and the put.
    """
    call_value = float(spec.func(inputs, CALL))
    put_value = float(spec.func(inputs, PUT))
    assert call_value == pytest.approx(put_value, rel=1e-14, abs=1e-300)


@pytest.mark.parametrize("inputs", GRID, ids=lambda i: f"S{i.S:g}K{i.K:g}T{i.T:g}b{i.b:g}")
class TestSignsAndRanges:
    def test_gamma_and_vega_non_negative(self, inputs):
        """Long options are convex in spot and long volatility. Always."""
        assert float(gamma(inputs)) >= -1e-15
        assert float(vega(inputs)) >= -1e-15

    def test_delta_within_carry_adjusted_bounds(self, inputs):
        bound = float(inputs.df_carry)
        assert -1e-15 <= float(delta(inputs, CALL)) <= bound + 1e-15
        assert -bound - 1e-15 <= float(delta(inputs, PUT)) <= 1e-15

    def test_epsilon_signs(self, inputs):
        """Dividends hurt calls and help puts."""
        assert float(epsilon(inputs, CALL)) <= 1e-15
        assert float(epsilon(inputs, PUT)) >= -1e-15

    def test_dual_delta_signs(self, inputs):
        """Calls fall in strike, puts rise."""
        assert float(dual_delta(inputs, CALL)) <= 1e-15
        assert float(dual_delta(inputs, PUT)) >= -1e-15


class TestMixedPartialSymmetry:
    r"""Vanna is $\partial^2 V/\partial S\partial\sigma$ either way round.

    Clairaut's theorem guarantees the mixed partial is symmetric, so computing
    it as :math:`\partial\Delta/\partial\sigma` and as
    :math:`\partial\mathcal{V}/\partial S` must agree. Both routes go through
    different analytical formulas here, so agreement is real evidence.
    """

    @staticmethod
    def _noise_floor(function_value: float, variable_scale: float) -> float:
        r"""Round-off floor of a first-order central difference.

        The differenced quantity carries absolute round-off :math:`\varepsilon
        |f|`, and dividing by a step of :math:`\varepsilon^{1/3}\,s` amplifies
        it to :math:`\varepsilon^{2/3}|f|/s`.

        This floor is not optional. Richardson's error estimate is the spread
        between two step sizes, and that spread can come out at exactly zero
        when both land on the same floating-point value — which says the two
        agree, not that the answer is exact. Trusting a zero estimate would
        demand infinite precision from a finite difference.
        """
        eps = float(np.finfo(float).eps)
        return eps ** (2.0 / 3.0) * abs(function_value) / variable_scale

    @pytest.mark.parametrize("inputs", GRID[::7], ids=lambda i: f"S{i.S:g}K{i.K:g}")
    def test_vanna_both_ways(self, inputs):
        from qar.greeks.second import vanna
        from qar.validate.fd import differentiate

        analytic = float(vanna(inputs))

        via_delta = differentiate(lambda i: delta(i, CALL), inputs, ("sigma",), order=1)
        floor = self._noise_floor(float(delta(inputs, CALL)), float(inputs.sigma))
        tolerance = max(1e-7 * abs(analytic), 10 * via_delta.error_estimate, floor)
        assert abs(analytic - via_delta.value) <= tolerance

        via_vega = differentiate(lambda i: vega(i, CALL), inputs, ("S",), order=1)
        floor = self._noise_floor(float(vega(inputs, CALL)), float(inputs.S))
        tolerance = max(1e-7 * abs(analytic), 10 * via_vega.error_estimate, floor)
        assert abs(analytic - via_vega.value) <= tolerance


def test_registry_covers_every_exported_greek():
    """A newly added Greek must be registered, or it escapes the validator."""
    import qar.greeks as greeks_module

    registered = {spec.name for spec in REGISTRY}
    exported = {
        name
        for name in greeks_module.__all__
        if callable(getattr(greeks_module, name, None)) and not name[0].isupper()
    }
    # `vomma` is a documented alias for `volga`, deliberately not double-counted;
    # `by_order` is a registry helper, not a Greek.
    assert exported - registered == {"vomma", "by_order"}
    assert len(REGISTRY) == 17


def test_rho_regimes_actually_differ():
    """The carry_depends_on_r flag must change the answer, or it is decoration."""
    tracking = BSMInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.3, b=0.0,
                         carry_depends_on_r=True)
    pinned = tracking.replace(carry_depends_on_r=False)
    assert float(rho(tracking, CALL)) != pytest.approx(float(rho(pinned, CALL)), rel=1e-6)
