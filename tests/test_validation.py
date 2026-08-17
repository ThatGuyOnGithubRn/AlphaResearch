"""Every analytical Greek against an independent numerical derivative.

Part 2 of the project. Two independent channels, so an error in one cannot
quietly confirm an error in the other:

*   **Complex step** for first derivatives — exact to machine precision, and
    entirely free of the subtractive cancellation that limits differencing.
*   **Richardson-extrapolated central differences** for second and third order,
    compared against the scheme's own estimate of its error.
"""

import numpy as np
import pytest

from qar.core.bsm import CALL, PUT, BSMInputs, price
from qar.greeks import REGISTRY, by_order, delta, vega
from qar.validate import sweep, validate_greek
from qar.validate.complex_step import (
    CS_STEP,
    delta_by_complex_step,
    vega_by_complex_step,
)
from qar.validate.fd import differentiate
from qar.validate.harness import default_grid

REPRESENTATIVE = [
    BSMInputs(S=100.0, K=90.0, T=0.5, r=0.05, sigma=0.25, b=0.05),
    BSMInputs(S=100.0, K=100.0, T=1.0, r=0.04, sigma=0.35, b=0.01),
    BSMInputs(S=100.0, K=120.0, T=2.0, r=0.02, sigma=0.55, b=0.0,
              carry_depends_on_r=False),
    BSMInputs(S=100.0, K=100.0, T=0.25, r=-0.01, sigma=0.20, b=-0.01),
]


class TestComplexStep:
    """Delta and Vega — the two the project bullet names explicitly."""

    @pytest.mark.parametrize("inputs", REPRESENTATIVE, ids=lambda i: f"K{i.K:g}T{i.T:g}")
    @pytest.mark.parametrize("kind", [CALL, PUT])
    def test_delta_to_machine_precision(self, inputs, kind):
        analytic = float(delta(inputs, kind))
        numerical = delta_by_complex_step(inputs, kind)
        assert numerical == pytest.approx(analytic, rel=1e-13)

    @pytest.mark.parametrize("inputs", REPRESENTATIVE, ids=lambda i: f"K{i.K:g}T{i.T:g}")
    @pytest.mark.parametrize("kind", [CALL, PUT])
    def test_vega_to_machine_precision(self, inputs, kind):
        analytic = float(vega(inputs, kind))
        numerical = vega_by_complex_step(inputs, kind)
        assert numerical == pytest.approx(analytic, rel=1e-13)

    def test_complex_step_beats_finite_differences(self):
        """The claim that motivates the whole method, tested rather than asserted.

        Complex step should be several orders of magnitude more accurate than
        even a Richardson-extrapolated central difference.
        """
        inputs = BSMInputs(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.3)
        analytic = float(delta(inputs, CALL))

        complex_error = abs(delta_by_complex_step(inputs, CALL) - analytic)
        fd_error = abs(
            differentiate(lambda i: price(i, CALL), inputs, ("S",), order=1).value
            - analytic
        )

        assert complex_error < 1e-15
        assert complex_error < fd_error / 100.0

    def test_step_is_small_enough_to_kill_truncation(self):
        """Truncation error is O(h^2); at h=1e-200 that is exactly zero in float64."""
        assert CS_STEP**2 == 0.0


class TestFullSweep:
    """The headline result: all 17 Greeks, both kinds, across the whole grid."""

    @pytest.mark.slow
    def test_every_greek_every_point(self):
        rows = sweep()
        failures = [r for r in rows if not r.passed]
        assert not failures, "\n".join(
            f"{r.greek} ({r.kind}) {r.label}: analytic={r.analytic:.10g} "
            f"numerical={r.numerical:.10g} rel={r.rel_error:.3e} tol={r.tolerance:.3e}"
            for r in failures[:20]
        )

    @pytest.mark.slow
    def test_sweep_is_actually_exercising_things(self):
        """Guard against a sweep that trivially passes by skipping everything."""
        rows = sweep()
        resolved = [r for r in rows if not r.negligible]
        assert len(resolved) > 0.7 * len(rows)
        assert len({r.greek for r in resolved}) == 17

    @pytest.mark.parametrize("spec", REGISTRY, ids=lambda s: s.name)
    @pytest.mark.parametrize("kind", [CALL, PUT])
    def test_greek_on_representative_points(self, spec, kind):
        """Fast per-Greek check, so a failure names the culprit directly."""
        for inputs in REPRESENTATIVE:
            row = validate_greek(spec, inputs, kind)
            assert row.passed, (
                f"{spec.name} ({kind}) at {row.label}: "
                f"analytic={row.analytic:.10g} numerical={row.numerical:.10g} "
                f"rel={row.rel_error:.3e} tol={row.tolerance:.3e}"
            )


class TestValidatorItself:
    """A validator nobody validates is just a second opinion from the same brain."""

    def test_detects_a_deliberately_wrong_greek(self):
        """Perturb a correct formula by 1% — the harness must reject it."""
        from qar.greeks import GreekSpec

        broken = GreekSpec("broken_delta", lambda i, k: delta(i, k) * 1.01, 1, ("S",))
        row = validate_greek(broken, REPRESENTATIVE[1], CALL)
        assert not row.passed

    def test_detects_a_sign_error(self):
        """The classic failure mode: right magnitude, wrong sign."""
        from qar.greeks import GreekSpec

        flipped = GreekSpec("flipped_vega", lambda i, k: -vega(i, k), 1, ("sigma",))
        row = validate_greek(flipped, REPRESENTATIVE[1], CALL)
        assert not row.passed

    def test_detects_a_missing_carry_term(self):
        """Carry-term bugs are invisible at b = r, so test where b != r.

        This is exactly why the grid includes dividend and futures regimes.
        """
        from qar.greeks import GreekSpec
        from qar.greeks._common import prepare, resolve_kind

        def theta_missing_carry(inputs, kind):
            phi = resolve_kind(kind).sign
            state = prepare(inputs)
            decay = -(state.S * state.df_carry * state.n1 * state.sigma) / (
                2.0 * state.safe_sqrt_T
            )
            discount = -phi * state.r * state.K * state.df_r * state.N(phi * state.d2)
            return float(np.asarray(decay + discount))  # carry term dropped

        spec = GreekSpec("theta_no_carry", theta_missing_carry, 1, ("T",), sign=-1.0)

        at_bsm = BSMInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.3, b=0.05)
        assert validate_greek(spec, at_bsm, CALL).passed, (
            "with b = r the dropped term is zero, so this must still pass — "
            "which is precisely why a grid that only tests b = r proves nothing"
        )

        with_dividend = at_bsm.replace(b=0.01)
        assert not validate_greek(spec, with_dividend, CALL).passed

    def test_richardson_improves_on_plain_differencing(self):
        inputs = BSMInputs(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.3)
        from qar.greeks.second import gamma

        analytic = float(gamma(inputs))
        plain = differentiate(
            lambda i: price(i, CALL), inputs, ("S", "S"), order=2, extrapolate=False
        )
        extrapolated = differentiate(
            lambda i: price(i, CALL), inputs, ("S", "S"), order=2, extrapolate=True
        )
        assert abs(extrapolated.value - analytic) <= abs(plain.value - analytic)

    def test_error_estimate_is_meaningful(self):
        """The self-reported error should bracket the actual error."""
        inputs = BSMInputs(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.3)
        from qar.greeks.second import gamma

        result = differentiate(lambda i: price(i, CALL), inputs, ("S", "S"), order=2)
        actual_error = abs(result.value - float(gamma(inputs)))
        assert actual_error <= max(10 * result.error_estimate, 1e-12)


def test_r_bump_respects_the_carry_coupling():
    """Bumping r must move b with it when the carry tracks the rate.

    Getting this wrong would make the validator 'confirm' the wrong Rho, which
    is the subtlest way this whole exercise could produce a confident lie.
    """
    from qar.validate.fd import bump

    tracking = BSMInputs(S=100, K=100, T=1, r=0.05, sigma=0.3, b=0.02,
                         carry_depends_on_r=True)
    bumped = bump(tracking, "r", 0.01)
    assert bumped.r == pytest.approx(0.06)
    assert bumped.b == pytest.approx(0.03)   # carry followed

    pinned = tracking.replace(carry_depends_on_r=False)
    bumped = bump(pinned, "r", 0.01)
    assert bumped.r == pytest.approx(0.06)
    assert bumped.b == pytest.approx(0.02)   # carry stayed put


def test_q_bump_moves_carry_the_other_way():
    """b = r - q, so raising q lowers b."""
    from qar.validate.fd import bump

    inputs = BSMInputs(S=100, K=100, T=1, r=0.05, sigma=0.3, b=0.05)
    assert bump(inputs, "q", 0.01).b == pytest.approx(0.04)


def test_grid_covers_the_regimes_it_claims_to():
    grid = default_grid()
    assert any(i.b == i.r for i in grid)                   # BSM
    assert any(i.b != i.r and i.b != 0 for i in grid)      # dividends
    assert any(i.b == 0 for i in grid)                     # futures
    assert any(i.r < 0 for i in grid)                      # negative rates
    assert any(not i.carry_depends_on_r for i in grid)
    assert len({float(i.T) for i in grid}) >= 4
