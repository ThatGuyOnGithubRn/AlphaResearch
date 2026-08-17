r"""Sweep every Greek against its numerical derivative over a parameter grid.

Checking one Greek at one point proves almost nothing — sign errors and missing
terms routinely vanish at the money, at :math:`b = r`, or at one particular
maturity. The grid below is chosen so that every term in every formula is
exercised somewhere:

*   **Moneyness** deep ITM / ATM / deep OTM, so terms weighted by
    :math:`N(d_1)` and by :math:`n(d_1)` are separately visible.
*   **Maturity** one week to two years, which separates the :math:`1/T` terms
    in Charm, Veta and Color from the rest.
*   **Carry** :math:`b = r` (BSM), :math:`b = r - q` (dividends), :math:`b = 0`
    (futures). A wrong carry term is invisible at :math:`b = r`, where it
    multiplies zero — the single most common way a hand-derived Greek library
    is wrong and passes its own tests.
*   **Rates** including a negative rate, which flushes out sign assumptions.

Tolerances come from the finite-difference error analysis in
:mod:`qar.validate.fd`, with a safety factor, not from tuning until green.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from qar.core.bsm import CALL, PUT, BSMInputs, price
from qar.greeks import REGISTRY, GreekSpec
from qar.validate.complex_step import complex_derivative
from qar.validate.fd import differentiate, step_for

__all__ = [
    "ValidationRow",
    "default_grid",
    "validate_greek",
    "sweep",
    "format_report",
    "RTOL_BY_ORDER",
]

#: Relative tolerance per derivative order, from the eps**(2/(k+2)) error
#: analysis in :mod:`qar.validate.fd` with a safety factor. This is the *floor*
#: on what counts as agreement; where the numerical method knows it is doing
#: worse than this, its own error estimate takes over instead.
RTOL_BY_ORDER = {1: 1e-7, 2: 1e-5, 3: 1e-3}

#: Multiple of the finite-difference method's self-reported error that the
#: analytical value is allowed to deviate by. The comparison is only ever as
#: sharp as the numerical reference is trustworthy; demanding better would be
#: testing the differencing scheme, not the formula. 10x leaves room for the
#: estimate itself being approximate without letting a genuinely wrong term —
#: which is wrong by a factor, not by a few multiples of the noise floor — slip
#: through.
FD_ERROR_SAFETY = 10.0

#: Below this multiple of the price's own round-off, a Greek is unresolvable in
#: double precision and the comparison carries no information. Reached by deep
#: out-of-the-money short-dated options, where the true value is legitimately
#: ~1e-140 and the numerical derivative simply underflows to zero.
NEGLIGIBLE_ULPS = 100.0


@dataclass(frozen=True)
class ValidationRow:
    """One (Greek, option kind, parameter point) comparison."""

    greek: str
    kind: str
    order: int
    inputs: BSMInputs
    analytic: float
    numerical: float
    abs_error: float
    rel_error: float
    tolerance: float
    passed: bool
    method: str
    negligible: bool = False

    @property
    def label(self) -> str:
        i = self.inputs
        return (
            f"S={float(i.S):g} K={float(i.K):g} T={float(i.T):g} "
            f"r={float(i.r):g} sig={float(i.sigma):g} b={float(i.b):g}"
        )


def default_grid() -> tuple[BSMInputs, ...]:
    """Parameter points covering the regimes described in the module docstring."""
    points: list[BSMInputs] = []
    spot = 100.0
    for strike in (70.0, 100.0, 140.0):          # deep ITM, ATM, deep OTM
        for maturity in (7 / 365, 0.25, 1.0, 2.0):
            for rate, carry, tracks in (
                (0.05, 0.05, True),               # BSM: b = r
                (0.05, 0.02, True),               # dividend yield q = 3%
                (0.05, 0.00, False),              # Black-76 futures
                (-0.01, -0.01, True),             # negative rates
            ):
                for vol in (0.10, 0.35, 0.80):
                    points.append(
                        BSMInputs(
                            S=spot, K=strike, T=maturity, r=rate,
                            sigma=vol, b=carry, carry_depends_on_r=tracks,
                        )
                    )
    return tuple(points)


def _resolution_floor(spec: GreekSpec, inputs: BSMInputs) -> float:
    r"""Noise floor of the numerical derivative — below this, nothing is measurable.

    The price is accumulated from terms of order :math:`S`, so it carries an
    absolute round-off of roughly :math:`\varepsilon S`. What happens to that
    round-off next depends entirely on the method:

    **Finite differences** divide by :math:`h^k`, and :math:`h` is small, so the
    round-off is *amplified*:

    .. math:: \text{noise}_{\text{FD}} \approx \frac{\varepsilon S}{\prod_i h_i}

    For Gamma on a 100-dollar underlying, :math:`h \approx 10^{-2}` gives a
    noise floor near :math:`10^{-10}` — eleven orders of magnitude above the
    price's own round-off. Using the variable's *scale* here instead of the
    actual step (the obvious mistake) understates the floor by
    :math:`(\text{scale}/h)^k`, which for a second derivative is a factor of
    ~:math:`10^7`.

    **Complex step** performs no subtraction, so nothing is amplified and the
    floor is just the price's round-off divided by the variable's scale.

    A Greek below this floor cannot be confirmed numerically by any amount of
    care — there are no bits left to confirm it with. Deep out-of-the-money
    short-dated options land here routinely.
    """
    scales = {
        "S": float(np.asarray(inputs.S)),
        "K": float(np.asarray(inputs.K)),
        "T": max(float(np.asarray(inputs.T)), 1e-3),
        "sigma": float(np.asarray(inputs.sigma)),
        "r": max(abs(float(np.asarray(inputs.r))), 0.01),
        "q": max(abs(float(np.asarray(inputs.r - inputs.b))), 0.01),
        "b": max(abs(float(np.asarray(inputs.b))), 0.01),
    }
    floor = float(np.finfo(float).eps) * abs(float(np.asarray(inputs.S)))

    if spec.order == 1:
        return floor / scales[spec.wrt[0]]

    for var in spec.wrt:
        floor /= step_for(inputs, var, spec.order)
    return floor


def validate_greek(spec: GreekSpec, inputs: BSMInputs, kind: Any = CALL) -> ValidationRow:
    """Compare one analytical Greek against its numerical counterpart.

    First-order Greeks go through complex step, which is exact to machine
    precision, so they are held to a flat relative tolerance. Second and third
    order use Richardson-extrapolated finite differences and are held to
    whichever is looser: that flat tolerance, or the differencing scheme's own
    estimate of how far off it is. See :data:`FD_ERROR_SAFETY`.

    Either way, a check is skipped as vacuous when the quantity falls below the
    resolution floor of double precision — see :func:`_resolution_floor`.
    """
    analytic = float(np.asarray(spec.func(inputs, kind)))

    if spec.order == 1:
        numerical = spec.sign * complex_derivative(
            lambda i: price(i, kind), inputs, spec.wrt[0]
        )
        method = "complex-step"
        method_error = 0.0
    else:
        result = differentiate(
            lambda i: price(i, kind), inputs, spec.wrt, order=spec.order
        )
        numerical = spec.sign * result.value
        method = "richardson-fd"
        method_error = result.error_estimate

    abs_error = abs(analytic - numerical)
    denominator = max(abs(analytic), abs(numerical))
    rel_error = abs_error / denominator if denominator > 0 else 0.0

    floor = NEGLIGIBLE_ULPS * _resolution_floor(spec, inputs)
    negligible = denominator < floor

    tolerance = max(
        RTOL_BY_ORDER[spec.order] * abs(analytic),
        FD_ERROR_SAFETY * method_error,
        floor,
    )
    passed = negligible or abs_error <= tolerance

    return ValidationRow(
        greek=spec.name,
        kind=str(kind.value if hasattr(kind, "value") else kind),
        order=spec.order,
        inputs=inputs,
        analytic=analytic,
        numerical=numerical,
        abs_error=abs_error,
        rel_error=rel_error,
        tolerance=tolerance,
        passed=passed,
        method=method,
        negligible=negligible,
    )


def sweep(
    grid: Sequence[BSMInputs] | None = None,
    specs: Iterable[GreekSpec] = REGISTRY,
    kinds: Sequence[Any] = (CALL, PUT),
) -> list[ValidationRow]:
    """Validate every Greek at every grid point for both option kinds."""
    points = default_grid() if grid is None else grid
    return [
        validate_greek(spec, inputs, kind)
        for spec in specs
        for inputs in points
        for kind in kinds
    ]


def format_report(rows: Sequence[ValidationRow]) -> str:
    """Human-readable summary, worst relative error per Greek."""
    by_greek: dict[str, list[ValidationRow]] = {}
    for row in rows:
        by_greek.setdefault(row.greek, []).append(row)

    lines = [
        f"{'greek':<12}{'ord':>4}{'method':>15}{'checked':>9}{'skipped':>9}"
        f"{'worst rel':>12}  status",
        "-" * 78,
    ]
    for spec in REGISTRY:
        group = by_greek.get(spec.name)
        if not group:
            continue
        resolved = [r for r in group if not r.negligible]
        skipped = len(group) - len(resolved)
        worst = max(resolved, key=lambda r: r.rel_error) if resolved else None
        failures = [r for r in group if not r.passed]
        status = "PASS" if not failures else f"FAIL ({len(failures)})"
        worst_text = f"{worst.rel_error:>12.2e}" if worst else f"{'-':>12}"
        lines.append(
            f"{spec.name:<12}{spec.order:>4}{group[0].method:>15}"
            f"{len(resolved):>9}{skipped:>9}{worst_text}  {status}"
        )

    failures = [r for r in rows if not r.passed]
    skipped_total = sum(1 for r in rows if r.negligible)
    lines.append("-" * 78)
    lines.append(
        f"{len(rows) - len(failures)}/{len(rows)} checks passed "
        f"across {len({id(r.inputs) for r in rows})} parameter points "
        f"({skipped_total} below double-precision resolution, not counted)"
    )
    for row in failures[:10]:
        lines.append(
            f"  FAIL {row.greek} ({row.kind}) {row.label}: "
            f"analytic={row.analytic:.10g} numerical={row.numerical:.10g} "
            f"rel={row.rel_error:.3e}"
        )
    if len(failures) > 10:
        lines.append(f"  ... and {len(failures) - 10} more")
    return "\n".join(lines)
