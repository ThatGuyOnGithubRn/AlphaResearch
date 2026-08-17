"""The normal distribution layer, checked against exact and published values."""

import math

import numpy as np
import pytest

from qar.core.distributions import norm_cdf, norm_pdf, norm_ppf

# Quantiles every statistician knows by heart, to full double precision.
KNOWN_QUANTILES = [
    (0.5, 0.0),
    (0.75, 0.6744897501960817),
    (0.90, 1.2815515655446004),
    (0.95, 1.6448536269514722),
    (0.975, 1.959963984540054),
    (0.99, 2.3263478740408408),
    (0.995, 2.5758293035489004),
    (0.999, 3.090232306167813),
]


class TestNormPdf:
    def test_peak_value(self):
        assert norm_pdf(0.0) == pytest.approx(1.0 / math.sqrt(2 * math.pi), rel=1e-15)

    def test_symmetry(self):
        for x in (0.3, 1.0, 2.5, 7.0):
            assert norm_pdf(x) == pytest.approx(norm_pdf(-x), rel=1e-15)

    def test_integrates_to_one(self):
        """Trapezoidal integration over +-12 sigma recovers unit mass."""
        grid = np.linspace(-12, 12, 200_001)
        assert np.trapezoid(norm_pdf(grid), grid) == pytest.approx(1.0, abs=1e-12)


class TestNormCdf:
    def test_median(self):
        assert norm_cdf(0.0) == pytest.approx(0.5, rel=1e-16)

    def test_reflection_identity(self):
        """N(-x) = 1 - N(x), the identity the unified call/put forms rest on."""
        for x in (0.1, 0.7, 1.9, 4.0):
            assert norm_cdf(-x) == pytest.approx(1.0 - norm_cdf(x), abs=1e-16)

    def test_known_values(self):
        for probability, quantile in KNOWN_QUANTILES:
            assert norm_cdf(quantile) == pytest.approx(probability, abs=1e-15)

    def test_tails_saturate_without_nan(self):
        assert norm_cdf(-40.0) == pytest.approx(0.0, abs=1e-300)
        assert norm_cdf(40.0) == pytest.approx(1.0, abs=1e-15)
        assert norm_cdf(np.inf) == 1.0
        assert norm_cdf(-np.inf) == 0.0

    def test_accepts_complex(self):
        """Required for complex-step differentiation to work at all."""
        value = norm_cdf(complex(0.5, 1e-200))
        assert isinstance(value, complex)
        assert value.real == pytest.approx(norm_cdf(0.5), rel=1e-15)
        # d/dx N(x) = n(x), recovered from the imaginary part.
        assert value.imag / 1e-200 == pytest.approx(norm_pdf(0.5), rel=1e-14)

    def test_vectorises(self):
        grid = np.array([-2.0, 0.0, 1.5])
        assert np.allclose(norm_cdf(grid), [norm_cdf(x) for x in grid], rtol=1e-15)


class TestNormPpf:
    def test_known_quantiles_to_machine_precision(self):
        """Acklam plus one Halley step must reach full double precision."""
        for probability, quantile in KNOWN_QUANTILES:
            assert norm_ppf(probability) == pytest.approx(quantile, rel=1e-14)

    def test_round_trip(self):
        for probability in (1e-8, 0.01, 0.3, 0.5, 0.87, 0.999999):
            assert norm_cdf(norm_ppf(probability)) == pytest.approx(probability, rel=1e-12)

    def test_refinement_beats_raw_approximation(self):
        """Confirm the Halley step is doing real work, not decoration."""
        from qar.core.distributions import _ppf_acklam

        worst_raw = max(
            abs(_ppf_acklam(p) - q) / abs(q) for p, q in KNOWN_QUANTILES if q != 0
        )
        worst_refined = max(
            abs(norm_ppf(p) - q) / abs(q) for p, q in KNOWN_QUANTILES if q != 0
        )
        assert worst_raw > 1e-10        # the approximation alone is ~1e-9
        assert worst_refined < 1e-14    # the refinement lands at machine precision

    def test_boundaries_and_domain(self):
        assert norm_ppf(0.0) == -math.inf
        assert norm_ppf(1.0) == math.inf
        assert math.isnan(norm_ppf(-0.1))
        assert math.isnan(norm_ppf(1.1))
