"""Arbitrage detection, and proof that every trade it emits is actually riskless.

The strongest test in this file is :meth:`Trade.worst_case`: each leg knows its
own payoff as a function of terminal spot, so a violation's trade can be swept
across a wide range of terminal prices and checked for a non-negative outcome.
That is what distinguishes a real static arbitrage from a plausible-looking
portfolio, and it caught a genuine bug during development — a conversion under a
dividend yield that was silently short a fraction of a share.
"""

import math

import numpy as np
import pytest

from qar.arb import (
    Quote,
    check_bounds,
    check_butterfly,
    check_calendar,
    check_parity,
    check_strike_monotonicity,
    imply_call,
    imply_forward,
    imply_put,
    parity_residual,
    scan,
)
from qar.core.bsm import CALL, PUT, BSMInputs, price

SPOT, RATE, VOL = 100.0, 0.05, 0.25
SWEEP = np.linspace(1e-6, 400.0, 4001)


def fair(K, T=0.5, kind=CALL, b=None, sigma=VOL):
    inputs = BSMInputs(S=SPOT, K=K, T=T, r=RATE, sigma=sigma, b=b)
    return float(price(inputs, kind))


def chain(strikes, T=0.5, kind="call", b=None):
    kind_enum = CALL if kind == "call" else PUT
    return [
        Quote(price=fair(k, T, kind_enum, b), S=SPOT, K=k, T=T, r=RATE, kind=kind, b=b)
        for k in strikes
    ]


class TestParity:
    @pytest.mark.parametrize("b", [None, 0.01, 0.0, -0.02])
    @pytest.mark.parametrize("K", [80.0, 100.0, 130.0])
    def test_model_prices_satisfy_parity(self, b, K):
        result = check_parity(fair(K, kind=CALL, b=b), fair(K, kind=PUT, b=b),
                              SPOT, K, 0.5, RATE, b)
        assert result
        assert result.residual == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize("b", [None, 0.01, -0.02])
    @pytest.mark.parametrize("mispricing", [0.18, -0.25, 1.5])
    def test_violation_trade_is_riskless_and_profitable(self, b, mispricing):
        """The core claim: cash in today, exactly nothing owed at expiry."""
        K = 100.0
        result = check_parity(
            fair(K, kind=CALL, b=b), fair(K, kind=PUT, b=b) - mispricing,
            SPOT, K, 0.5, RATE, b, tol=1e-6,
        )
        assert result.violated
        violation = result.violation

        assert violation.profit == pytest.approx(abs(mispricing), rel=1e-9)
        # Parity locks exactly: zero at expiry for every terminal price.
        payoffs = [violation.trade.cashflow_at_expiry(float(s)) for s in SWEEP]
        assert max(abs(p) for p in payoffs) < 1e-8

    def test_no_false_positive_within_tolerance(self):
        K = 100.0
        result = check_parity(fair(K, kind=CALL), fair(K, kind=PUT) + 1e-9,
                              SPOT, K, 0.5, RATE, tol=1e-6)
        assert result
        assert result.violation is None

    def test_residual_sign_identifies_the_rich_leg(self):
        K = 100.0
        rich_call = check_parity(fair(K, kind=CALL) + 0.3, fair(K, kind=PUT),
                                 SPOT, K, 0.5, RATE, tol=1e-6)
        assert rich_call.residual > 0
        assert "Conversion" in rich_call.violation.trade.rationale

        rich_put = check_parity(fair(K, kind=CALL), fair(K, kind=PUT) + 0.3,
                                SPOT, K, 0.5, RATE, tol=1e-6)
        assert rich_put.residual < 0
        assert "Reversal" in rich_put.violation.trade.rationale


class TestImpliedLegs:
    @pytest.mark.parametrize("b", [None, 0.01, -0.03])
    def test_round_trip(self, b):
        K, T = 110.0, 0.75
        call, put = fair(K, T, CALL, b), fair(K, T, PUT, b)
        assert imply_call(put, SPOT, K, T, RATE, b) == pytest.approx(call, rel=1e-12)
        assert imply_put(call, SPOT, K, T, RATE, b) == pytest.approx(put, rel=1e-12)

    def test_implied_forward_recovers_the_carry(self):
        """Backing the forward out of a call/put pair, with no carry assumption."""
        K, T, b = 100.0, 1.5, 0.015
        call, put = fair(K, T, CALL, b), fair(K, T, PUT, b)
        assert imply_forward(call, put, K, T, RATE) == pytest.approx(
            SPOT * math.exp(b * T), rel=1e-10
        )


class TestBounds:
    @pytest.mark.parametrize("K", [70.0, 100.0, 140.0])
    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_fair_prices_are_inside_bounds(self, K, kind):
        quote = chain([K], kind=kind)[0]
        assert check_bounds(quote) == []

    def test_below_intrinsic_call_is_detected(self):
        quote = chain([70.0])[0]
        cheap = Quote(price=quote.price * 0.5, S=SPOT, K=70.0, T=0.5, r=RATE, kind="call")
        violations = check_bounds(cheap)
        assert len(violations) == 1
        assert violations[0].profit > 0
        assert violations[0].trade.worst_case(SWEEP) >= -1e-9

    def test_call_above_underlying_is_detected(self):
        rich = Quote(price=SPOT * 1.2, S=SPOT, K=100.0, T=0.5, r=RATE, kind="call")
        violations = check_bounds(rich)
        assert len(violations) == 1
        assert violations[0].profit > 0
        assert violations[0].trade.worst_case(SWEEP) >= -1e-9

    def test_put_above_discounted_strike_is_detected(self):
        rich = Quote(price=100.0, S=SPOT, K=90.0, T=0.5, r=RATE, kind="put")
        violations = check_bounds(rich)
        assert violations
        assert violations[0].trade.worst_case(SWEEP) >= -1e-9


class TestStrikeStructure:
    def test_fair_chain_is_clean(self):
        assert scan(chain([80.0, 90.0, 100.0, 110.0, 120.0])) == []

    def test_fair_put_chain_is_clean(self):
        assert scan(chain([80.0, 90.0, 100.0, 110.0, 120.0], kind="put")) == []

    def test_monotonicity_breach_detected(self):
        low, high = chain([90.0, 100.0])
        high = Quote(price=low.price + 0.4, S=SPOT, K=100.0, T=0.5, r=RATE, kind="call")
        violation = check_strike_monotonicity(low, high)
        assert violation is not None
        assert violation.profit > 0
        assert violation.trade.worst_case(SWEEP) >= -1e-9

    def test_butterfly_breach_detected(self):
        low, mid, high = chain([90.0, 100.0, 110.0])
        # The fair butterfly costs ~1.05 here, so the middle strike has to be
        # marked up by more than that before convexity actually breaks.
        mid = Quote(price=mid.price + 2.0, S=SPOT, K=100.0, T=0.5, r=RATE, kind="call")
        violation = check_butterfly(low, mid, high)
        assert violation is not None
        assert violation.profit > 0
        assert violation.trade.worst_case(SWEEP) >= -1e-9

    def test_butterfly_handles_unequal_spacing(self):
        """Weights must adapt, or evenly-priced uneven strikes look arbitrageable."""
        assert check_butterfly(*chain([80.0, 100.0, 150.0])) is None

    def test_butterfly_requires_increasing_strikes(self):
        low, mid, high = chain([90.0, 100.0, 110.0])
        with pytest.raises(ValueError):
            check_butterfly(high, mid, low)


class TestCalendar:
    def test_fair_chain_is_clean(self):
        quotes = [chain([100.0], T=t)[0] for t in (0.25, 0.5, 1.0, 2.0)]
        assert [v for v in scan(quotes) if v.kind.name == "CALENDAR"] == []

    def test_breach_detected(self):
        far = chain([100.0], T=1.0)[0]
        near = Quote(price=far.price + 1.0, S=SPOT, K=100.0, T=0.25, r=RATE, kind="call")
        violation = check_calendar(near, far)
        assert violation is not None
        assert violation.profit > 0

    def test_futures_carry_loosens_the_bound(self):
        """At b = 0 the near call may exceed the far one by the carry factor.

        Applying the textbook b = r rule here would flag a violation that is
        not there.
        """
        far = chain([100.0], T=1.0, b=0.0)[0]
        scale = __import__("math").exp(RATE * 0.75)
        just_inside = Quote(price=far.price * scale * 0.999, S=SPOT, K=100.0,
                            T=0.25, r=RATE, kind="call", b=0.0)
        just_outside = Quote(price=far.price * scale * 1.001, S=SPOT, K=100.0,
                             T=0.25, r=RATE, kind="call", b=0.0)
        assert check_calendar(just_inside, far) is None
        assert check_calendar(just_outside, far) is not None

    def test_negative_carry_is_declined_rather_than_guessed(self):
        """Under negative carry no same-strike calendar test is valid.

        A long-dated call on a heavily backwardated underlying is genuinely
        allowed to be cheaper, and the dominance argument that justifies the
        b >= 0 bound reverses. Returning None is the honest answer; the naive
        rule would report a phantom arbitrage here.
        """
        b = -0.30
        quotes = [chain([100.0], T=t, b=b)[0] for t in (0.25, 1.0)]
        assert quotes[1].price < quotes[0].price  # confirm the situation is real
        assert check_calendar(quotes[0], quotes[1]) is None

    def test_requires_matching_strikes(self):
        near, far = chain([100.0], T=0.25)[0], chain([110.0], T=1.0)[0]
        with pytest.raises(ValueError):
            check_calendar(near, far)


class TestScan:
    def test_finds_every_planted_violation(self):
        quotes = chain([90.0, 100.0, 110.0])
        tampered = list(quotes)
        tampered[1] = Quote(price=quotes[1].price + 1.2, S=SPOT, K=100.0,
                            T=0.5, r=RATE, kind="call")
        violations = scan(tampered)
        assert violations
        assert all(v.profit > 0 for v in violations)
        assert all(v.trade.worst_case(SWEEP) >= -1e-9 for v in violations)

    def test_every_violation_reports_cleanly(self):
        quotes = chain([90.0, 100.0, 110.0])
        tampered = list(quotes)
        tampered[1] = Quote(price=quotes[1].price + 1.2, S=SPOT, K=100.0,
                            T=0.5, r=RATE, kind="call")
        for violation in scan(tampered):
            text = violation.report()
            assert "cashflow now" in text
            assert "cashflow at T" in text
            assert violation.trade.legs


def test_parity_residual_matches_check_parity():
    K = 105.0
    call, put = fair(K, kind=CALL), fair(K, kind=PUT)
    assert parity_residual(call, put + 0.3, SPOT, K, 0.5, RATE) == pytest.approx(
        check_parity(call, put + 0.3, SPOT, K, 0.5, RATE, tol=1e-6).residual
    )


def test_implied_vol_rejects_arbitrageable_quotes():
    """The implied-vol solver and the arbitrage checker must agree on what is impossible."""
    from qar.core.iv import NoImpliedVolError, implied_vol

    below_intrinsic = 0.5
    with pytest.raises(NoImpliedVolError):
        implied_vol(below_intrinsic, S=SPOT, K=70.0, T=0.5, r=RATE, kind=CALL)

    quote = Quote(price=below_intrinsic, S=SPOT, K=70.0, T=0.5, r=RATE, kind="call")
    assert check_bounds(quote)
