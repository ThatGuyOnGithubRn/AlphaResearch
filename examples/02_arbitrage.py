"""Plant mispricings in a fair option chain and print the trades that capture them.

Run:  .venv/bin/python examples/02_arbitrage.py
"""

from __future__ import annotations

import numpy as np

from qar.arb import Quote, check_parity, imply_forward, scan
from qar.core import CALL, PUT, BSMInputs, price

SPOT, RATE, DIVIDEND, MATURITY, VOL = 100.0, 0.05, 0.02, 0.5, 0.28
CARRY = RATE - DIVIDEND


def fair(strike: float, kind=CALL, maturity: float = MATURITY) -> float:
    return float(
        price(
            BSMInputs(S=SPOT, K=strike, T=maturity, r=RATE, sigma=VOL, b=CARRY), kind
        )
    )


def banner(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def parity_demo() -> None:
    banner("1. PUT-CALL PARITY")
    strike = 100.0
    call, put = fair(strike, CALL), fair(strike, PUT)

    clean = check_parity(call, put, SPOT, strike, MATURITY, RATE, CARRY)
    print(f"Fair quotes:  call={call:.6f}  put={put:.6f}")
    print(f"Parity holds: {bool(clean)}   residual = {clean.residual:.2e}")
    print(f"Implied forward from the pair: {imply_forward(call, put, strike, MATURITY, RATE):.6f}")
    print(f"Actual forward S*exp(b*T):     {SPOT * np.exp(CARRY * MATURITY):.6f}")

    print("\nNow mark the put 0.18 too cheap:\n")
    violated = check_parity(call, put - 0.18, SPOT, strike, MATURITY, RATE, CARRY, tol=1e-6)
    print(violated.violation.report())


def bounds_demo() -> None:
    banner("2. STATIC BOUNDS — a call marked below its intrinsic forward")
    strike = 80.0
    quote = Quote(
        price=fair(strike) * 0.6, S=SPOT, K=strike, T=MATURITY, r=RATE,
        kind="call", b=CARRY,
    )
    print(f"Fair value {fair(strike):.6f}, marked at {quote.price:.6f}")
    print(f"Lower bound max(F_d - K_d, 0) = "
          f"{max(quote.discounted_forward - quote.discounted_strike, 0):.6f}\n")

    from qar.arb import check_bounds

    for violation in check_bounds(quote):
        print(violation.report())


def butterfly_demo() -> None:
    banner("3. BUTTERFLY CONVEXITY — a negative implied density")
    strikes = (90.0, 100.0, 110.0)
    quotes = [
        Quote(price=fair(k), S=SPOT, K=k, T=MATURITY, r=RATE, kind="call", b=CARRY)
        for k in strikes
    ]

    print(f"Fair chain: " + "  ".join(f"K={q.K:g}:{q.price:.4f}" for q in quotes))
    print(f"Violations in the fair chain: {len(scan(quotes))}")

    tampered = list(quotes)
    tampered[1] = Quote(
        price=quotes[1].price + 2.0, S=SPOT, K=100.0, T=MATURITY, r=RATE,
        kind="call", b=CARRY,
    )
    print(f"\nMark the 100 strike 2.00 too rich -> "
          f"{len(scan(tampered))} violation(s):\n")
    for violation in scan(tampered):
        print(violation.report())
        print()


def verify_all() -> None:
    banner("4. VERIFICATION — every trade swept across terminal prices")
    strike = 100.0
    result = check_parity(
        fair(strike, CALL), fair(strike, PUT) - 0.18,
        SPOT, strike, MATURITY, RATE, CARRY, tol=1e-6,
    )
    trade = result.violation.trade

    print(f"{'terminal spot':>16}{'cashflow at T':>18}")
    print("-" * 34)
    for terminal in (0.01, 50.0, 90.0, 100.0, 110.0, 200.0, 500.0):
        print(f"{terminal:>16.2f}{trade.cashflow_at_expiry(terminal):>18.10f}")

    print(
        f"\nWorst case across 2000 terminal prices: "
        f"{trade.worst_case():.2e}"
    )
    print(
        f"Cash banked at inception:               {trade.cashflow_now:+.6f}\n\n"
        "Zero at expiry for every terminal price, positive today. That is what\n"
        "makes it an arbitrage rather than a position."
    )


if __name__ == "__main__":
    parity_demo()
    bounds_demo()
    butterfly_demo()
    verify_all()
