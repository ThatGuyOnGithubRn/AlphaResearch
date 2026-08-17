"""Pull the live BTC option chain and run the arbitrage checks against it.

This is where parts 1 and 2 meet real quotes. Three things are demonstrated:

1.  Our implied-vol solver reproduces Deribit's own mark IV.
2.  Put-call parity, run across the chain, backs out the market's implied
    forward — and on a crypto venue that is a funding-rate reading.
3.  The static no-arbitrage checks, run with a tolerance set from the actual
    bid-ask spread rather than from zero.

Run:  .venv/bin/python examples/03_deribit_chain.py
      (add --refresh to bypass the cache)
"""

from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np

from qar.arb import Quote, check_parity, imply_forward, scan
from qar.core import CALL, PUT, BSMInputs, NoImpliedVolError, implied_vol, price
from qar.data import DeribitClient

RATE = 0.0          # Deribit options are inverse-margined; USD funding sits in the forward
CARRY = 0.0         # Black-76 on the future


def main(refresh: bool = False) -> None:
    client = DeribitClient()
    print("Fetching BTC option chain from Deribit...")
    try:
        quotes = client.option_chain("BTC", refresh=refresh)
    except Exception as exc:
        print(f"could not reach Deribit ({exc}).")
        print("If you have run this before, the cached payload in data/raw/ is used.")
        return

    spot = quotes[0].underlying_price
    print(f"{len(quotes)} instruments, underlying {spot:,.2f}\n")

    # -- group by expiry -------------------------------------------------
    by_expiry: dict[int, list] = defaultdict(list)
    for quote in quotes:
        by_expiry[quote.expiry_timestamp_ms].append(quote)

    expiries = sorted(by_expiry)[:4]

    print("=" * 78)
    print("1. OUR IMPLIED VOL vs DERIBIT'S MARK IV")
    print("=" * 78)
    print(f"{'instrument':<26}{'mark':>12}{'deribit IV':>12}{'our IV':>12}{'diff':>12}")
    print("-" * 78)

    differences = []
    for expiry in expiries[:2]:
        chain = sorted(by_expiry[expiry], key=lambda q: abs(q.strike - spot))[:6]
        for quote in chain:
            if quote.time_to_expiry <= 0 or quote.mark_price <= 0:
                continue
            try:
                solved = implied_vol(
                    quote.mark_price,
                    S=quote.underlying_price,
                    K=quote.strike,
                    T=quote.time_to_expiry,
                    r=RATE,
                    b=CARRY,
                    kind=CALL if quote.kind == "call" else PUT,
                )
            except NoImpliedVolError as exc:
                print(f"{quote.instrument:<26}{quote.mark_price:>12.2f}  no solution: {exc}")
                continue

            difference = solved.sigma - quote.implied_vol
            differences.append(difference)
            print(
                f"{quote.instrument:<26}{quote.mark_price:>12.2f}"
                f"{quote.implied_vol:>12.4f}{solved.sigma:>12.4f}{difference:>+12.4f}"
            )

    if differences:
        print(
            f"\nmedian |difference| = {np.median(np.abs(differences)):.5f} vol points.\n"
            "Residual gaps come from Deribit marking against the future while we\n"
            "price off the index, and from their own smile interpolation."
        )

    # -- parity across the chain -----------------------------------------
    print("\n" + "=" * 78)
    print("2. PUT-CALL PARITY -> THE MARKET'S IMPLIED FORWARD")
    print("=" * 78)
    print(f"{'expiry (days)':>14}{'strike':>12}{'implied fwd':>14}{'basis vs spot':>16}")
    print("-" * 78)

    observed_bases: list[float] = []
    for expiry in expiries:
        chain = by_expiry[expiry]
        calls = {q.strike: q for q in chain if q.kind == "call"}
        puts = {q.strike: q for q in chain if q.kind == "put"}
        common = sorted(set(calls) & set(puts), key=lambda k: abs(k - spot))
        if not common:
            continue

        strike = common[0]
        call, put = calls[strike], puts[strike]
        if call.time_to_expiry <= 0:
            continue

        forward = imply_forward(call.mark_price, put.mark_price, strike, call.time_to_expiry, RATE)
        days = call.time_to_expiry * 365
        basis = forward / spot - 1.0
        observed_bases.append(basis)
        print(f"{days:>14.1f}{strike:>12,.0f}{forward:>14,.2f}{basis:>+15.2%}")

    if observed_bases:
        mean_basis = float(np.mean(observed_bases))
        shape = "contango" if mean_basis > 0 else "backwardation"
        direction = "above" if mean_basis > 0 else "below"
        print(
            f"\nMean basis {mean_basis:+.2%} — {shape}: the forward sits {direction}\n"
            "spot, so holders are being paid to be short (or charged to be long).\n"
            "Extracted with no funding or dividend assumption whatsoever; it falls\n"
            "straight out of parity, which is what makes it worth trusting."
        )

    # -- arbitrage scan --------------------------------------------------
    print("\n" + "=" * 78)
    print("3. STATIC NO-ARBITRAGE SCAN")
    print("=" * 78)

    for expiry in expiries[:3]:
        chain = [q for q in by_expiry[expiry] if q.kind == "call" and q.time_to_expiry > 0]
        if len(chain) < 3:
            continue
        maturity = chain[0].time_to_expiry

        spreads = [q.spread for q in chain if q.spread is not None]
        tolerance = float(np.median(spreads)) if spreads else 1.0

        arb_quotes = [
            Quote(price=q.mark_price, S=q.underlying_price, K=q.strike,
                  T=maturity, r=RATE, kind="call", b=CARRY)
            for q in sorted(chain, key=lambda q: q.strike)
        ]
        violations = scan(arb_quotes, tol=tolerance)

        print(
            f"\nexpiry in {maturity * 365:.1f} days: {len(arb_quotes)} strikes, "
            f"tolerance {tolerance:.2f} (median bid-ask)"
        )
        if not violations:
            print("  no violations — the chain is internally consistent")
        else:
            for violation in violations[:5]:
                print(f"  {violation}")
            if len(violations) > 5:
                print(f"  ... and {len(violations) - 5} more")

    print(
        "\nSetting the tolerance to the bid-ask spread is the point. At zero\n"
        "tolerance a live chain always shows 'violations' that no one can trade,\n"
        "because crossing the spread costs more than the edge."
    )


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
