"""All 17 Greeks across a moneyness/maturity grid, with the validation report.

Run:  .venv/bin/python examples/01_greeks_surface.py
"""

from __future__ import annotations

import numpy as np

from qar.core import CALL, PUT, BSMInputs, price
from qar.greeks import REGISTRY
from qar.validate import format_report, sweep


def print_chain() -> None:
    """Every Greek at three strikes, one maturity."""
    spot, maturity, rate, vol = 100.0, 0.5, 0.05, 0.30
    strikes = (85.0, 100.0, 115.0)

    print("=" * 78)
    print(f"GREEKS  S={spot:g}  T={maturity:g}y  r={rate:.0%}  sigma={vol:.0%}  (BSM, b=r)")
    print("=" * 78)

    header = f"{'greek':<12}{'ord':>4}" + "".join(f"{f'K={k:g}':>14}" for k in strikes)
    for kind, label in ((CALL, "CALL"), (PUT, "PUT")):
        print(f"\n{label}")
        print(header)
        print("-" * 78)
        for spec in REGISTRY:
            row = f"{spec.name:<12}{spec.order:>4}"
            for strike in strikes:
                inputs = BSMInputs(S=spot, K=strike, T=maturity, r=rate, sigma=vol)
                row += f"{float(spec.func(inputs, kind)):>14.6f}"
            print(row)

        print(f"{'price':<12}{'-':>4}", end="")
        for strike in strikes:
            inputs = BSMInputs(S=spot, K=strike, T=maturity, r=rate, sigma=vol)
            print(f"{float(price(inputs, kind)):>14.6f}", end="")
        print()


def print_carry_regimes() -> None:
    """The same option under four carry conventions."""
    spot, strike, maturity, rate, vol = 100.0, 100.0, 1.0, 0.05, 0.30
    regimes = [
        ("BSM (b=r)", rate, True),
        ("dividend q=3%", rate - 0.03, True),
        ("Black-76 futures", 0.0, False),
        ("FX, r_f=1%", rate - 0.01, True),
    ]

    print("\n" + "=" * 78)
    print("THE SAME OPTION UNDER FOUR CARRY CONVENTIONS")
    print("=" * 78)
    print(f"{'regime':<20}{'b':>8}{'call':>12}{'delta':>12}{'rho':>12}{'charm':>12}")
    print("-" * 78)

    from qar.greeks import charm, delta, rho

    for label, carry, tracks in regimes:
        inputs = BSMInputs(
            S=spot, K=strike, T=maturity, r=rate, sigma=vol,
            b=carry, carry_depends_on_r=tracks,
        )
        print(
            f"{label:<20}{carry:>8.3f}"
            f"{float(price(inputs, CALL)):>12.6f}"
            f"{float(delta(inputs, CALL)):>12.6f}"
            f"{float(rho(inputs, CALL)):>12.6f}"
            f"{float(charm(inputs, CALL)):>12.6f}"
        )
    print(
        "\nNote how Rho and Charm move with the carry convention while the price\n"
        "barely does. Both depend on whether b tracks r — the flag that\n"
        "qar.greeks.first.rho makes explicit rather than assuming."
    )


def print_validation() -> None:
    print("\n" + "=" * 78)
    print("NUMERICAL VALIDATION — every Greek against an independent derivative")
    print("=" * 78)
    print(format_report(sweep()))


def maybe_plot() -> None:
    """Plot Delta, Gamma, Vega and Theta across spot, if matplotlib is present."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed — skipping the plot)")
        return

    from qar.greeks import delta, gamma, theta, vega

    spots = np.linspace(50, 150, 400)
    maturities = (7 / 365, 0.25, 1.0)
    greeks = [("Delta", delta), ("Gamma", gamma), ("Vega", vega), ("Theta", theta)]

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, (name, function) in zip(axes.ravel(), greeks):
        for maturity in maturities:
            values = [
                float(function(BSMInputs(S=s, K=100.0, T=maturity, r=0.05, sigma=0.30), CALL))
                for s in spots
            ]
            axis.plot(spots, values, label=f"T={maturity:.2f}y")
        axis.set_title(name)
        axis.set_xlabel("spot")
        axis.axvline(100.0, color="grey", linewidth=0.6, linestyle="--")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)

    figure.suptitle("Call Greeks across spot, K=100, r=5%, sigma=30%")
    figure.tight_layout()
    figure.savefig("examples/greeks_surface.png", dpi=130)
    print("\nSaved examples/greeks_surface.png")


if __name__ == "__main__":
    print_chain()
    print_carry_regimes()
    print_validation()
    maybe_plot()
