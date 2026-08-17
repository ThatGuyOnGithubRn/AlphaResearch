"""Model-free arbitrage detection, with the trades that monetise each violation."""

from qar.arb.bounds import (
    Quote,
    check_bounds,
    check_butterfly,
    check_calendar,
    check_strike_monotonicity,
    scan,
)
from qar.arb.parity import (
    ParityResult,
    check_parity,
    imply_call,
    imply_forward,
    imply_put,
    parity_residual,
)
from qar.arb.violation import ArbitrageViolation, Leg, Trade, ViolationKind

__all__ = [
    "Quote",
    "check_bounds",
    "check_butterfly",
    "check_calendar",
    "check_strike_monotonicity",
    "scan",
    "ParityResult",
    "check_parity",
    "imply_call",
    "imply_forward",
    "imply_put",
    "parity_residual",
    "ArbitrageViolation",
    "Leg",
    "Trade",
    "ViolationKind",
]
