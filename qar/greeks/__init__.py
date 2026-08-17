"""All 17 analytically derived Greeks, plus a registry for iterating over them.

The registry is what lets the validation harness sweep every Greek without
naming them one at a time, and what keeps a newly added Greek from silently
escaping the test suite.
"""

from typing import Callable, NamedTuple

from qar.greeks.first import delta, dual_delta, epsilon, rho, theta, vega
from qar.greeks.second import (
    charm,
    dual_gamma,
    gamma,
    vanna,
    vera,
    veta,
    volga,
    vomma,
)
from qar.greeks.third import color, speed, ultima, zomma

__all__ = [
    "delta", "vega", "theta", "rho", "epsilon", "dual_delta",
    "gamma", "vanna", "volga", "vomma", "charm", "veta", "vera", "dual_gamma",
    "speed", "zomma", "color", "ultima",
    "GreekSpec", "REGISTRY", "by_order",
]


class GreekSpec(NamedTuple):
    """How to differentiate the price to reproduce a given Greek numerically.

    Attributes
    ----------
    name:
        Display name.
    func:
        The analytical implementation, ``f(inputs, kind) -> float``.
    order:
        Total order of the derivative (1, 2 or 3). Sets the finite-difference
        tolerance, since achievable accuracy degrades sharply with order.
    wrt:
        Which :class:`~qar.core.bsm.BSMInputs` fields to differentiate against,
        in order and with repetition. ``("S", "S")`` means twice in spot.
    sign:
        Multiplier applied to the numerical derivative before comparison. This
        is ``-1`` exactly for the calendar-time Greeks (Theta, Charm, Veta,
        Color), because differentiating the code gives :math:`\\partial/\\partial
        T` while the convention reports :math:`\\partial/\\partial t`.
    call_put_invariant:
        True when the Greek must be identical for calls and puts. Asserted
        directly as a structural test.
    """

    name: str
    func: Callable
    order: int
    wrt: tuple[str, ...]
    sign: float = 1.0
    call_put_invariant: bool = False


#: Every Greek, with the derivative that defines it. Order matters only for
#: display; the harness iterates the whole list.
REGISTRY: tuple[GreekSpec, ...] = (
    # -- first order -------------------------------------------------------
    GreekSpec("delta", delta, 1, ("S",)),
    GreekSpec("vega", vega, 1, ("sigma",), call_put_invariant=True),
    GreekSpec("theta", theta, 1, ("T",), sign=-1.0),
    GreekSpec("rho", rho, 1, ("r",)),
    GreekSpec("epsilon", epsilon, 1, ("q",)),
    GreekSpec("dual_delta", dual_delta, 1, ("K",)),
    # -- second order ------------------------------------------------------
    GreekSpec("gamma", gamma, 2, ("S", "S"), call_put_invariant=True),
    GreekSpec("vanna", vanna, 2, ("S", "sigma"), call_put_invariant=True),
    GreekSpec("volga", volga, 2, ("sigma", "sigma"), call_put_invariant=True),
    GreekSpec("charm", charm, 2, ("S", "T"), sign=-1.0),
    GreekSpec("veta", veta, 2, ("sigma", "T"), sign=-1.0, call_put_invariant=True),
    GreekSpec("vera", vera, 2, ("sigma", "r"), call_put_invariant=True),
    GreekSpec("dual_gamma", dual_gamma, 2, ("K", "K"), call_put_invariant=True),
    # -- third order -------------------------------------------------------
    GreekSpec("speed", speed, 3, ("S", "S", "S"), call_put_invariant=True),
    GreekSpec("zomma", zomma, 3, ("S", "S", "sigma"), call_put_invariant=True),
    GreekSpec("color", color, 3, ("S", "S", "T"), sign=-1.0, call_put_invariant=True),
    GreekSpec("ultima", ultima, 3, ("sigma", "sigma", "sigma"), call_put_invariant=True),
)


def by_order(order: int) -> tuple[GreekSpec, ...]:
    """All registered Greeks of a given derivative order."""
    return tuple(spec for spec in REGISTRY if spec.order == order)
