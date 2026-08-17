r"""Complex-step differentiation — machine-precision Delta and Vega.

The headline validation. Finite differences are limited to ~1e-11 relative
accuracy no matter how carefully the step is chosen, because subtracting two
nearly equal numbers destroys significant digits. The complex-step method has
no subtraction at all.

Take :math:`f` analytic and expand along the imaginary axis:

.. math::
    f(x + ih) = f(x) + i h f'(x) - \frac{h^2}{2}f''(x)
                - \frac{i h^3}{6}f'''(x) + \dots

Take imaginary parts and divide by :math:`h`:

.. math::
    \frac{\operatorname{Im} f(x + ih)}{h} = f'(x) - \frac{h^2}{6}f'''(x) + O(h^4)

The truncation error is :math:`O(h^2)` — and crucially, **no cancellation
occurs**, because the derivative is recovered from the imaginary part rather
than from a difference of two real values. So :math:`h` can be driven to
1e-200, where the truncation term is 1e-400, i.e. exactly zero in double
precision. What comes back is the derivative to full machine accuracy.

Why it works here: :func:`qar.core.distributions.norm_cdf` is written in terms
of ``erf``, and ``scipy.special.erf`` is defined on the complex plane. Had the
normal CDF been a piecewise rational approximation — as it is in many option
libraries — this method would be unavailable. That is a concrete payoff from
building the distribution layer properly.

**Limitation.** The method needs one *analytic* perturbation, so it gives first
derivatives only. Second and third order still go through
:mod:`qar.validate.fd`.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from qar.core.bsm import CALL, BSMInputs, price
from qar.validate.fd import bump

__all__ = ["CS_STEP", "complex_derivative", "delta_by_complex_step", "vega_by_complex_step"]

#: Step size. Any value small enough that ``h**2`` underflows to zero works
#: identically; 1e-200 leaves ~120 orders of magnitude of headroom before the
#: smallest denormal, so intermediate products cannot underflow.
CS_STEP = 1e-200


def _bump_complex(inputs: BSMInputs, var: str, h: float) -> BSMInputs:
    """Shift ``var`` by ``ih``, preserving the ``r``/``b`` coupling in :func:`bump`."""
    return bump(inputs, var, complex(0.0, h))


def complex_derivative(
    func: Callable[[BSMInputs], Any],
    inputs: BSMInputs,
    var: str,
    h: float = CS_STEP,
) -> float:
    r"""First derivative of ``func`` with respect to ``var``, to ~1e-16 relative.

    Parameters
    ----------
    func:
        Must be analytic in ``var`` at ``inputs`` — true of the BSM price
        anywhere in the interior of the parameter domain, which is everywhere
        except the degenerate boundary :math:`\sigma\sqrt{T} = 0`.
    var:
        One of ``S``, ``K``, ``T``, ``sigma``, ``r``, ``q``, ``b``.
    """
    perturbed = _bump_complex(inputs, var, h)
    value = func(perturbed)
    return float(np.imag(np.asarray(value)) / h)


def delta_by_complex_step(inputs: BSMInputs, kind: Any = CALL) -> float:
    r"""Delta, :math:`\partial V/\partial S`, by complex step."""
    return complex_derivative(lambda i: price(i, kind), inputs, "S")


def vega_by_complex_step(inputs: BSMInputs, kind: Any = CALL) -> float:
    r"""Vega, :math:`\partial V/\partial\sigma`, by complex step."""
    return complex_derivative(lambda i: price(i, kind), inputs, "sigma")
