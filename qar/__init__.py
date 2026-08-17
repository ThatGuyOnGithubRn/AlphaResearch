"""Quantitative Alpha Research — a derivatives analytics suite from first principles.

Layout
------
``qar.core``
    Normal distribution, generalised Black-Scholes-Merton pricing, implied vol.
``qar.greeks``
    All 17 first-, second- and third-order Greeks, derived analytically.
``qar.validate``
    Independent numerical confirmation of every one of them.
``qar.arb``
    Put-call parity, static no-arbitrage bounds, and the trades that monetise
    a violation.
``qar.data``, ``qar.models``, ``qar.evaluation``, ``qar.backtest``
    Volatility-forecasting research harness.

No pricing formula in this package is imported from a third-party library.
NumPy and SciPy are used for arrays, optimisation and statistical tests only.
"""

__version__ = "0.1.0"
