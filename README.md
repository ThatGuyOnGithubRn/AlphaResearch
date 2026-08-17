# Quantitative Alpha Research

A derivatives analytics suite built from first principles — Black-Scholes-Merton
pricing, seventeen analytically derived Greeks, independent numerical validation
of every one of them, and model-free arbitrage detection that emits the trade,
not just the complaint.

No pricing formula in this package comes from a library. NumPy and SciPy are used
for arrays, optimisation and statistical tests; the normal CDF, the pricing
equation, every Greek, and the GARCH likelihood are implemented from their
derivations.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest                        # 3629 tests
.venv/bin/python examples/01_greeks_surface.py
```

---

## What is finished

### Part 1 — pricing and Greeks

Generalised **cost-of-carry** form, so one implementation covers four markets:

| `b` | Model | Underlying |
|---|---|---|
| `b = r` | Black-Scholes-Merton | Non-dividend stock |
| `b = r - q` | Merton (1973) | Continuous dividend yield |
| `b = 0` | Black (1976) | Futures |
| `b = r - r_f` | Garman-Kohlhagen | FX |

**All 17 Greeks**, each derived analytically with the derivation in its docstring:

| Order | Greeks |
|---|---|
| First | Delta, Vega, Theta, Rho, Epsilon, Dual Delta |
| Second | Gamma, Vanna, Volga, Charm, Veta, Vera, Dual Gamma |
| Third | Speed, Zomma, Color, Ultima |

Degenerate limits (`T → 0`, `σ → 0`, deep ITM/OTM) return their analytic limits
rather than `nan`.

### Part 2 — validation and arbitrage

**Two independent numerical channels**, so an error in one cannot confirm an
error in the other:

- **Complex-step differentiation** for first derivatives. Because
  `f(x + ih)/h` involves no subtraction, there is no cancellation, so `h` can be
  driven to `1e-200` and Delta and Vega come back accurate to **~1e-13 relative**.
  This works only because the normal CDF is built on `erf`, which is defined on
  the complex plane — a concrete payoff from writing the distribution layer
  properly instead of using a rational approximation.
- **Richardson-extrapolated central differences** for second and third order,
  compared against the differencing scheme's *own* estimate of its error rather
  than a hand-tuned constant.

```
greek        ord         method  checked  skipped   worst rel  status
delta          1   complex-step      280        8    8.45e-13  PASS
vega           1   complex-step      272       16    1.08e-12  PASS
gamma          2  richardson-fd      240       48    3.52e-05  PASS
ultima         3  richardson-fd      224       64    1.13e-02  PASS
...
4896/4896 checks passed across 144 parameter points
```

The grid deliberately spans `b = r`, `b = r - q`, `b = 0` and negative rates. A
missing carry term is invisible at `b = r`, where it multiplies zero — the most
common way a hand-derived Greek library is wrong and passes its own tests. There
is a test asserting exactly that.

**Arbitrage** — put-call parity, static bounds, strike monotonicity, butterfly
convexity, and carry-scaled calendar spreads. Every violation returns the trade:

```
put-call parity: c - p = 1.65399 but S*e^((b-r)T) - K*e^(-rT) = 1.47399

  SELL       1.0000  call K=100 @ 7.6830
  BUY        1.0000  put K=100 @ 6.0290
  BUY        0.9900  underlying (grows to 1.0000) @ 100.0000
  BORROW    97.5310  cash at r=0.05 to T=0.5 (repays 100.0000)

  cashflow now:     +0.1800
  cashflow at T:    -0.0000  (worst case, locked)
```

Each leg knows its own payoff as a function of terminal spot, so every trade is
swept across a range of terminal prices and checked for a non-negative outcome.
That sweep caught a real bug during development: a conversion under a dividend
yield that was silently short a fraction of a share.

### Part 3 — scaffold, running end to end

Baselines and evaluation are implemented; the interesting models are stubs.

```
model                  QLIKE           MSE    MZ slope     MZ R2        bias
garch(1,1)           0.50541    1.0077e-06       1.784     0.083      biased
ewma                 0.57155    1.0329e-06       0.890     0.051    unbiased
random-walk          1.15670    1.6696e-06       0.227     0.051      biased

comparison                               DM stat   p-value
garch(1,1) vs random-walk                 -5.752    0.0000
garch(1,1) vs ewma                        -0.965    0.3349
```

GARCH beats the random walk decisively and is statistically indistinguishable
from EWMA — which is the well-known finding, and a sign the harness is honest.

---

## What is not done

`qar/models/research.py` holds four stubs, each with its equation, its expected
advantage, and the specific thing that usually goes wrong. They raise
`NotImplementedError` rather than returning a plausible number, so a
half-finished model cannot quietly enter a results table.

- **GJR-GARCH** — asymmetric response; note stationarity becomes `α + γ/2 + β < 1`
- **Markov-switching** — multi-modal likelihood, needs multi-start and a label-order fix
- **HAR-RV** — the real benchmark; fit on `log RV` with the Jensen correction
- **Neural** — target `log RV`, standardise on training statistics only, keep it small

Each already conforms to the `Forecaster` interface, so filling one in makes it
appear in the comparison automatically.

The backtest's implied-vol series is a **trailing-realised proxy**, not market
data — Deribit's public API does not serve historical chains. Its numbers size
the machinery and the cost sensitivity; they are not evidence of profitability.

---

## Layout

```
qar/
  core/         distributions, generalised BSM, implied vol
  greeks/       first.py, second.py, third.py + a registry
  validate/     complex-step, Richardson FD, sweep harness
  arb/          parity, bounds, violations with executable trades
  data/         Deribit client (cached), realised-variance estimators
  models/       Forecaster interface, baselines, research stubs
  evaluation/   QLIKE/MSE, Diebold-Mariano, Mincer-Zarnowitz, walk-forward
  backtest/     delta-hedged straddle, transaction-cost sweep
examples/       01-05, runnable, printing real numbers
tests/          3629 tests
```

## Examples

| | |
|---|---|
| `01_greeks_surface.py` | All 17 Greeks, four carry regimes, the validation report, a plot |
| `02_arbitrage.py` | Planted mispricings and the trades that capture them |
| `03_deribit_chain.py` | Live BTC chain: our IV vs Deribit's, implied forward, arbitrage scan |
| `04_baseline_eval.py` | Walk-forward comparison with significance tests |
| `05_hedged_backtest.py` | Delta-hedged straddle, transaction-cost sweep |

Example 03 reproduces Deribit's own mark IV to a median of **0.002 vol points**.

## Notes on method

Three decisions that shaped the code more than anything else:

**Rho is ambiguous unless you say what happens to the carry rate.** Under BSM
`b = r`, so `ρ = TKe^{-rT}N(d₂)`; under Black-76 `b` is pinned and `ρ = -TV`.
The API takes an explicit flag rather than picking one, because the wrong answer
here looks entirely plausible. The finite-difference validator has the matching
coupling, so it cannot "confirm" the wrong Rho.

**Tolerances come from error analysis, not from tuning until green.** A central
difference of order *k* has error `O(h²) + O(ε/hᵏ)`, giving a best achievable
accuracy of `ε^(2/(k+2))`. Where the scheme knows it is doing worse than that,
its own Richardson error estimate takes over. Values below the method's noise
floor are reported as unresolvable rather than silently passed.

**Look-ahead is the failure mode that leaves every number looking fine.** The
walk-forward evaluator constructs training slices itself; there is a test that
instruments a model to record what it was shown and asserts it never saw its own
target. The one place a leak did get in — a trailing average whose window
included the current day — inflated the backtest Sharpe from 1.93 to 3.06 before
it was caught.
