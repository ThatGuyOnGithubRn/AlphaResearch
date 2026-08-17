# Quantitative Alpha Research

Self-directed project, Spring 2026. A derivatives analytics suite built from first
principles — no wrapping of `QuantLib`, `py_vollib`, or similar. Deriving and
implementing the math is the point of the project, so reaching for a library that
already solves the problem defeats it.

## Scope

**Built:**
- Black-Scholes-Merton pricing for European options
- All first- and second-order Greeks, derived analytically rather than bumped
- Numerical validation of Delta and Vega against finite-difference approximations
- Put-call parity enforced as a constraint, used to flag arbitrage

**In progress:**
- Neural network models
- Markovian models
- Advanced time-series volatility forecasting (GARCH-family and beyond)

## Working rules

- **Derive, don't import.** Closed-form results get implemented from the derivation.
  A library is fine for linear algebra, optimization, and data handling; not for
  pricing, Greeks, or volatility models.
- **Every analytical result needs a numerical check.** New Greeks and closed forms
  ship alongside a finite-difference test that confirms them within tolerance.
  This is already the pattern for Delta and Vega — extend it, don't skip it.
- **State conventions explicitly.** Whether rates are continuous or discrete, whether
  vol is annualized, whether time is in years or days, and the day-count basis.
  Silent mismatches here are the main source of wrong numbers in this domain.
- **Keep edge cases honest.** Zero time to expiry, zero volatility, and deep ITM/OTM
  should return defensible values rather than `nan`.

## Notes

- No git repository yet.
- The directory is empty apart from config — the code lives elsewhere or hasn't
  landed here yet. Confirm where before assuming a layout.
