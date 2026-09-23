# Design: Discrete Delta-Hedging P&L Simulator

Status: **Python module and tests implemented** (`hedging_sim.py`,
`tests/test_hedging_sim.py`). The dashboard section (section 6) is still a
proposal and is not part of this change.

## 1. Motivation

The README states the thesis of this project: the interesting work is one layer
below the closed-form price, in the derivatives "a trader hedges, finances, and
stress-tests". Today the dashboard shows those derivatives as static curves. It
never shows them *doing their job*.

This proposal adds the missing step: sell (or buy) the option, delta-hedge it
with the Greeks the calculator already produces, and look at the P&L that
results. Three textbook facts become visible and interactive:

1. A delta-hedged option is a trade of realized against implied volatility:

   ```
   dP&L ~ 1/2 * Gamma * S^2 * (sigma_real^2 - sigma_implied^2) * dt      (long option)
   ```

2. Which volatility you plug into delta changes the *shape* of the P&L, not
   just its size (Ahmad & Wilmott 2005).
3. Hedging a finite number of times leaves noise that shrinks as `1/sqrt(N)`
   (Derman & Kamal 1999):

   ```
   stdev(P&L) ~ sqrt(pi/4) * Vega * sigma / sqrt(N)        sqrt(pi/4) ~ 0.886
   ```

The gamma-theta trade-off is already in the CFA Level II curriculum; this makes
it something the reader can move with a slider.

## 2. Scope

In scope:

- One European call or put, long or short, on the existing BSM inputs.
- Underlying follows GBM with a user-chosen **real** volatility and drift.
- Delta hedge rebalanced at `N` equally spaced dates, self-financing.
- Choice of hedge volatility: implied or real.
- One new Python module, one new dashboard section, one test file.

Out of scope (deliberately):

- Transaction costs (Leland 1985). Natural follow-up, not in this change.
- Stochastic volatility, jumps, gamma/vega hedging with other options.
- American exercise, discrete dividends.
- Any change to `black_scholes()` or to existing dashboard sections.

## 3. The three volatilities

Notation follows Ahmad & Wilmott:

| Symbol | Meaning | Source in the UI |
|---|---|---|
| `sigma_i` | implied: the vol the option is traded at | existing volatility slider |
| `sigma_r` | real: the vol the simulated path actually has | new slider |
| `sigma_h` | hedge: the vol used to compute delta | new toggle, `implied` or `real` |

Expected behaviour, short option, `sigma_i > sigma_r`:

| Hedge with | Path of P&L | Total P&L |
|---|---|---|
| real | noisy mark-to-market | (almost) deterministic: `V(sigma_i) - V(sigma_r)`, carried to expiry |
| implied | smooth, same sign every step | path-dependent; largest when spot finishes near the strike |

## 4. Hedging mechanics

Short one option, `N` rebalances, `dt = tau / N`, dates `t_0 .. t_N`.

```
t_0:   premium  C_0   = V(S_0, sigma_i)
       shares   D_0   = delta(S_0, tau, sigma_h)
       cash     B_0   = C_0 - D_0 * S_0

t_k (k = 1 .. N-1):
       B_k = B_{k-1} * exp(r*dt)                        interest on cash
             + D_{k-1} * S_k * (exp(q*dt) - 1)          dividends on shares held
             - (D_k - D_{k-1}) * S_k                    rebalance

t_N:   B_N as above without the rebalance term
       P&L_short = B_N + D_{N-1} * S_N - payoff(S_N)
```

`P&L_long = -P&L_short`. P&L is reported at expiry, and also as a fraction of
the premium `C_0`.

Path simulation uses the exact GBM step, so there is no discretization error in
the underlying, only in the hedge:

```
S_{k+1} = S_k * exp( (mu - q - sigma_r^2 / 2) * dt + sigma_r * sqrt(dt) * Z )
```

Delta is never evaluated at `tau = 0`: the last hedge ratio is `D_{N-1}`. This
also keeps the simulator clear of the degenerate branch in `black_scholes()`
(see section 8).

## 5. Python module: `hedging_sim.py`

Same style as `bsm_calculator.py`: NumPy, dataclass result, vectorized across
paths, `__main__` demo. It imports `black_scholes` and duplicates no pricing
logic.

```python
def simulate_gbm(S0, mu, q, sigma_real, tau, n_steps, n_paths, seed=None) -> np.ndarray:
    """Exact GBM paths, shape (n_paths, n_steps + 1)."""

@dataclass
class HedgeResult:
    pnl: np.ndarray            # terminal P&L per path, shape (n_paths,)
    pnl_paths: np.ndarray      # marked-to-market P&L, shape (n_paths, n_steps + 1)
    premium: float             # C_0 at sigma_implied
    mean: float
    std: float
    std_over_premium: float
    derman_kamal_std: float    # sqrt(pi/4) * vega_0 * sigma / sqrt(N)

def delta_hedge_pnl(
    S0, K, r, q, tau,
    sigma_implied, sigma_real,
    hedge_vol: HedgeVol = HedgeVol.IMPLIED,        # Enum: IMPLIED | REAL
    option: Literal["call", "put"] = "call",       # same contract as black_scholes
    position: Position = Position.SHORT,           # Enum: SHORT | LONG
    mu: float | None = None,       # defaults to r
    n_steps: int = 21,
    n_paths: int = 5000,
    seed: int | None = None,
) -> HedgeResult: ...
```

Implementation note: deltas for all paths and all hedge dates come from one
broadcast call to `black_scholes(S[:, :-1], K, r, q, sigma_h, tau_grid[:-1])`,
so the loop over time only does cash-account arithmetic.

Marked-to-market P&L along the path values the option at `sigma_i` (the market
keeps quoting implied), which is what makes the "hedge with real" path noisy.

`__main__` prints a small table: `N in {5, 21, 84, 252}` against simulated
`std / premium` and the Derman-Kamal prediction, then the two Ahmad-Wilmott
cases.

## 6. Dashboard section

New narrative section, same layout pattern as the existing ones, placed after
"Time and volatility" and before the 3D surfaces. Working title: **"Putting the
Greeks to work"**.

Reused inputs: spot, strike, rate, dividend yield, volatility (read as
`sigma_i`), maturity, call/put.

New controls:

| Control | Range | Default |
|---|---|---|
| Real volatility `sigma_r` | same range as the vol slider | equal to `sigma_i` |
| Rebalances `N` | 1 to 252 | 21 |
| Hedge with | implied / real | implied |
| Position | short / long | short |
| Drift `mu` | -20% to +30% | `r` |
| New paths | button | n/a |

Charts (Plotly, existing theme helpers `baseLayout` / `readColors`):

1. **P&L histogram** at expiry, in units of premium, with the simulated mean and
   a shaded band at plus/minus the Derman-Kamal standard deviation.
2. **P&L paths**: about 20 marked-to-market paths through time. This is where
   "hedge with implied is smooth, hedge with real is noisy" is seen directly.

Readout strip under the charts: mean, stdev, stdev / premium, Derman-Kamal
prediction, and the closed-form `V(sigma_i) - V(sigma_r)`.

Behaviour:

- 500 paths in the browser. Worst case `500 x 252` calls to the existing JS
  `bsm()`; this is tens of milliseconds, so the section re-renders from the
  normal `render()` cycle with a short debounce.
- **Common random numbers**: normals are drawn once from a seeded generator
  (`mulberry32` plus Box-Muller, about 15 lines, no dependency) and reused until
  the user presses "New paths". Moving a slider then changes the parameter and
  not the noise, so the charts deform smoothly instead of flickering.
- No new CDN dependency, no build step.

The JS and Python simulators use different generators, so they agree in
distribution, not path by path. This is the same relationship the two pricing
engines already have.

## 7. Validation

`tests/test_hedging_sim.py` (pytest plus one hypothesis property test, listed
in `requirements-dev.txt`). Fixed seeds, tolerances sized to Monte Carlo error.
The table is the core; the file also covers `simulate_gbm`, input validation
and result shapes.

| # | Setup | Assertion |
|---|---|---|
| 1 | `sigma_r = sigma_i`, ATM call, `N = 21` | mean P&L within 3 standard errors of 0 |
| 2 | same, `N in {21, 84}` | `std` within 15% of `0.886 * vega_0 * sigma / sqrt(N)` |
| 3 | same | `std(N=84) / std(N=21)` within 10% of `0.5` |
| 4 | short, `sigma_i = 0.30`, `sigma_r = 0.20`, hedge with real, `N = 1000` | mean within tolerance of `(V(sigma_i) - V(sigma_r)) * exp(r*tau)`; std small relative to premium |
| 5 | same vols, hedge with implied | marked-to-market increments are smoother than in case 4 (lower per-step stdev) with a positive mean; terminal P&L has materially higher dispersion than case 4 |
| 6 | any setup | `P&L_long == -P&L_short` exactly, same seed |
| 7 | `mu` sweep, hedge with real | mean terminal P&L independent of `mu` within Monte Carlo error |

Note on test 2: Derman and Kamal report 19.3% of premium at `N = 21` and 9.7%
at `N = 84` for their example. The test asserts against the formula with this
repo's inputs, not against those two figures, since their exact contract
parameters are not reproduced here.

## 8. Related issue found while reading the code

Not part of this change, but relevant to anyone calling `black_scholes()` with
arrays:

```python
if np.any(np.asarray(tau) <= 0) or np.any(np.asarray(sigma) <= 0):
    return _degenerate_result(S, K, r, q, tau, option)
```

If a single element of a vectorized input is degenerate, the *whole* array is
routed to the degenerate branch and every Greek comes back as zero. A masked
`np.where` would fix it. The simulator sidesteps this by never pricing at
`tau = 0`. Happy to send a separate small PR with the fix and a regression
test.

## 9. File changes

This change:

```
hedging_sim.py                  new
tests/test_hedging_sim.py       new
requirements-dev.txt            new (pytest, hypothesis)
pyproject.toml                  new, pytest path config only
README.md                       new "Hedging simulator" subsection, Files tree
```

Follow-up, if wanted:

```
index.html                      one new <section>, one renderHedging() function, ~200 lines
```

## 10. Open questions for the maintainer

1. Is a simulation section in scope for the dashboard, or should it stay purely
   closed-form? The Python module stands on its own either way.
2. Placement and title of the section.
3. Tests live in a pytest file; the `__main__` demo prints the same checks in
   the style the repo already uses. Fine to drop either if you prefer one.

## References

- Derman, E. and Kamal, M. (1999). "When You Cannot Hedge Continuously: The
  Corrections of Black-Scholes." Goldman Sachs Quantitative Strategies Research
  Notes; *Risk*, January 1999.
- Ahmad, R. and Wilmott, P. (2005). "Which Free Lunch Would You Like Today,
  Sir? Delta Hedging, Volatility Arbitrage and Optimal Portfolios." *Wilmott
  Magazine*, November 2005.
- El Karoui, N., Jeanblanc-Picque, M. and Shreve, S. (1998). "Robustness of the
  Black and Scholes Formula." *Mathematical Finance* 8(2). Background for the
  P&L identity in section 1.
- Hull, J. *Options, Futures, and Other Derivatives*, chapter on the Greek
  letters (delta-hedging simulation tables).
