# Black-Scholes-Merton Calculator

An interactive, browser-based calculator for European options pricing under the Black-Scholes-Merton model, with the full set of first- and second-order Greeks updating live as you move the inputs.

By [Eduardo Ramos, CFA](https://github.com/EERamos)

## Live Demo

<https://cfamexico.github.io/bsm-calculator/>

## What It Does

The dashboard prices European calls and puts under the Black-Scholes-Merton framework (with continuous dividend yield) and computes all the standard risk sensitivities a trader actually uses on a real options book.

Six sliders control the market and contract inputs (spot, strike, risk-free rate, dividend yield, volatility, time to maturity). Every chart on the page redraws in real time as you move them.

### Greeks computed

| Greek | Symbol | Meaning |
|-------|--------|---------|
| Delta | Δ | Sensitivity to spot price |
| Gamma | Γ | Convexity in spot |
| Vega  | V | Sensitivity to volatility |
| Theta | Θ | Time decay |
| Rho   | ρ | Sensitivity to interest rate |
| Vanna | ∂Δ/∂σ | Cross-sensitivity, spot and vol |
| Volga | ∂V/∂σ | Convexity in volatility |
| Charm | ∂Δ/∂t | Delta decay (calendar time, t = T − τ) |

### Visualizations

* Payoff diagram at maturity for both call and put
* Option value as a function of spot
* Each Greek plotted across a spot range
* Time decay and volatility sensitivity curves
* Two 3D price surfaces over (spot, volatility)
* A live put-call parity residual on the page as a built-in sanity check

## Why This Exists

Most options material starts and ends with the closed-form price. The interesting work begins one layer down: the partial derivatives of that price with respect to each input. Those derivatives are what a trader hedges, finances, and stress-tests. The goal of this dashboard is to make those sensitivities feel tangible. You move a slider, you see exactly how every Greek deforms in response.

## Mathematical Reference

The Black-Scholes-Merton pricing formula with continuous dividend yield `q`:

```
C = S · e^(-qτ) · N(d₁)  -  K · e^(-rτ) · N(d₂)

P = K · e^(-rτ) · N(-d₂)  -  S · e^(-qτ) · N(-d₁)

d₁ = [ ln(S/K) + (r - q + ½σ²)τ ] / (σ√τ)

d₂ = d₁ - σ√τ
```

Where `N(·)` is the standard normal CDF. The dashboard implements it via the Abramowitz & Stegun (1964) rational approximation, accurate to about 6 decimal places.

## Accuracy

The pricing engine has been cross-validated against:

* Hull, *Options, Futures, and Other Derivatives* (10th ed., Ch. 15) reference values
* `scipy.stats.norm.cdf` in Python
* Finite-difference numerical Greeks

There are two implementations and they have different precision floors:

* The **Python module** uses `scipy.stats.norm.cdf` (IEEE-754 double precision), so prices match Hull and scipy to machine precision in every test case.
* The **browser dashboard** uses the Abramowitz & Stegun rational approximation for the CDF, which carries a maximum absolute error of ~7.5e-8 in N(·). This translates to ~1e-5 absolute error on price for typical inputs.

Across 9 test scenarios (ATM, deep ITM/OTM, near-expiry, long-dated, high-vol, negative rates, with and without dividend), both implementations agree with Hull within their respective precision floors. Put-call parity holds to machine precision (~1e-15) in every case.

## Files

```
bsm-calculator/
├── index.html          The interactive dashboard (open in any browser)
├── bsm_calculator.py   Vectorized Python implementation with full Greeks
├── requirements.txt    Python dependencies
├── LICENSE             MIT license
└── README.md           This file
```

## Quick Start

### Dashboard (no install needed)

Just open `index.html` in any modern browser, or visit the [Live Demo](#live-demo) link above.

### Python module

```bash
pip install -r requirements.txt
python bsm_calculator.py
```

This runs a built-in demo that prints prices, all 8 Greeks, a put-call parity check, and a finite-difference validation of every analytic formula.

To use the module directly:

```python
from bsm_calculator import black_scholes

result = black_scholes(
    S=100, K=100, r=0.05, q=0.02,
    sigma=0.20, tau=1.0, option='call'
)

print(f"Price: {result.price:.4f}")
print(f"Delta: {result.delta:.4f}")
print(f"Gamma: {result.gamma:.4f}")
```

All inputs broadcast under NumPy rules, so you can pass arrays for vectorized batch pricing.

## Tech Stack

* **Dashboard**: pure HTML, vanilla JavaScript, Plotly.js (CDN), MathJax (CDN)
* **Python module**: NumPy, SciPy
* No frameworks, no build pipeline

## License

MIT. See [LICENSE](LICENSE).

## Author

Eduardo Ramos, CFA ([@EERamos](https://github.com/EERamos))
