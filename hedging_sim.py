"""
Discrete Delta-Hedging P&L Simulator
====================================

Puts the Greeks from `bsm_calculator` to work: trade a European option at an
implied volatility, delta-hedge it at N equally spaced dates while the
underlying follows GBM with a (possibly different) real volatility, and look
at the resulting P&L.

What it makes visible
---------------------
- A delta-hedged option is a trade of realized against implied volatility:
      dP&L ~ 1/2 * Gamma * S^2 * (sigma_real^2 - sigma_implied^2) * dt   (long)
- The volatility plugged into delta changes the shape of the P&L
  (Ahmad & Wilmott, 2005): hedging with the real vol locks in the total and
  leaves a noisy path; hedging with the implied vol gives a smooth path and a
  path-dependent total.
- Hedging a finite number of times leaves noise that decays as 1/sqrt(N)
  (Derman & Kamal, 1999):
      stdev(P&L) ~ sqrt(pi/4) * Vega * sigma / sqrt(N)

Conventions
-----------
Same as `bsm_calculator`. Three volatilities, following Ahmad & Wilmott:
sigma_implied : the vol the option is traded (and marked) at
sigma_real    : the vol the simulated path actually has
hedge vol     : the vol used to compute delta, either of the two above

P&L is reported at expiry, in currency units, for one option.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal

import numpy as np

from bsm_calculator import black_scholes


class HedgeVol(Enum):
    """Which volatility goes into the hedge ratio."""
    IMPLIED = auto()
    REAL = auto()


class Position(Enum):
    """Side of the option trade being hedged."""
    SHORT = auto()
    LONG = auto()


DERMAN_KAMAL_CONSTANT = math.sqrt(math.pi / 4.0)  # ~0.886


@dataclass(frozen=True)
class HedgeResult:
    pnl: np.ndarray            # terminal P&L per path, shape (n_paths,)
    pnl_paths: np.ndarray      # marked-to-market P&L, shape (n_paths, n_steps + 1)
    premium: float             # option value at sigma_implied, t = 0
    mean: float
    std: float
    std_over_premium: float
    derman_kamal_std: float    # sqrt(pi/4) * vega_0 * sigma_real / sqrt(N)


def _payoff(S: np.ndarray, K: float, option: Literal["call", "put"]) -> np.ndarray:
    if option == "call":
        return np.maximum(S - K, 0.0)
    return np.maximum(K - S, 0.0)


# ---------------------------------------------------------------------------
# Underlying: exact GBM step, so the only discretization error is the hedge's
# ---------------------------------------------------------------------------
def simulate_gbm(
    S0: float,
    mu: float,
    q: float,
    sigma_real: float,
    tau: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    GBM paths of shape (n_paths, n_steps + 1), first column equal to S0.

    mu is the total expected return; the price drifts at mu - q.
    """
    if S0 <= 0:
        raise ValueError("S0 must be strictly positive")
    if sigma_real <= 0 or tau <= 0:
        raise ValueError("sigma_real and tau must be strictly positive")
    if n_steps < 1 or n_paths < 1:
        raise ValueError("n_steps and n_paths must be at least 1")

    rng = np.random.default_rng(seed)
    dt = tau / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_steps = ((mu - q - 0.5 * sigma_real**2) * dt
                 + sigma_real * math.sqrt(dt) * z)
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_steps, axis=1)], axis=1)
    return S0 * np.exp(log_paths)


# ---------------------------------------------------------------------------
# Self-financing delta hedge
# ---------------------------------------------------------------------------
def delta_hedge_pnl(
    S0: float,
    K: float,
    r: float,
    q: float,
    tau: float,
    sigma_implied: float,
    sigma_real: float,
    hedge_vol: HedgeVol = HedgeVol.IMPLIED,
    option: Literal["call", "put"] = "call",
    position: Position = Position.SHORT,
    mu: float | None = None,
    n_steps: int = 21,
    n_paths: int = 5000,
    seed: int | None = None,
) -> HedgeResult:
    """
    P&L of one option traded at sigma_implied and delta-hedged n_steps times.

    Short position: collect the premium, hold delta shares, keep the rest in
    cash at r. At every hedge date the cash account earns interest, receives
    the dividend yield on the shares held, and pays for the rebalance. At
    expiry the hedge is unwound against the payoff. Long is the mirror image.

    Along the path the option is marked at sigma_implied (the market keeps
    quoting implied), so `pnl_paths` is what a desk would see day by day.

    mu defaults to r. It only matters for the distribution of the paths; the
    hedge itself never uses it.
    """
    if K <= 0:
        raise ValueError("K must be strictly positive")
    if sigma_implied <= 0:
        raise ValueError("sigma_implied must be strictly positive")
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")

    drift = r if mu is None else mu
    S = simulate_gbm(S0, drift, q, sigma_real, tau, n_steps, n_paths, seed)

    dt = tau / n_steps
    # Hedge dates t_0 .. t_{N-1}. Delta is never evaluated at tau = 0.
    tau_grid = tau - dt * np.arange(n_steps)

    marks = black_scholes(S[:, :-1], K, r, q, sigma_implied, tau_grid, option)
    if hedge_vol is HedgeVol.IMPLIED:
        delta = np.asarray(marks.delta)
    else:
        delta = np.asarray(black_scholes(S[:, :-1], K, r, q, sigma_real,
                                         tau_grid, option).delta)
    option_value = np.concatenate(
        [np.asarray(marks.price), _payoff(S[:, -1:], K, option)], axis=1)

    premium = float(option_value[0, 0])
    growth = math.exp(r * dt)
    div_accrual = math.exp(q * dt) - 1.0

    cash = np.full(n_paths, premium) - delta[:, 0] * S0
    short_pnl = np.zeros_like(S)
    for k in range(1, n_steps + 1):
        held = delta[:, k - 1]
        cash = cash * growth + held * S[:, k] * div_accrual
        short_pnl[:, k] = cash + held * S[:, k] - option_value[:, k]
        if k < n_steps:
            cash = cash - (delta[:, k] - held) * S[:, k]

    sign = 1.0 if position is Position.SHORT else -1.0
    pnl_paths = sign * short_pnl
    pnl = pnl_paths[:, -1]

    vega_0 = float(black_scholes(S0, K, r, q, sigma_implied, tau, option).vega)
    std = float(np.std(pnl))
    return HedgeResult(
        pnl=pnl,
        pnl_paths=pnl_paths,
        premium=premium,
        mean=float(np.mean(pnl)),
        std=std,
        # A far out-of-the-money premium can underflow to exactly zero
        std_over_premium=std / premium if premium > 0 else math.nan,
        derman_kamal_std=(DERMAN_KAMAL_CONSTANT * vega_0 * sigma_real
                          / math.sqrt(n_steps)),
    )


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Discrete Delta-Hedging P&L Simulator")
    print("=" * 70)
    S0, K, r, q, tau = 100.0, 100.0, 0.05, 0.02, 1.0 / 12.0

    # 1) Hedging noise vs number of rebalances (Derman & Kamal)
    sigma = 0.20
    print(f"\nShort ATM call, sigma_real = sigma_implied = {sigma:.0%}, "
          f"tau = 1 month")
    print("Hedging noise as a fraction of the premium:\n")
    print(f"{'N':>6}{'simulated':>14}{'Derman-Kamal':>16}")
    print("-" * 36)
    for n in (5, 21, 84, 252):
        res = delta_hedge_pnl(S0, K, r, q, tau, sigma, sigma,
                              n_steps=n, n_paths=20000, seed=7)
        print(f"{n:>6}{res.std_over_premium:>14.2%}"
              f"{res.derman_kamal_std / res.premium:>16.2%}")

    # 2) Which volatility goes into delta (Ahmad & Wilmott)
    sigma_i, sigma_r, tau = 0.30, 0.20, 1.0
    v_i = float(black_scholes(S0, K, r, q, sigma_i, tau).price)
    v_r = float(black_scholes(S0, K, r, q, sigma_r, tau).price)
    locked_in = (v_i - v_r) * math.exp(r * tau)
    print(f"\nShort ATM call sold at {sigma_i:.0%} implied, real vol "
          f"{sigma_r:.0%}, tau = 1 year, N = 252")
    print(f"(V(implied) - V(real)) carried to expiry = {locked_in:.4f}\n")
    print(f"{'hedge with':<12}{'mean P&L':>12}{'stdev P&L':>12}"
          f"{'path noise':>12}")
    print("-" * 48)
    for hv in (HedgeVol.REAL, HedgeVol.IMPLIED):
        res = delta_hedge_pnl(S0, K, r, q, tau, sigma_i, sigma_r,
                              hedge_vol=hv, n_steps=252, n_paths=20000,
                              seed=7)
        path_noise = float(np.mean(np.std(np.diff(res.pnl_paths, axis=1),
                                          axis=1)))
        print(f"{hv.name.lower():<12}{res.mean:>12.4f}{res.std:>12.4f}"
              f"{path_noise:>12.4f}")
    print("\npath noise = average per-step stdev of the marked-to-market P&L")
