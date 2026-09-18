"""
Tests for hedging_sim. Fixed seeds; tolerances sized to Monte Carlo error.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bsm_calculator import black_scholes
from hedging_sim import (
    DERMAN_KAMAL_CONSTANT,
    HedgeResult,
    HedgeVol,
    Position,
    delta_hedge_pnl,
    simulate_gbm,
)

S0, K, R, Q = 100.0, 100.0, 0.05, 0.02
ONE_MONTH = 1.0 / 12.0
SEED = 20260918


def _fair_vol_hedge(n_steps: int, n_paths: int = 20000) -> HedgeResult:
    """Short ATM call where implied, real and hedge vol all coincide."""
    return delta_hedge_pnl(S0, K, R, Q, ONE_MONTH, 0.20, 0.20,
                           n_steps=n_steps, n_paths=n_paths, seed=SEED)


@pytest.fixture(scope="module")
def monthly_daily() -> HedgeResult:
    return _fair_vol_hedge(21)


@pytest.fixture(scope="module")
def monthly_four_a_day() -> HedgeResult:
    return _fair_vol_hedge(84)


# ---------------------------------------------------------------------------
# simulate_gbm
# ---------------------------------------------------------------------------
class TestSimulateGbm:
    def test_shape_and_start(self) -> None:
        paths = simulate_gbm(S0, 0.05, 0.0, 0.2, 1.0, 12, 50, seed=SEED)

        assert paths.shape == (50, 13)
        assert np.all(paths[:, 0] == S0)
        assert np.all(paths > 0)

    def test_same_seed_same_paths(self) -> None:
        a = simulate_gbm(S0, 0.05, 0.0, 0.2, 1.0, 12, 50, seed=SEED)
        b = simulate_gbm(S0, 0.05, 0.0, 0.2, 1.0, 12, 50, seed=SEED)

        assert np.array_equal(a, b)

    def test_terminal_moments_match_lognormal(self) -> None:
        mu, q, sigma, tau, n_paths = 0.08, 0.02, 0.25, 1.0, 200000

        terminal = simulate_gbm(S0, mu, q, sigma, tau, 4, n_paths,
                                seed=SEED)[:, -1]
        log_ret = np.log(terminal / S0)

        std_err = sigma * math.sqrt(tau / n_paths)
        assert abs(log_ret.mean() - (mu - q - 0.5 * sigma**2) * tau) < 4 * std_err
        assert log_ret.std() == pytest.approx(sigma * math.sqrt(tau), rel=0.01)

    @pytest.mark.parametrize("kwargs", [
        {"S0": 0.0}, {"sigma_real": 0.0}, {"tau": 0.0},
        {"n_steps": 0}, {"n_paths": 0},
    ])
    def test_rejects_invalid_inputs(self, kwargs: dict[str, float]) -> None:
        params: dict[str, float] = dict(S0=S0, mu=0.05, q=0.0, sigma_real=0.2,
                                        tau=1.0, n_steps=12, n_paths=10)
        params.update(kwargs)

        with pytest.raises(ValueError):
            simulate_gbm(**params)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Fair vol: only discrete-hedging noise remains (Derman & Kamal)
# ---------------------------------------------------------------------------
class TestHedgingNoise:
    def test_mean_pnl_is_zero(self, monthly_daily: HedgeResult) -> None:
        std_err = monthly_daily.std / math.sqrt(monthly_daily.pnl.size)

        assert abs(monthly_daily.mean) < 3 * std_err

    @pytest.mark.parametrize("fixture_name", ["monthly_daily",
                                              "monthly_four_a_day"])
    def test_std_matches_derman_kamal(
        self, fixture_name: str, request: pytest.FixtureRequest
    ) -> None:
        res: HedgeResult = request.getfixturevalue(fixture_name)

        assert res.std == pytest.approx(res.derman_kamal_std, rel=0.15)

    def test_four_times_the_hedges_halves_the_noise(
        self, monthly_daily: HedgeResult, monthly_four_a_day: HedgeResult
    ) -> None:
        ratio = monthly_four_a_day.std / monthly_daily.std

        assert ratio == pytest.approx(0.5, rel=0.10)

    def test_derman_kamal_field_is_the_formula(
        self, monthly_daily: HedgeResult
    ) -> None:
        vega = float(black_scholes(S0, K, R, Q, 0.20, ONE_MONTH).vega)

        expected = DERMAN_KAMAL_CONSTANT * vega * 0.20 / math.sqrt(21)
        assert monthly_daily.derman_kamal_std == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Implied != real: which vol goes into delta (Ahmad & Wilmott)
# ---------------------------------------------------------------------------
class TestThreeVolatilities:
    SIGMA_I, SIGMA_R, TAU, N = 0.30, 0.20, 1.0, 1000

    def _run(self, hedge_vol: HedgeVol, mu: float | None = None) -> HedgeResult:
        return delta_hedge_pnl(S0, K, R, Q, self.TAU, self.SIGMA_I,
                               self.SIGMA_R, hedge_vol=hedge_vol, mu=mu,
                               n_steps=self.N, n_paths=2000, seed=SEED)

    def _locked_in(self) -> float:
        v_i = float(black_scholes(S0, K, R, Q, self.SIGMA_I, self.TAU).price)
        v_r = float(black_scholes(S0, K, R, Q, self.SIGMA_R, self.TAU).price)
        return (v_i - v_r) * math.exp(R * self.TAU)

    def test_hedging_with_real_vol_locks_in_the_vol_spread(self) -> None:
        res = self._run(HedgeVol.REAL)

        assert res.mean == pytest.approx(self._locked_in(), rel=0.01)
        assert res.std < 0.05 * res.premium

    def test_locked_in_total_does_not_depend_on_drift(self) -> None:
        for mu in (-0.10, 0.25):
            res = self._run(HedgeVol.REAL, mu=mu)

            assert res.mean == pytest.approx(self._locked_in(), rel=0.01)

    def test_hedging_with_implied_vol_smooth_path_uncertain_total(self) -> None:
        real = self._run(HedgeVol.REAL)
        implied = self._run(HedgeVol.IMPLIED)

        def path_noise(res: HedgeResult) -> float:
            return float(np.mean(np.std(np.diff(res.pnl_paths, axis=1), axis=1)))

        assert path_noise(implied) < 0.5 * path_noise(real)
        assert implied.std > 2 * real.std
        # Short gamma with real < implied: the hedge earns on average every step
        assert np.mean(np.diff(implied.pnl_paths, axis=1)) > 0


# ---------------------------------------------------------------------------
# Structural properties
# ---------------------------------------------------------------------------
class TestStructure:
    def test_shapes_and_mark_to_market_endpoints(self) -> None:
        res = delta_hedge_pnl(S0, K, R, Q, 0.5, 0.25, 0.20,
                              n_steps=10, n_paths=30, seed=SEED)

        assert res.pnl.shape == (30,)
        assert res.pnl_paths.shape == (30, 11)
        assert np.all(res.pnl_paths[:, 0] == 0.0)
        assert np.array_equal(res.pnl_paths[:, -1], res.pnl)

    def test_premium_is_bsm_price_at_implied_vol(self) -> None:
        res = delta_hedge_pnl(S0, 110.0, R, Q, 0.5, 0.25, 0.20, option="put",
                              n_steps=5, n_paths=10, seed=SEED)

        expected = float(black_scholes(S0, 110.0, R, Q, 0.25, 0.5, "put").price)
        assert res.premium == pytest.approx(expected)

    def test_worthless_option_reports_nan_ratio_instead_of_raising(self) -> None:
        res = delta_hedge_pnl(S0, 60.0, R, Q, 0.03125, 0.0625, 0.5,
                              option="put", n_steps=1, n_paths=20, seed=SEED)

        assert res.premium == 0.0
        assert math.isnan(res.std_over_premium)

    def test_put_hedge_also_centres_on_zero_at_fair_vol(self) -> None:
        res = delta_hedge_pnl(S0, K, R, Q, ONE_MONTH, 0.20, 0.20, option="put",
                              n_steps=21, n_paths=20000, seed=SEED)

        assert abs(res.mean) < 3 * res.std / math.sqrt(res.pnl.size)

    @settings(max_examples=25, deadline=None)
    @given(
        strike=st.floats(min_value=60.0, max_value=140.0),
        sigma_i=st.floats(min_value=0.05, max_value=0.80),
        sigma_r=st.floats(min_value=0.05, max_value=0.80),
        tau=st.floats(min_value=0.02, max_value=3.0),
        n_steps=st.integers(min_value=1, max_value=60),
        option=st.sampled_from(["call", "put"]),
        hedge_vol=st.sampled_from(list(HedgeVol)),
    )
    def test_long_is_the_mirror_of_short(
        self, strike: float, sigma_i: float, sigma_r: float, tau: float,
        n_steps: int, option: str, hedge_vol: HedgeVol,
    ) -> None:
        common = dict(S0=S0, K=strike, r=R, q=Q, tau=tau,
                      sigma_implied=sigma_i, sigma_real=sigma_r,
                      hedge_vol=hedge_vol, option=option,
                      n_steps=n_steps, n_paths=20, seed=SEED)

        short = delta_hedge_pnl(position=Position.SHORT, **common)  # type: ignore[arg-type]
        long = delta_hedge_pnl(position=Position.LONG, **common)  # type: ignore[arg-type]

        assert np.all(np.isfinite(short.pnl))
        assert np.array_equal(long.pnl_paths, -short.pnl_paths)

    @pytest.mark.parametrize("kwargs", [
        {"K": 0.0}, {"sigma_implied": 0.0}, {"sigma_real": -0.1},
        {"option": "straddle"},
    ])
    def test_rejects_invalid_inputs(self, kwargs: dict[str, object]) -> None:
        params: dict[str, object] = dict(S0=S0, K=K, r=R, q=Q, tau=1.0,
                                         sigma_implied=0.2, sigma_real=0.2,
                                         n_steps=5, n_paths=5)
        params.update(kwargs)

        with pytest.raises(ValueError):
            delta_hedge_pnl(**params)  # type: ignore[arg-type]
