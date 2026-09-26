"""
Tests for black_scholes() degenerate inputs (tau <= 0 or sigma <= 0).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bsm_calculator import black_scholes, put_call_parity_residual

S0, K, R, Q = 100.0, 100.0, 0.05, 0.0
GREEKS = ("price", "delta", "gamma", "vega", "theta",
          "rho", "vanna", "volga", "charm")


# ---------------------------------------------------------------------------
# Mixed arrays: one degenerate element must not leak into the others
# ---------------------------------------------------------------------------
class TestMixedArrays:
    def test_tau_zero_does_not_leak(self) -> None:
        res = black_scholes(S0, K, R, Q, 0.2, np.array([0.0, 1.0]))
        one_year = black_scholes(S0, K, R, Q, 0.2, 1.0)

        for g in GREEKS:
            assert np.asarray(getattr(res, g))[1] == pytest.approx(
                float(getattr(one_year, g))), g
        assert float(one_year.price) == pytest.approx(10.4506, abs=1e-4)
        assert float(one_year.delta) == pytest.approx(0.6368, abs=1e-4)

    def test_sigma_zero_does_not_leak(self) -> None:
        res = black_scholes(S0, K, R, Q, np.array([0.0, 0.2]), 1.0)
        normal = black_scholes(S0, K, R, Q, 0.2, 1.0)

        for g in GREEKS:
            assert np.asarray(getattr(res, g))[1] == pytest.approx(
                float(getattr(normal, g))), g

    @pytest.mark.parametrize("option", ["call", "put"])
    def test_output_takes_broadcast_shape(self, option: str) -> None:
        sigma = np.array([[0.0, 0.2, 0.3]])
        tau = np.array([[0.0], [1.0]])

        res = black_scholes(S0, K, R, Q, sigma, tau, option)

        for g in GREEKS:
            assert np.shape(getattr(res, g)) == (2, 3), g

    def test_no_runtime_warnings(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            black_scholes(S0, K, R, Q, np.array([0.0, 0.2, 0.2]),
                          np.array([1.0, 0.0, 1.0]))

    @settings(max_examples=200, deadline=None)
    @given(
        spots=st.lists(st.floats(50.0, 150.0), min_size=4, max_size=4),
        sigmas=st.lists(st.sampled_from([0.0, 0.1, 0.35]),
                        min_size=4, max_size=4),
        taus=st.lists(st.sampled_from([0.0, 0.25, 2.0]),
                      min_size=4, max_size=4),
        option=st.sampled_from(["call", "put"]),
    )
    def test_vectorized_matches_elementwise(
        self, spots: list[float], sigmas: list[float], taus: list[float],
        option: str,
    ) -> None:
        res = black_scholes(np.array(spots), K, R, 0.02, np.array(sigmas),
                            np.array(taus), option)

        for i, (s, sig, t) in enumerate(zip(spots, sigmas, taus)):
            one = black_scholes(s, K, R, 0.02, sig, t, option)
            for g in GREEKS:
                assert np.asarray(getattr(res, g))[i] == pytest.approx(
                    float(getattr(one, g)), rel=1e-12, abs=1e-12), g


# ---------------------------------------------------------------------------
# sigma = 0 with time left: a deterministic forward
# ---------------------------------------------------------------------------
class TestZeroVol:
    @pytest.mark.parametrize("option, delta", [("call", 1.0), ("put", 0.0)])
    def test_delta_follows_forward_not_spot(self, option: str,
                                            delta: float) -> None:
        # S = K but the forward S*e^{-q tau} - K*e^{-r tau} > 0: the call
        # finishes in the money for sure.
        res = black_scholes(S0, K, R, Q, 0.0, 1.0, option)

        assert float(res.delta) == pytest.approx(delta)

    @pytest.mark.parametrize("option", ["call", "put"])
    @pytest.mark.parametrize("spot", [90.0, 100.0, 110.0])
    def test_continuous_with_tiny_vol(self, option: str, spot: float) -> None:
        zero = black_scholes(spot, K, 0.05, 0.02, 0.0, 1.0, option)
        tiny = black_scholes(spot, K, 0.05, 0.02, 1e-8, 1.0, option)

        assert float(zero.price) == pytest.approx(float(tiny.price), abs=1e-9)
        assert float(zero.delta) == pytest.approx(float(tiny.delta), abs=1e-9)


# ---------------------------------------------------------------------------
# tau <= 0: intrinsic value, same as bsm() in index.html
# ---------------------------------------------------------------------------
class TestExpired:
    @pytest.mark.parametrize("tau", [0.0, -0.5])
    def test_price_is_intrinsic(self, tau: float) -> None:
        spots = np.array([90.0, 100.0, 110.0])

        call = black_scholes(spots, K, R, 0.02, 0.2, tau, "call")
        put = black_scholes(spots, K, R, 0.02, 0.2, tau, "put")

        np.testing.assert_allclose(call.price, [0.0, 0.0, 10.0])
        np.testing.assert_allclose(put.price, [10.0, 0.0, 0.0])
        np.testing.assert_allclose(call.delta, [0.0, 0.5, 1.0])
        np.testing.assert_allclose(put.delta, [-1.0, -0.5, 0.0])


# ---------------------------------------------------------------------------
# Regular inputs stay untouched
# ---------------------------------------------------------------------------
def test_textbook_values() -> None:
    call = black_scholes(100.0, 100.0, 0.05, 0.02, 0.20, 1.0, "call")
    put = black_scholes(100.0, 100.0, 0.05, 0.02, 0.20, 1.0, "put")

    assert float(call.price) == pytest.approx(9.2270, abs=1e-4)
    assert float(put.price) == pytest.approx(6.3301, abs=1e-4)


def test_put_call_parity_with_degenerate_elements() -> None:
    residual = put_call_parity_residual(
        np.array([90.0, 100.0, 110.0]), K, R, 0.02,
        np.array([0.0, 0.2, 0.3]), np.array([1.0, 1.0, 0.5]))

    assert residual < 1e-10


def test_rejects_non_positive_spot() -> None:
    with pytest.raises(ValueError):
        black_scholes(np.array([100.0, 0.0]), K, R, Q, 0.2, 1.0)
