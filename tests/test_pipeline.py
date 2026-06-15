"""Tests for features, metrics, and the trading environment."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kdrl.features import FEATURE_COLUMNS, compute_features  # noqa: E402
from kdrl.metrics import compute_metrics  # noqa: E402
from kdrl.env import TradingEnv  # noqa: E402


def _ohlcv(n=400, seed=1):
    rng = np.random.default_rng(seed)
    close = np.cumsum(rng.normal(scale=0.5, size=n)) + 1000.0
    idx = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": close + rng.normal(scale=0.1, size=n),
        "high": close + np.abs(rng.normal(scale=0.3, size=n)),
        "low": close - np.abs(rng.normal(scale=0.3, size=n)),
        "close": close,
        "volume": np.abs(rng.normal(loc=100, scale=10, size=n)),
    }, index=idx)


@pytest.mark.parametrize("use_kalman", [False, True])
def test_features_shape_no_nan(use_kalman):
    df = _ohlcv()
    feats, close = compute_features(df, use_kalman=use_kalman)
    assert list(feats.columns) == FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 22
    assert not feats.isna().any().any()
    assert len(feats) == len(close)
    assert np.isfinite(feats.to_numpy()).all()


def test_metrics_known_values():
    eq = np.array([100.0, 110.0, 99.0, 108.0])  # peak 110 -> trough 99 = -10%
    m = compute_metrics(eq, periods_per_year=252)
    assert m.cumulative_return == pytest.approx(0.08)
    assert m.max_drawdown == pytest.approx(-0.1)


def test_env_steps_and_terminates():
    feats = np.zeros((50, 22), dtype=np.float32)
    close = np.linspace(100, 110, 50)
    env = TradingEnv(feats, close, cost_rate=1e-4, beta=0.1)
    obs, _ = env.reset()
    assert obs.shape == (23,)              # 22 features + position
    steps, done = 0, False
    while not done:
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
        steps += 1
    assert steps == len(close) - 1         # one reward per transition
    assert obs.shape == (23,)


def test_env_long_profits_on_uptrend():
    # Always-long on a strictly rising series must end with equity > 1.
    feats = np.zeros((30, 22), dtype=np.float32)
    close = np.linspace(100, 130, 30)
    env = TradingEnv(feats, close, cost_rate=0.0, beta=0.0)
    env.reset()
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step(2)  # action 2 = long
        done = terminated or truncated
    assert info["equity"] > 1.0
