"""
Roll a trained policy through an environment deterministically and return its
equity curve. Handles the recurrent (LSTM) policy's hidden-state threading.
"""
from __future__ import annotations

import numpy as np


def run_policy(model, env, recurrent: bool = False) -> np.ndarray:
    obs, _ = env.reset()
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)
    done = False
    while not done:
        if recurrent:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_start, deterministic=True
            )
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
        episode_start = np.array([done])
    return np.asarray(env.equity_curve, dtype=np.float64)


def buy_and_hold(close: np.ndarray) -> np.ndarray:
    """Equity curve for a passive long position (reference baseline)."""
    close = np.asarray(close, dtype=np.float64)
    return close / close[0]
