"""
Trading environment (Gymnasium) for the Kalman-DRL replication.

Causal by construction: at step t the agent sees the feature vector built from
data up to t, picks a target position, and earns the return realised from t to
t+1. PnL is always marked on the RAW close, so denoising only affects what the
agent observes, never the prices it trades at.

Action space (discrete, 3): 0 = short (-1), 1 = flat (0), 2 = long (+1).
Reward (paper Eq. 22 form):  r_t = pnl_t - gamma * cost_t - beta * drawdown_t
Transaction cost: cost_rate * |position change| (fraction of unit notional).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ACTION_TO_POSITION = {0: -1.0, 1: 0.0, 2: 1.0}


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        close: np.ndarray,
        cost_rate: float = 1e-4,   # 0.01% commission, per the paper
        beta: float = 0.1,         # drawdown penalty weight
        gamma: float = 1.0,        # transaction-cost penalty weight
    ):
        super().__init__()
        self.features = np.asarray(features, dtype=np.float32)
        self.close = np.asarray(close, dtype=np.float64)
        assert len(self.features) == len(self.close)
        self.cost_rate = float(cost_rate)
        self.beta = float(beta)
        self.gamma = float(gamma)

        self.n = len(self.close)
        n_feat = self.features.shape[1]
        # observation = market features + current position
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_feat + 1,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._t = 0
        self.position = 0.0
        self.equity = 1.0
        self.peak = 1.0
        self.equity_curve: list[float] = []

    def _obs(self) -> np.ndarray:
        return np.concatenate(
            [self.features[self._t], np.array([self.position], dtype=np.float32)]
        ).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        self.position = 0.0
        self.equity = 1.0
        self.peak = 1.0
        self.equity_curve = [1.0]
        return self._obs(), {}

    def step(self, action):
        target = ACTION_TO_POSITION[int(action)]
        cost = self.cost_rate * abs(target - self.position)
        self.position = target

        # realise the return from t -> t+1 on the raw close
        ret = self.close[self._t + 1] / self.close[self._t] - 1.0
        pnl = self.position * ret
        net = pnl - cost
        self.equity *= (1.0 + net)
        self.equity_curve.append(self.equity)

        self.peak = max(self.peak, self.equity)
        drawdown = self.equity / self.peak - 1.0  # <= 0

        reward = pnl - self.gamma * cost - self.beta * abs(drawdown)

        self._t += 1
        terminated = self._t >= self.n - 1
        return self._obs(), float(reward), bool(terminated), False, {"equity": self.equity}
