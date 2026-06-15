"""
Strictly causal Kalman filter for denoising price series.

This is the heart of the paper's "noise-resilient" claim, and the single place
where a replication can go wrong: a Kalman *smoother* (RTS) uses future
observations and would leak look-ahead into the features, inflating results. We
deliberately implement only the forward FILTER — at each step t the estimate uses
observations up to and including t, never beyond. No smoothing pass is provided.

Model: a local-level (random-walk + noise) state-space per channel:
    x_t = x_{t-1} + w_t,      w_t ~ N(0, q)      (process / state noise)
    z_t = x_t     + v_t,      v_t ~ N(0, r)      (observation noise)

The ratio q/r controls smoothing: small q/r => heavier denoising.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def kalman_filter_1d(z: np.ndarray, q: float = 1e-4, r: float = 1e-2) -> np.ndarray:
    """Causal scalar Kalman filter (local-level model). Returns filtered series.

    Only past+present observations are used for each estimate (no look-ahead).
    """
    z = np.asarray(z, dtype=np.float64)
    n = z.size
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    x = z[0]          # state estimate, seeded with the first observation
    p = 1.0           # estimate covariance
    out[0] = x
    for t in range(1, n):
        # --- predict ---
        # x_pred = x (local level); p grows by process noise q
        p = p + q
        # --- update with observation z[t] ---
        k = p / (p + r)               # Kalman gain
        x = x + k * (z[t] - x)
        p = (1.0 - k) * p
        out[t] = x
    return out


def kalman_filter_ohlcv(
    df: pd.DataFrame,
    q: float = 1e-4,
    r: float = 1e-2,
    columns: tuple[str, ...] = ("open", "high", "low", "close"),
    filter_volume: bool = False,
) -> pd.DataFrame:
    """Apply the causal 1-D filter to each price channel (and optionally volume).

    Returns a new DataFrame with the same index/columns; unfiltered columns are
    passed through unchanged.
    """
    out = df.copy()
    cols = list(columns)
    if filter_volume and "volume" in df.columns:
        cols.append("volume")
    for c in cols:
        if c in out.columns:
            out[c] = kalman_filter_1d(out[c].to_numpy(), q=q, r=r)
    return out
