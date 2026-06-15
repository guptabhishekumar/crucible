"""
Feature engineering -> the 22-dimensional market state.

When use_kalman=True the indicators are computed on the CAUSALLY filtered OHLC;
when False they use the raw OHLC. The traded price (for PnL) is always the RAW
close, so the only thing that changes between the two runs is what the agent
*sees* — which is exactly the variable the paper claims matters.

All indicators are hand-written (numpy/pandas) to avoid TA-Lib build issues and
to keep every computation causal (rolling/ewm only look backwards).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .kalman import kalman_filter_ohlcv

FEATURE_COLUMNS = [
    "open_over_close", "high_over_close", "low_over_close", "ret", "vol_z",
    "sma10_r", "sma20_r", "sma50_r", "ema12_r", "ema26_r",
    "macd_n", "macd_signal_n", "macd_hist_n", "rsi14", "boll_pctb",
    "atr14_n", "mom10", "vol20", "vol50", "ret_lag1", "ret_lag2", "ret_lag3",
]  # 22 features


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0) / 100.0  # scaled 0..1


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def compute_features(
    df: pd.DataFrame,
    use_kalman: bool,
    q: float = 1e-4,
    r: float = 1e-2,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (features[22 cols], raw_close) aligned and NaN-free.

    df must have columns open/high/low/close/volume and a DatetimeIndex.
    """
    raw_close = df["close"].astype("float64").copy()
    src = kalman_filter_ohlcv(df, q=q, r=r) if use_kalman else df.copy()

    o, h, l, c, v = (src[k].astype("float64") for k in ["open", "high", "low", "close", "volume"])
    f = pd.DataFrame(index=df.index)

    # 1-5: OHLCV-derived
    f["open_over_close"] = o / c - 1.0
    f["high_over_close"] = h / c - 1.0
    f["low_over_close"] = l / c - 1.0
    f["ret"] = c.pct_change()
    f["vol_z"] = (v - v.rolling(20).mean()) / v.rolling(20).std(ddof=0)

    # 6-10: moving-average ratios
    f["sma10_r"] = c.rolling(10).mean() / c - 1.0
    f["sma20_r"] = c.rolling(20).mean() / c - 1.0
    f["sma50_r"] = c.rolling(50).mean() / c - 1.0
    f["ema12_r"] = _ema(c, 12) / c - 1.0
    f["ema26_r"] = _ema(c, 26) / c - 1.0

    # 11-13: MACD (normalised by price)
    macd = _ema(c, 12) - _ema(c, 26)
    macd_signal = _ema(macd, 9)
    f["macd_n"] = macd / c
    f["macd_signal_n"] = macd_signal / c
    f["macd_hist_n"] = (macd - macd_signal) / c

    # 14-16: oscillators / volatility
    f["rsi14"] = _rsi(c, 14)
    mid = c.rolling(20).mean()
    std = c.rolling(20).std(ddof=0)
    f["boll_pctb"] = (c - (mid - 2 * std)) / (4 * std)
    f["atr14_n"] = _atr(src, 14) / c

    # 17-22: momentum, realised vol, lagged returns
    f["mom10"] = c / c.shift(10) - 1.0
    ret = c.pct_change()
    f["vol20"] = ret.rolling(20).std(ddof=0)
    f["vol50"] = ret.rolling(50).std(ddof=0)
    f["ret_lag1"] = ret.shift(1)
    f["ret_lag2"] = ret.shift(2)
    f["ret_lag3"] = ret.shift(3)

    f = f[FEATURE_COLUMNS]
    valid = f.notna().all(axis=1)
    f = f.loc[valid]
    raw_close = raw_close.loc[valid]
    return f.replace([np.inf, -np.inf], 0.0), raw_close
