"""
End-to-end experiment pipeline: data -> features -> chronological split ->
leakage-safe scaling -> train agent -> evaluate out-of-sample -> metrics.

Keeping this in one place guarantees the Kalman and non-Kalman runs (and every
agent) go through exactly the same steps, so the only variable is what the brief
under study changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .agents import build_agent
from .env import TradingEnv
from .evaluate import buy_and_hold, run_policy
from .features import compute_features
from .metrics import Metrics, compute_metrics

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_ohlcv(path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[["open", "high", "low", "close", "volume"]].dropna()


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score using TRAIN statistics only (no test leakage)."""
    mu = train.mean(axis=0)
    sd = train.std(axis=0) + 1e-8
    return (train - mu) / sd, (test - mu) / sd


@dataclass
class RunResult:
    algo: str
    use_kalman: bool
    metrics: Metrics
    equity: np.ndarray
    bh_metrics: Metrics
    n_train: int
    n_test: int


def run_experiment(
    df: pd.DataFrame,
    algo: str,
    use_kalman: bool,
    timesteps: int,
    seed: int = 0,
    split: float = 0.8,
    cost: float = 1e-4,
    beta: float = 0.1,
    q: float = 1e-4,
    r: float = 1e-2,
) -> RunResult:
    feats, close = compute_features(df, use_kalman=use_kalman, q=q, r=r)
    X = feats.to_numpy(dtype="float32")
    c = close.to_numpy(dtype="float64")

    k = int(len(X) * split)
    Xtr, Xte = X[:k], X[k:]
    ctr, cte = c[:k], c[k:]
    Xtr, Xte = _standardize(Xtr, Xte)

    train_env = TradingEnv(Xtr, ctr, cost_rate=cost, beta=beta)
    test_env = TradingEnv(Xte, cte, cost_rate=cost, beta=beta)

    model = build_agent(algo, train_env, seed=seed)
    model.learn(total_timesteps=timesteps, progress_bar=False)

    equity = run_policy(model, test_env, recurrent=(algo.lower() == "rppo"))
    metrics = compute_metrics(equity)
    bh_metrics = compute_metrics(buy_and_hold(cte))

    return RunResult(algo, use_kalman, metrics, equity, bh_metrics, len(Xtr), len(Xte))


def append_result(rr: RunResult, path: Path = None) -> None:
    path = path or (RESULTS / "results.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"algo": rr.algo, "kalman": rr.use_kalman, **rr.metrics.as_row(),
           "n_train": rr.n_train, "n_test": rr.n_test}
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)
