"""
Performance metrics, computed identically for every agent and the baselines.

All figures derive from a per-step equity curve. Conventions are stated once and
applied everywhere (this is exactly the discipline that keeps a comparison fair):
  * Sharpe / volatility annualised with sqrt(PERIODS_PER_YEAR)
  * risk-free rate = 0
  * hourly XAU/USD => ~24h x ~252 trading days = 6048 periods/year
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 24 * 252  # 6048; gold trades ~24h on ~252 weekdays/year


@dataclass(frozen=True)
class Metrics:
    cumulative_return: float   # fraction over the whole test window
    cagr: float                # annualised
    sharpe: float              # annualised, rf=0
    max_drawdown: float        # negative fraction
    volatility: float          # annualised stdev of per-step returns
    n_steps: int

    def as_row(self) -> dict:
        d = asdict(self)
        d["cumulative_return_pct"] = round(self.cumulative_return * 100, 4)
        d["cagr_pct"] = round(self.cagr * 100, 4)
        d["max_drawdown_pct"] = round(self.max_drawdown * 100, 4)
        d["volatility_pct"] = round(self.volatility * 100, 4)
        d["sharpe"] = round(self.sharpe, 4)
        return d

    def pretty(self) -> str:
        return (
            f"  Cumulative return : {self.cumulative_return:+.2%}\n"
            f"  CAGR              : {self.cagr:+.2%}\n"
            f"  Sharpe            : {self.sharpe:.3f}\n"
            f"  Max drawdown      : {self.max_drawdown:.2%}\n"
            f"  Volatility (ann.) : {self.volatility:.2%}\n"
            f"  Steps             : {self.n_steps}"
        )


def compute_metrics(equity: np.ndarray, periods_per_year: int = PERIODS_PER_YEAR) -> Metrics:
    equity = np.asarray(equity, dtype=np.float64)
    equity = equity[np.isfinite(equity)]
    if equity.size < 2:
        return Metrics(0.0, 0.0, float("nan"), 0.0, 0.0, int(equity.size))

    rets = np.diff(equity) / equity[:-1]
    cumulative = equity[-1] / equity[0] - 1.0

    years = equity.size / periods_per_year
    cagr = (equity[-1] / equity[0]) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    sd = rets.std(ddof=1)
    sharpe = (rets.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")
    vol = sd * np.sqrt(periods_per_year)

    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    max_dd = float(drawdown.min())

    return Metrics(float(cumulative), float(cagr), float(sharpe), max_dd, float(vol),
                   int(equity.size))
