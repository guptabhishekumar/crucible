"""
Run the full matrix (each agent, Kalman on and off) and write the comparison
table the paper is built around -> results/comparison.csv and comparison.md.

    python run_all.py --timesteps 100000
"""
from __future__ import annotations

import argparse

import pandas as pd

from kdrl.agents import ALGOS
from kdrl.experiment import ROOT, append_result, load_ohlcv, run_experiment


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--algos", nargs="+", default=list(ALGOS))
    ap.add_argument("--data", default=str(ROOT / "data" / "xauusd_h1.csv"))
    args = ap.parse_args()

    df = load_ohlcv(args.data)
    print(f"bars={len(df):,} | timesteps/agent={args.timesteps:,} | algos={args.algos}")

    rows, bh = [], None
    for algo in args.algos:
        for kal in (False, True):
            print(f"\n=== {algo.upper()} | kalman={kal} ===")
            rr = run_experiment(df, algo, kal, args.timesteps, seed=args.seed)
            append_result(rr)
            m = rr.metrics
            bh = rr.bh_metrics
            rows.append({
                "Agent": algo.upper(), "Kalman": "yes" if kal else "no",
                "CumRet%": round(m.cumulative_return * 100, 2),
                "CAGR%": round(m.cagr * 100, 2),
                "Sharpe": round(m.sharpe, 3),
                "MaxDD%": round(m.max_drawdown * 100, 2),
                "Vol%": round(m.volatility * 100, 2),
            })
            print(m.pretty())

    table = pd.DataFrame(rows)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    table.to_csv(results / "comparison.csv", index=False)

    # Markdown
    hdr = "| " + " | ".join(table.columns) + " |\n"
    sep = "|" + "|".join(["---"] * len(table.columns)) + "|\n"
    body = "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in table.values)
    md = (
        "# Results: Kalman-enhanced DRL on XAU/USD H1 (out-of-sample)\n\n"
        f"Bars: {len(df):,} | train/test split 80/20 | timesteps/agent "
        f"{args.timesteps:,} | cost 0.01% | reward = pnl - cost - beta*drawdown.\n\n"
        + hdr + sep + body +
        f"\nBuy & hold (test window): {bh.cumulative_return:+.2%} cumulative, "
        f"Sharpe {bh.sharpe:.3f}, max drawdown {bh.max_drawdown:.2%}.\n"
    )
    (results / "comparison.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print("wrote -> results/comparison.csv and results/comparison.md")


if __name__ == "__main__":
    main()
