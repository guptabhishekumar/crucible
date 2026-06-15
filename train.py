"""
Train and evaluate a single agent.

    python train.py --algo ppo --kalman --timesteps 100000
    python train.py --algo dqn --no-kalman --timesteps 100000
"""
from __future__ import annotations

import argparse

from kdrl.agents import ALGOS
from kdrl.experiment import ROOT, append_result, load_ohlcv, run_experiment


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", choices=ALGOS, default="ppo")
    ap.add_argument("--kalman", dest="kalman", action="store_true")
    ap.add_argument("--no-kalman", dest="kalman", action="store_false")
    ap.set_defaults(kalman=True)
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=str(ROOT / "data" / "xauusd_h1.csv"))
    args = ap.parse_args()

    df = load_ohlcv(args.data)
    print(f"algo={args.algo} | kalman={args.kalman} | timesteps={args.timesteps:,} | bars={len(df):,}")

    rr = run_experiment(df, args.algo, args.kalman, args.timesteps, seed=args.seed)
    print(f"\n[{args.algo.upper()} | kalman={args.kalman}] out-of-sample, {rr.n_test:,} steps:")
    print(rr.metrics.pretty())
    print(f"\n  buy & hold (same window): {rr.bh_metrics.cumulative_return:+.2%} "
          f"cumulative, Sharpe {rr.bh_metrics.sharpe:.3f}")

    append_result(rr)
    print("\nwrote -> results/results.csv")


if __name__ == "__main__":
    main()
