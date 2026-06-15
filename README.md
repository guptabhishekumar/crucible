<h1 align="center">Kalman-DRL Replication</h1>

<p align="center"><i>Replicating, and stress-testing, "Kalman-Enhanced Deep Reinforcement Learning for Noise-Resilient Algorithmic Trading" (IJACSA 16(11)).</i></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.12" height="20">
  <img src="https://img.shields.io/badge/PyTorch-CPU-EE4C2C?style=flat-square" alt="PyTorch" height="20">
  <img src="https://img.shields.io/badge/stable--baselines3-DQN%20%C2%B7%20PPO%20%C2%B7%20RPPO-1f77b4?style=flat-square" alt="SB3" height="20">
  <img src="https://img.shields.io/badge/tests-8%20passing-2ea44f?style=flat-square" alt="tests" height="20">
</p>

## The paper

It applies a **Kalman filter to denoise hourly XAU/USD**, then trains **DQN, PPO and Recurrent-PPO** agents on a 22-feature state with a multi-objective reward (return, minus drawdown and transaction cost). It reports that the Kalman step transforms performance, e.g. **PPO + Kalman: 80.21% cumulative, Sharpe 12.10, −0.48% max drawdown**, versus raw PPO at Sharpe 0.45 / −12.52% drawdown — up to a **29× Sharpe improvement** over 8 years.

## The question this repo answers

A **Sharpe near 12 with a sub-1% drawdown over 8 years** is extraordinary. That is exactly the kind of number worth distrusting until it is shown to survive a strictly fair setup. The most common way a result like this appears is **look-ahead / data leakage** — most plausibly a Kalman *smoother* (which uses future observations) instead of a *filter*.

So this repo rebuilds the method faithfully but under conditions where leakage is **impossible**, and asks: **do the gains survive?**

- The Kalman filter is **strictly causal** (forward filter only; no RTS smoother). This is unit-tested: perturbing a future observation leaves every past filtered value byte-identical (`tests/test_kalman_causal.py`).
- Evaluation is **out-of-sample** (chronological 80/20 split) and features are z-scored on **train statistics only**.
- PnL is always marked on the **raw** close; denoising only changes what the agent *sees*, never the prices it *trades*.

## Method

| | |
|---|---|
| Data | XAU/USD, H1, **47,302 bars** (2017-01-02 → 2024-12-31), Dukascopy, SHA-256 committed (the paper used 47,304) |
| Denoising | causal local-level Kalman filter on OHLC (`kdrl/kalman.py`) |
| State | 22 hand-written features (`kdrl/features.py`) + current position |
| Agents | DQN, PPO (stable-baselines3), Recurrent-PPO (sb3-contrib) |
| Env | 3 actions (short/flat/long); reward = pnl − cost − β·drawdown; 0.01% commission (`kdrl/env.py`) |
| Split | chronological 80/20, leakage-safe scaling |
| Metrics | cumulative return, CAGR, Sharpe (ann. √6048), max drawdown, volatility |

## Results (out-of-sample)

<!-- RESULTS -->
Test window = last 20% (~9.5k bars), 60k training steps/agent, single seed:

| Agent | Kalman | Cum. Return | CAGR | Sharpe | Max DD | Vol |
|---|---|---:|---:|---:|---:|---:|
| DQN | no | −12.50% | −8.19% | −0.655 | −18.45% | 11.95% |
| DQN | **yes** | −20.55% | −13.69% | −1.297 | −22.67% | 10.89% |
| PPO | no | −47.77% | −34.01% | −3.387 | −49.45% | 12.06% |
| PPO | **yes** | −20.53% | −13.68% | −1.155 | −27.44% | 12.10% |
| RPPO | no | −30.54% | −20.80% | −1.965 | −34.57% | 11.53% |
| RPPO | **yes** | −9.10% | −5.92% | −0.667 | −14.06% | 8.60% |

Buy & hold over the same window: **+34.84%**, Sharpe **1.50** (gold rallied hard through 2023–2024).

### What this shows

- **The headline does not survive a leakage-free setup.** The paper reports PPO+Kalman at **Sharpe 12.10**; under a strictly causal filter and a clean out-of-sample split, the best run here is RPPO+Kalman at **Sharpe −0.67**. Every agent loses money and trails passive buy-and-hold. A Sharpe of ~12 is the classic signature of look-ahead / leakage (most plausibly a Kalman *smoother* using future data, or train/test contamination) — exactly what the causality test in this repo makes impossible.
- **The Kalman step is a real but modest and mixed effect, not a universal 29×.** It helps the policy-gradient agents (PPO −3.39 → −1.16; RPPO −1.97 → −0.67) and *hurts* DQN (−0.66 → −1.30). Useful denoising for some agents, not a free Sharpe-12 machine for all.

### Caveats (kept honest)

- Modest budget (60k steps) and a single seed; RL is high-variance, so absolute numbers will move with more training / seed averaging. The qualitative gap to Sharpe 12 is structural, not a budget artifact.
- The test window is a strong gold uptrend, which flatters always-long buy-and-hold and penalises a tactical long/flat/short agent.
- The paper does not fully specify the 22 features, the Kalman Q/R, or all hyperparameters, so this is a faithful reconstruction, not a line-by-line port.

**Bottom line:** high confidence reproducing the *method*; the *headline metrics* are not reproducible under causal, out-of-sample conditions, and that gap is the result.
<!-- /RESULTS -->

## Reproduce

```bash
python -m venv .venv && .venv\Scripts\activate          # Python 3.12 (Windows)
pip install -r requirements.txt
python data/get_data.py            # pull XAU/USD H1 -> data/xauusd_h1.csv
python -m pytest tests -q          # 8 passing (incl. the causality proof)
python run_all.py --timesteps 100000   # full DQN/PPO/RPPO x Kalman matrix
# or a single run:
python train.py --algo ppo --kalman --timesteps 100000
```

## Layout

```
kdrl/        kalman.py · features.py · env.py · agents.py · evaluate.py · metrics.py · experiment.py
data/        get_data.py + xauusd_h1.csv (47,302 bars, committed)
tests/       test_kalman_causal.py (causality proof) · test_pipeline.py
train.py · run_all.py · results/  (comparison.csv + comparison.md)
```

## Honesty note on AI tools

Built with an AI coding assistant. It was useful for scaffolding the SB3 pipeline, and it was kept honest by primary sources and by the same instinct that drives this whole repo: a result that looks too good (here, Sharpe 12) is treated as a bug to be traced, not a win to be reported.

## License

MIT © 2026 Abhishek Kumar Gupta
