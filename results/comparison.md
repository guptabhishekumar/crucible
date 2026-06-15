# Results: Kalman-enhanced DRL on XAU/USD H1 (out-of-sample)

Bars: 47,302 | train/test split 80/20 | timesteps/agent 60,000 | cost 0.01% | reward = pnl - cost - beta*drawdown.

| Agent | Kalman | CumRet% | CAGR% | Sharpe | MaxDD% | Vol% |
|---|---|---|---|---|---|---|
| DQN | no | -12.5 | -8.19 | -0.655 | -18.45 | 11.95 |
| DQN | yes | -20.55 | -13.69 | -1.297 | -22.67 | 10.89 |
| PPO | no | -47.77 | -34.01 | -3.387 | -49.45 | 12.06 |
| PPO | yes | -20.53 | -13.68 | -1.155 | -27.44 | 12.1 |
| RPPO | no | -30.54 | -20.8 | -1.965 | -34.57 | 11.53 |
| RPPO | yes | -9.1 | -5.92 | -0.667 | -14.06 | 8.6 |

Buy & hold (test window): +34.84% cumulative, Sharpe 1.496, max drawdown -8.79%.
