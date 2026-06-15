"""
kdrl — a replication of "Kalman-Enhanced Deep Reinforcement Learning for
Noise-Resilient Algorithmic Trading" (IJACSA, Vol. 16 No. 11).

The package denoises hourly XAU/USD with a strictly CAUSAL Kalman filter, builds
a feature state, and trains DQN / PPO / Recurrent-PPO agents in a trading
environment, with the Kalman step toggleable so its effect can be isolated.
"""
__version__ = "0.1.0"
