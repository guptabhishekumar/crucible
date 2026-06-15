"""
Agent factory — DQN, PPO (stable-baselines3) and Recurrent-PPO (sb3-contrib),
the three architectures the paper evaluates. Hyperparameters follow the paper
where stated (e.g. PPO learning rate 3e-4) and use sensible defaults otherwise.
"""
from __future__ import annotations

from stable_baselines3 import DQN, PPO
from sb3_contrib import RecurrentPPO

ALGOS = ("dqn", "ppo", "rppo")


def build_agent(algo: str, env, seed: int = 0, verbose: int = 0):
    algo = algo.lower()
    if algo == "ppo":
        return PPO(
            "MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=256,
            gamma=0.99, gae_lambda=0.95, ent_coef=0.0, seed=seed, verbose=verbose,
        )
    if algo == "dqn":
        return DQN(
            "MlpPolicy", env, learning_rate=1e-4, buffer_size=100_000,
            learning_starts=1_000, batch_size=128, gamma=0.99, train_freq=4,
            target_update_interval=1_000, exploration_fraction=0.2,
            exploration_final_eps=0.05, seed=seed, verbose=verbose,
        )
    if algo == "rppo":
        return RecurrentPPO(
            "MlpLstmPolicy", env, learning_rate=3e-4, n_steps=1024, batch_size=256,
            gamma=0.99, gae_lambda=0.95, ent_coef=0.0, seed=seed, verbose=verbose,
        )
    raise ValueError(f"unknown algo: {algo!r} (choose from {ALGOS})")
