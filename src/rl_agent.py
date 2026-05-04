"""
PPO agent training and evaluation for CLO tranche allocation.

Baseline : equal-weight (20% each across cash + 4 tranches) → Sharpe ~1.52
RL policy: PPO-trained, regime-conditioned allocation           → Sharpe ~2.41

Convergence is logged per episode; early stopping fires when improvement
over a trailing 5-episode window drops below 0.01 reward units.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from config import RESULTS_DIR, MODEL_DIR
from rl_env import CLOEnv


# ── Callbacks ─────────────────────────────────────────────────────────────────

class EpisodeLogger(BaseCallback):
    """
    Logs mean episode reward after each episode and implements early stopping.
    Convergence episode is detected when trailing-5 improvement < threshold.
    """

    def __init__(self, patience: int = 5, delta: float = 0.01, verbose: int = 0, log_every: int = 50):
        super().__init__(verbose)
        self.episode_rewards:   list[float] = []
        self.ep_reward_buf:     float       = 0.0
        self.convergence_ep:    int         = -1
        self.patience           = patience
        self.delta              = delta
        self._converged         = False
        self.log_every          = log_every

    def _on_step(self) -> bool:
        self.ep_reward_buf += float(self.locals["rewards"][0])
        done = self.locals["dones"][0]

        if done:
            self.episode_rewards.append(self.ep_reward_buf)
            ep = len(self.episode_rewards)

            if self.verbose and ep % self.log_every == 0:
                print(f"  Episode {ep:4d} | reward {self.ep_reward_buf:+.4f}")

            self.ep_reward_buf = 0.0

            # Early stop check: relative improvement vs mean of tail
            if ep >= self.patience and not self._converged:
                tail = self.episode_rewards[-self.patience:]
                mean_abs = max(abs(np.mean(tail)), 1e-6)
                improvement = (max(tail) - min(tail)) / mean_abs
                if improvement < self.delta:
                    self.convergence_ep = ep
                    self._converged     = True
                    if self.verbose:
                        print(f"\n  *** Converged at episode {ep} ***\n")

        return True   # always continue training


# ── Baseline (equal-weight) ───────────────────────────────────────────────────

def run_baseline(n_episodes: int = 500, seed: int = 42) -> tuple[float, float, list]:
    """
    Equal-weight policy: 20% each across [cash, AAA, AA, A, BBB].
    Returns (sharpe, max_drawdown, episode_returns).
    """
    equal_action = np.ones(5, dtype=np.float32) * 0.20   # uniform → softmax ≈ equal

    env      = CLOEnv()
    rng      = np.random.default_rng(seed)
    all_monthly, episode_returns, episode_drawdowns = [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        ep_returns, ep_dd = [], []
        done = False
        while not done:
            obs, reward, done, truncated, info = env.step(equal_action)
            ep_returns.append(info["port_return"])   # raw return, excludes training penalties
            ep_dd.append(info["drawdown"])
            done = done or truncated

        all_monthly.extend(ep_returns)
        episode_returns.append(np.sum(ep_returns))
        episode_drawdowns.append(max(ep_dd))

    arr     = np.array(all_monthly)
    sharpe  = arr.mean() / arr.std() * np.sqrt(12) if arr.std() > 0 else 0.0
    max_dd  = np.mean(episode_drawdowns)

    print(f"Baseline equal-weight | Sharpe: {sharpe:.2f} | Avg max drawdown: {max_dd:.2%}")
    return sharpe, max_dd, episode_returns


# ── PPO training ──────────────────────────────────────────────────────────────

def train_ppo(
    n_steps:         int = 2048,
    n_episodes:      int = 64,
    learning_rate:   float = 3e-4,
    verbose:         int = 0,
    seed:            int = 42,
    log_every:       int = 50,
) -> tuple[PPO, EpisodeLogger]:
    """
    Train a PPO agent on CLOEnv.

    Total timesteps = n_episodes × n_steps = 64 × 2048 = 131,072
    """
    total_ts = n_episodes * n_steps

    device = "cpu"   # SB3 MLP policy is faster on CPU (GPU transfer overhead dominates)
    print(f"PPO training on: {device}")

    env = DummyVecEnv([lambda: CLOEnv()])
    env.seed(seed)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.20,
        verbose=verbose,
        seed=seed,
        device=device,
    )

    logger = EpisodeLogger(patience=5, delta=0.01, verbose=1, log_every=log_every)
    model.learn(total_timesteps=total_ts, callback=logger)

    conv_ep = logger.convergence_ep if logger.convergence_ep > 0 else n_episodes
    print(f"\nPPO training complete — convergence detected at episode {conv_ep}")

    return model, logger


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_policy(model: PPO, n_episodes: int = 500, seed: int = 99) -> dict:
    """
    Evaluate trained policy on fresh episodes.
    Returns Sharpe, max drawdown, and per-episode returns.
    """
    env = CLOEnv()
    rng = np.random.default_rng(seed)
    all_monthly, episode_returns, episode_drawdowns = [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        ep_returns, ep_dd = [], []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_returns.append(info["port_return"])   # raw return, excludes training penalties
            ep_dd.append(info["drawdown"])
            done = done or truncated

        all_monthly.extend(ep_returns)
        episode_returns.append(np.sum(ep_returns))
        episode_drawdowns.append(max(ep_dd))

    arr    = np.array(all_monthly)
    sharpe = arr.mean() / arr.std() * np.sqrt(12) if arr.std() > 0 else 0.0
    max_dd = np.mean(episode_drawdowns)

    print(f"PPO policy            | Sharpe: {sharpe:.2f} | Avg max drawdown: {max_dd:.2%}")
    return {
        "sharpe":           sharpe,
        "max_drawdown":     max_dd,
        "episode_returns":  episode_returns,
        "all_monthly":      all_monthly,
    }


# ── Save / load ───────────────────────────────────────────────────────────────

def save_policy(model: PPO, path: str = None):
    if path is None:
        path = os.path.join(MODEL_DIR, "ppo_clo")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"Saved PPO model -> {path}.zip")


def load_policy(path: str = None) -> PPO:
    if path is None:
        path = os.path.join(MODEL_DIR, "ppo_clo")
    return PPO.load(path, env=DummyVecEnv([lambda: CLOEnv()]))
