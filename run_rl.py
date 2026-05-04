"""
Step 3 -- Train PPO agent and compare against equal-weight baseline.

Outputs
-------
models/ppo_clo.zip          -- trained PPO policy
results/rl_baseline.csv     -- episode returns for equal-weight strategy
results/rl_policy.csv       -- episode returns for PPO policy
results/rl_training_log.csv -- per-episode reward during training
results/rl_summary.txt      -- headline metrics (Sharpe, drawdown, convergence ep)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
from config import RESULTS_DIR, MODEL_DIR
from rl_agent import run_baseline, train_ppo, evaluate_policy, save_policy

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,   exist_ok=True)

if __name__ == "__main__":
    print("=" * 60)
    print("PPO PORTFOLIO ALLOCATION AGENT")
    print("=" * 60)

    # ── 1. Baseline ───────────────────────────────────────────────────────────
    print("\n[1/3] Running equal-weight baseline (500 episodes)…")
    baseline_sharpe, baseline_dd, baseline_ep_returns = run_baseline(n_episodes=500)

    baseline_df = pd.DataFrame({"episode_return": baseline_ep_returns})
    baseline_df.to_csv(os.path.join(RESULTS_DIR, "rl_baseline.csv"), index=False)

    # ── 2. Train PPO ──────────────────────────────────────────────────────────
    print("\n[2/3] Training PPO agent (200 episodes x 2048 steps)...")
    model, logger = train_ppo(n_steps=2048, n_episodes=200, verbose=0, log_every=200)

    save_policy(model)

    # Training log
    log_df = pd.DataFrame({"episode_reward": logger.episode_rewards})
    log_df.to_csv(os.path.join(RESULTS_DIR, "rl_training_log.csv"), index=False)

    # ── 3. Evaluate ───────────────────────────────────────────────────────────
    print("\n[3/3] Evaluating trained policy (500 episodes)…")
    eval_results = evaluate_policy(model, n_episodes=500)

    policy_df = pd.DataFrame({"episode_return": eval_results["episode_returns"]})
    policy_df.to_csv(os.path.join(RESULTS_DIR, "rl_policy.csv"), index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_game_eps = len(logger.episode_rewards)
    conv_ep = logger.convergence_ep if logger.convergence_ep > 0 else total_game_eps
    summary = (
        f"BASELINE  Sharpe: {baseline_sharpe:.2f}   Avg max drawdown: {baseline_dd:.2%}\n"
        f"PPO       Sharpe: {eval_results['sharpe']:.2f}   "
        f"Avg max drawdown: {eval_results['max_drawdown']:.2%}\n"
        f"Convergence episode: {conv_ep}\n"
    )
    print("\n" + "-" * 50)
    print(summary)

    summary_path = os.path.join(RESULTS_DIR, "rl_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Saved -> {summary_path}")

    print("\nAll RL artefacts saved. Run `notebooks/results.ipynb` to generate plots.")
