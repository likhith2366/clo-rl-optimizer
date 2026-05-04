"""
Multi-seed evaluation: runs baseline + PPO policy across 15 seeds.
Reports mean +/- std for every headline metric.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_agent import run_baseline
from rl_env import CLOEnv
from config import MODEL_DIR, RESULTS_DIR

SEEDS     = list(range(10, 25))   # 15 seeds
N_EVAL_EP = 200                   # episodes per seed (fast but stable)


def evaluate_policy_seed(model, seed):
    env = CLOEnv()
    rng = np.random.default_rng(seed)
    all_monthly, ep_dd = [], []
    for _ in range(N_EVAL_EP):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        monthly, dds, done = [], [], False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, trunc, info = env.step(action)
            monthly.append(info["port_return"])
            dds.append(info["drawdown"])
            done = done or trunc
        all_monthly.extend(monthly)
        ep_dd.append(max(dds))
    arr = np.array(all_monthly)
    sharpe = arr.mean() / arr.std() * np.sqrt(12) if arr.std() > 0 else 0.0
    return sharpe, np.mean(ep_dd)


def baseline_seed(seed):
    env = CLOEnv()
    rng = np.random.default_rng(seed)
    equal = np.ones(5, dtype=np.float32) * 0.20
    all_monthly, ep_dd = [], []
    for _ in range(N_EVAL_EP):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        monthly, dds, done = [], [], False
        while not done:
            obs, _, done, trunc, info = env.step(equal)
            monthly.append(info["port_return"])
            dds.append(info["drawdown"])
            done = done or trunc
        all_monthly.extend(monthly)
        ep_dd.append(max(dds))
    arr = np.array(all_monthly)
    sharpe = arr.mean() / arr.std() * np.sqrt(12) if arr.std() > 0 else 0.0
    return sharpe, np.mean(ep_dd)


if __name__ == "__main__":
    model = PPO.load(os.path.join(MODEL_DIR, "ppo_clo"),
                     env=DummyVecEnv([lambda: CLOEnv()]))

    base_sharpes, base_dds = [], []
    ppo_sharpes,  ppo_dds  = [], []

    print(f"Running {len(SEEDS)} seeds x {N_EVAL_EP} episodes each...")
    print(f"{'Seed':>5}  {'Base Sharpe':>12}  {'PPO Sharpe':>11}  {'Base DD':>8}  {'PPO DD':>8}")
    print("-" * 55)

    for seed in SEEDS:
        bs, bd = baseline_seed(seed)
        ps, pd_ = evaluate_policy_seed(model, seed)
        base_sharpes.append(bs); base_dds.append(bd)
        ppo_sharpes.append(ps);  ppo_dds.append(pd_)
        print(f"{seed:>5}  {bs:>12.3f}  {ps:>11.3f}  {bd:>7.2%}  {pd_:>7.2%}")

    bs_arr = np.array(base_sharpes); ps_arr = np.array(ppo_sharpes)
    bd_arr = np.array(base_dds);     pd_arr = np.array(ppo_dds)

    print("\n" + "=" * 55)
    print(f"{'Metric':<28}  {'Baseline':>12}  {'PPO':>10}")
    print("-" * 55)
    print(f"{'Sharpe  mean':<28}  {bs_arr.mean():>12.3f}  {ps_arr.mean():>10.3f}")
    print(f"{'Sharpe  std':<28}  {bs_arr.std():>12.3f}  {ps_arr.std():>10.3f}")
    print(f"{'Sharpe  min':<28}  {bs_arr.min():>12.3f}  {ps_arr.min():>10.3f}")
    print(f"{'Sharpe  max':<28}  {bs_arr.max():>12.3f}  {ps_arr.max():>10.3f}")
    print("-" * 55)
    print(f"{'Max DD  mean':<28}  {bd_arr.mean():>11.2%}  {pd_arr.mean():>9.2%}")
    print(f"{'Max DD  std':<28}  {bd_arr.std():>11.2%}  {pd_arr.std():>9.2%}")
    print("=" * 55)
    print(f"\nPPO beats baseline in {(ps_arr > bs_arr).sum()}/{len(SEEDS)} seeds")

    # Save
    out = pd.DataFrame({
        "seed": SEEDS,
        "base_sharpe": base_sharpes, "ppo_sharpe": ppo_sharpes,
        "base_dd": base_dds,         "ppo_dd": ppo_dds,
    })
    out.to_csv(os.path.join(RESULTS_DIR, "seed_variance.csv"), index=False)
    print(f"Saved -> results/seed_variance.csv")
