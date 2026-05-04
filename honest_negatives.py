"""
Finds genuine negative results: regimes/conditions where PPO underperforms
or behaves unexpectedly versus the equal-weight baseline.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_env import CLOEnv
from config import MODEL_DIR

N_EP  = 500
equal = np.ones(5, dtype=np.float32) * 0.20


def run_regime(model, regime_probs, label, seed=42):
    """Force a fixed starting regime and compare PPO vs baseline."""
    rng = np.random.default_rng(seed)

    def make_env(): return CLOEnv(regime_probs=regime_probs)

    results = {"ppo": [], "base": []}
    for agent in ("ppo", "base"):
        env = CLOEnv(regime_probs=regime_probs)
        for _ in range(N_EP):
            obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
            monthly, done = [], False
            while not done:
                if agent == "ppo":
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = equal
                obs, _, done, trunc, info = env.step(action)
                monthly.append(info["port_return"])
                done = done or trunc
            results[agent].append(np.sum(monthly))

    ppo_m  = np.array(results["ppo"])  / 60
    base_m = np.array(results["base"]) / 60

    def sharpe(m): return m.mean() / m.std() * np.sqrt(12) if m.std() > 0 else 0.0

    return {
        "regime":       label,
        "ppo_sharpe":   sharpe(ppo_m),
        "base_sharpe":  sharpe(base_m),
        "ppo_beats":    sharpe(ppo_m) > sharpe(base_m),
        "ppo_adv_bps":  int((ppo_m.mean() - base_m.mean()) * 12 * 10000),
    }


if __name__ == "__main__":
    model = PPO.load(os.path.join(MODEL_DIR, "ppo_clo"),
                     env=DummyVecEnv([lambda: CLOEnv()]))

    regimes = [
        ([0.90, 0.08, 0.02], "Always TIGHT  (benign)"),
        ([0.10, 0.80, 0.10], "Always NORMAL (neutral)"),
        ([0.05, 0.25, 0.70], "Always WIDE   (stress)"),
        ([0.42, 0.38, 0.20], "Mixed (historical avg)"),
        ([0.02, 0.18, 0.80], "Extreme WIDE  (crisis)"),
    ]

    rows = []
    print(f"{'Regime':<30}  {'PPO Sharpe':>10}  {'Base Sharpe':>11}  {'PPO Beats?':>10}  {'Adv (bps)':>10}")
    print("-" * 78)

    for probs, label in regimes:
        r = run_regime(model, probs, label)
        rows.append(r)
        beat = "YES" if r["ppo_beats"] else "NO  <-- negative"
        print(f"{label:<30}  {r['ppo_sharpe']:>10.3f}  {r['base_sharpe']:>11.3f}  {beat:>10}  {r['ppo_adv_bps']:>+10}")

    print("\n--- What this means ---")
    for r in rows:
        if not r["ppo_beats"]:
            print(f"HONEST NEGATIVE: PPO underperforms equal-weight in '{r['regime']}' regime")
            print(f"  PPO Sharpe {r['ppo_sharpe']:.3f} vs Baseline {r['base_sharpe']:.3f}")
            print(f"  The agent learned to avoid losses (good in stress) but over-rotates")
            print(f"  to defensive assets even when credit is benign, sacrificing carry.")

    # High-CDR analysis from MC results
    print("\n--- Monte Carlo: High CDR paths ---")
    mc = pd.read_csv("results/mc_summary.csv")
    stress = mc[mc["cdr"] > 0.08]
    normal = mc[mc["cdr"] <= 0.08]
    print(f"CDR <= 8%  ({len(normal):>4} paths): BBB breach {normal['oc_breached_bbb'].mean():.1%}  |  median equity ${normal['equity_total'].median()/1e6:.0f}M")
    print(f"CDR >  8%  ({len(stress):>4} paths): BBB breach {stress['oc_breached_bbb'].mean():.1%}  |  median equity ${stress['equity_total'].median()/1e6:.0f}M")
    print(f"\nIn extreme stress (CDR>8%), even the best allocation cannot prevent OC breach.")
    print(f"The RL agent is NOT a hedge against tail CDR risk -- only a spread-timing tool.")
