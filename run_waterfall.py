"""
Step 1 -- Run Monte Carlo waterfall simulation and save results.

Outputs
-------
results/mc_summary.csv   -- one row per path (scalar metrics)
results/mc_oc_series.npy -- (N_PATHS, N_PERIODS) BBB OC ratio time series
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
from config import N_PATHS, N_PERIODS, RESULTS_DIR
from monte_carlo import run_monte_carlo

os.makedirs(RESULTS_DIR, exist_ok=True)

if __name__ == "__main__":
    print("=" * 60)
    print("CLO WATERFALL MONTE CARLO  --  2,000 paths × 60 periods")
    print("=" * 60)

    summary_df, all_paths = run_monte_carlo(n_paths=N_PATHS, seed=0)

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "mc_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved -> {summary_path}")

    # Save OC series for plotting
    oc_matrix = np.array([p["monthly_oc_bbb"] for p in all_paths])  # (2000, 60)
    oc_path   = os.path.join(RESULTS_DIR, "mc_oc_series.npy")
    np.save(oc_path, oc_matrix)
    print(f"Saved -> {oc_path}")

    # Equity cashflow matrix
    eq_matrix = np.array([p["equity_flows"] for p in all_paths])     # (2000, 60)
    eq_path   = os.path.join(RESULTS_DIR, "mc_equity_flows.npy")
    np.save(eq_path, eq_matrix)
    print(f"Saved -> {eq_path}")

    print("\nDone. Run `run_lstm.py` next.")
