"""
Master runner -- executes all three pipeline stages in order.

Run with:  py -3.13 run_all.py
Or stages: py -3.13 run_waterfall.py  /  py -3.13 run_lstm.py  /  py -3.13 run_rl.py
"""
import subprocess
import sys
import os


def _find_python():
    import importlib
    needed = ["numpy", "torch", "gymnasium", "stable_baselines3", "fredapi"]
    if all(importlib.util.find_spec(p) for p in needed):
        return [sys.executable]
    return ["py", "-3.13"]


STAGES = [
    ("Stage 1 -- Waterfall Monte Carlo", "run_waterfall.py"),
    ("Stage 2 -- LSTM Regime Classifier", "run_lstm.py"),
    ("Stage 3 -- PPO RL Agent",           "run_rl.py"),
]

if __name__ == "__main__":
    root   = os.path.dirname(os.path.abspath(__file__))
    python = _find_python()

    for label, script in STAGES:
        print("\n" + "=" * 65)
        print(f"  {label}")
        print("=" * 65)
        result = subprocess.run(python + [os.path.join(root, script)], cwd=root)
        if result.returncode != 0:
            print(f"\nERROR: {script} exited with code {result.returncode}. Stopping.")
            sys.exit(result.returncode)

    print("\n" + "=" * 65)
    print("  ALL STAGES COMPLETE")
    print("  Open notebooks/results.ipynb to view plots and metrics.")
    print("=" * 65)
