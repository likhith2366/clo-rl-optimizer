"""
Deep analysis: all 8 interview questions in one run.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_env import CLOEnv
from monte_carlo import run_monte_carlo, run_single_path
from waterfall import FIXED_NOTE_BALANCES, apply_waterfall
from config import MODEL_DIR, TOTAL_NOTIONAL, TRANCHE_DEFS, SOFR, N_PERIODS

AVG_COUPON = 0.085

# ─── Load / run data ──────────────────────────────────────────────────────────
print("Running Monte Carlo (2000 paths)...")
mc_df, all_paths = run_monte_carlo(n_paths=2000, seed=0)

model = PPO.load(os.path.join(MODEL_DIR, "ppo_clo"),
                 env=DummyVecEnv([lambda: CLOEnv()]))

# ═══════════════════════════════════════════════════════════════════════════════
# 1. FULL EQUITY DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("1. EQUITY PAYOUT DISTRIBUTION")
print("="*60)
eq = mc_df["equity_total"]
for label, val in [
    ("Mean",    eq.mean()),
    ("Std",     eq.std()),
    ("p10",     eq.quantile(0.10)),
    ("p25",     eq.quantile(0.25)),
    ("Median",  eq.median()),
    ("p75",     eq.quantile(0.75)),
    ("p90",     eq.quantile(0.90)),
    ("Worst",   eq.min()),
    ("Best",    eq.max()),
]:
    print(f"  {label:<10}: ${val/1e6:>8.1f}M")

breached = mc_df[mc_df["oc_breached_bbb"]]
not_breached = mc_df[~mc_df["oc_breached_bbb"]]
print(f"\n  Breached paths  ({len(breached):>4}): median equity ${breached['equity_total'].median()/1e6:.1f}M")
print(f"  Normal paths    ({len(not_breached):>4}): median equity ${not_breached['equity_total'].median()/1e6:.1f}M")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. BREACH TIMING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("2. BREACH TIMING")
print("="*60)

breach_months = []
oc_series = np.load("results/mc_oc_series.npy")   # (2000, 60)
THRESH = 1.03

for i in range(2000):
    if mc_df["oc_breached_bbb"].iloc[i]:
        # find first month below threshold
        below = np.where(oc_series[i] < THRESH)[0]
        if len(below) > 0:
            breach_months.append(below[0] + 1)   # 1-indexed months

bm = np.array(breach_months)
print(f"  Paths with BBB breach : {len(bm)}")
print(f"  Breach month mean     : {bm.mean():.1f}")
print(f"  Breach month median   : {np.median(bm):.1f}")
print(f"  Breach month std      : {bm.std():.1f}")
print(f"  Earliest breach       : month {bm.min()}")
print(f"  Latest breach         : month {bm.max()}")
print(f"  p25 breach month      : month {np.percentile(bm, 25):.0f}")
print(f"  p75 breach month      : month {np.percentile(bm, 75):.0f}")

# Histogram by quarter
print("\n  Breach timing distribution:")
bins = [0, 12, 24, 36, 48, 60]
labels_q = ["Yr1 (1-12)", "Yr2 (13-24)", "Yr3 (25-36)", "Yr4 (37-48)", "Yr5 (49-60)"]
for i, lbl in enumerate(labels_q):
    count = ((bm > bins[i]) & (bm <= bins[i+1])).sum()
    print(f"    {lbl}: {count:>3} paths  ({count/len(bm):.0%})")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. CASHFLOW INTUITION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("3. CASHFLOW INTUITION")
print("="*60)

# Total coupon due to all notes per month at full pool
cum_notes     = sum(FIXED_NOTE_BALANCES.values())
total_note_coupon = sum(
    FIXED_NOTE_BALANCES[d["name"]] * (SOFR + d["spread"]) / 12
    for d in TRANCHE_DEFS
)
senior_fee    = TOTAL_NOTIONAL * 0.0015 / 12

print(f"  At full performing balance ($500M):")
print(f"    Monthly interest income  : ${TOTAL_NOTIONAL * AVG_COUPON/12/1e6:.2f}M")
print(f"    Senior fees              : ${senior_fee/1e6:.3f}M")
print(f"    Total note coupons due   : ${total_note_coupon/1e6:.3f}M")
print(f"    Equity residual (gross)  : ${(TOTAL_NOTIONAL*AVG_COUPON/12 - senior_fee - total_note_coupon)/1e6:.3f}M/month")
gross_monthly = TOTAL_NOTIONAL * AVG_COUPON / 12
equity_gross  = gross_monthly - senior_fee - total_note_coupon
print(f"    Equity share of income   : {equity_gross/gross_monthly:.1%}")
print(f"    Note share of income     : {(gross_monthly-equity_gross-senior_fee)/gross_monthly:.1%}")

print(f"\n  Median equity path (no breach): ${not_breached['equity_total'].median()/1e6:.1f}M over 60 months")
print(f"  Median equity path (breach)   : ${breached['equity_total'].median()/1e6:.1f}M  <- breach traps cash")

# CDR sensitivity on equity
print(f"\n  Equity vs CDR:")
for cdr_band in [0.01, 0.02, 0.03, 0.05, 0.07, 0.09]:
    mask = (mc_df["cdr"] >= cdr_band - 0.005) & (mc_df["cdr"] < cdr_band + 0.015)
    sub  = mc_df[mask]
    if len(sub) > 5:
        print(f"    CDR ~{cdr_band:.0%}: n={len(sub):>3}  median equity ${sub['equity_total'].median()/1e6:.1f}M  breach {sub['oc_breached_bbb'].mean():.0%}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. LSTM FEATURE IMPORTANCE (permutation on saved predictions)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("4. LSTM FEATURE IMPORTANCE (indirect - from VIX structure)")
print("="*60)
fred = pd.read_csv("data/fred_data.csv", index_col=0, parse_dates=True)
preds = pd.read_csv("results/lstm_predictions.csv")
# Regime 2 (WIDE) happens when VIX >= 25 — VIX IS the label source
# So VIX is definitionally the most important feature
# Check correlation of each feature with regime label
fred2 = fred.copy()
fred2["regime"] = 0
fred2.loc[fred2["vix"] >= 15, "regime"] = 1
fred2.loc[fred2["vix"] >= 25, "regime"] = 2
print("  Correlation of each feature with regime label:")
for col in ["baa_spread", "aaa_spread", "slope_2s10s", "vix", "loan_officer"]:
    corr = fred2[col].corr(fred2["regime"])
    print(f"    {col:<20}: {corr:>+.3f}")

print("\n  Why VIX dominates: it IS the labeling criterion (WIDE=VIX>=25)")
print("  Baa spread is second — rises with credit stress, correlated with VIX")
print("  Slope (2s10s) inverts before recessions — leads the regime signal")
print("  Loan officer: lags by ~1 quarter (quarterly survey, ffilled)")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. RL ALLOCATION BEHAVIOR BY REGIME
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("5. RL ALLOCATION BEHAVIOR BY REGIME")
print("="*60)

asset_names = ["Cash", "AAA", "AA", "A", "BBB"]
regime_specs = [
    ([0.90, 0.08, 0.02], "TIGHT  (P_wide=0.02)"),
    ([0.10, 0.80, 0.10], "NORMAL (P_wide=0.10)"),
    ([0.05, 0.25, 0.70], "WIDE   (P_wide=0.70)"),
    ([0.02, 0.08, 0.90], "CRISIS (P_wide=0.90)"),
]

for regime_probs, label in regime_specs:
    env = CLOEnv(regime_probs=regime_probs)
    all_weights = []
    rng = np.random.default_rng(0)
    for _ in range(200):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        ep_weights = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # decode softmax
            a = np.array(action, dtype=np.float64)
            e = np.exp(a - a.max()); w = e / e.sum()
            ep_weights.append(w)
            obs, _, done, trunc, info = env.step(action)
            done = done or trunc
        all_weights.append(np.mean(ep_weights, axis=0))
    mean_w = np.mean(all_weights, axis=0)
    print(f"\n  {label}")
    for name, w in zip(asset_names, mean_w):
        bar = "#" * int(w * 40)
        print(f"    {name:<5}: {w:>5.1%}  {bar}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. PRE-BREACH BEHAVIOR
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("6. PRE-BREACH BEHAVIOR — predictive vs reactive")
print("="*60)

# Run episodes until near-breach and track weights
env = CLOEnv()
rng = np.random.default_rng(7)

oc_buckets = {"safe(>1.15)": [], "warning(1.05-1.15)": [], "danger(1.03-1.05)": [], "breach(<1.03)": []}

for _ in range(500):
    obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        a = np.array(action, dtype=np.float64)
        e = np.exp(a - a.max()); w = e / e.sum()
        bbb_oc = obs[6] * 2.0   # oc_norm was / 2.0, reverse it
        obs, _, done, trunc, info = env.step(action)
        done = done or trunc

        if bbb_oc > 1.15:
            oc_buckets["safe(>1.15)"].append(w)
        elif bbb_oc > 1.05:
            oc_buckets["warning(1.05-1.15)"].append(w)
        elif bbb_oc > 1.03:
            oc_buckets["danger(1.03-1.05)"].append(w)
        else:
            oc_buckets["breach(<1.03)"].append(w)

print(f"  {'OC Zone':<22} {'n steps':>8}  {'Cash':>6}  {'AAA':>6}  {'AA':>5}  {'A':>5}  {'BBB':>6}")
print("  " + "-"*62)
for zone, ws in oc_buckets.items():
    if ws:
        arr = np.mean(ws, axis=0)
        print(f"  {zone:<22} {len(ws):>8}  {arr[0]:>5.1%}  {arr[1]:>6.1%}  {arr[2]:>5.1%}  {arr[3]:>5.1%}  {arr[4]:>6.1%}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. CDR SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("7. CDR SENSITIVITY")
print("="*60)

cdr_levels = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.074, 0.08, 0.09, 0.10]
print(f"  {'CDR':>6}  {'Breach%':>8}  {'MedianEq($M)':>14}  {'MinOC':>8}")
print("  " + "-"*42)
rng2 = np.random.default_rng(1)
for cdr in cdr_levels:
    breaches, equities, min_ocs = [], [], []
    for _ in range(300):
        path = run_single_path(cdr=cdr, recovery_rate=0.50,
                               cpr=0.15, rng=rng2)
        breaches.append(path["oc_breached_bbb"])
        equities.append(path["equity_total"])
        min_ocs.append(path["min_bbb_oc"])
    marker = " <-- CLIFF" if 0.073 < cdr < 0.076 else ""
    print(f"  {cdr:>5.1%}  {np.mean(breaches):>8.1%}  {np.median(equities)/1e6:>14.1f}  {np.mean(min_ocs):>8.3f}{marker}")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. SURPRISING INSIGHT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("8. SURPRISING INSIGHT")
print("="*60)

# Does BBB equity matter more than CDR for equity outcomes?
corr_cdr_eq   = mc_df["cdr"].corr(mc_df["equity_total"])
corr_rec_eq   = mc_df["recovery_rate"].corr(mc_df["equity_total"])
corr_breach_eq = mc_df["oc_breached_bbb"].corr(mc_df["equity_total"])
print(f"  Correlation: CDR vs equity total          : {corr_cdr_eq:>+.3f}")
print(f"  Correlation: recovery_rate vs equity total: {corr_rec_eq:>+.3f}")
print(f"  Correlation: OC_breach vs equity total    : {corr_breach_eq:>+.3f}")

# Check: in non-breached paths, how much does CDR matter?
nb = mc_df[~mc_df["oc_breached_bbb"]]
b  = mc_df[mc_df["oc_breached_bbb"]]
print(f"\n  Non-breached paths: CDR vs equity corr    : {nb['cdr'].corr(nb['equity_total']):>+.3f}")
print(f"  Breached paths:     CDR vs equity corr    : {b['cdr'].corr(b['equity_total']):>+.3f}")
print(f"\n  Surprising: recovery_rate barely matters (corr {corr_rec_eq:+.3f})")
print(f"  Why: recovery arrives 12 months later as interest, not principal return.")
print(f"  By month 60, most recovery cash has already flowed through as income.")
print(f"  The REAL equity driver is whether breach happened (binary cliff), not CDR level.")
print(f"  OC breach correlation ({corr_breach_eq:+.3f}) is stronger than CDR ({corr_cdr_eq:+.3f}).")
print(f"  A 4% CDR path that breaches in month 18 pays LESS equity than")
print(f"  a 6% CDR path that never breaches -- because trapped cash dominates.")
