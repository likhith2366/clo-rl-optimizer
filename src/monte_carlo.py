"""
Monte Carlo runner -- 2,000 paths x 60 monthly periods.

Aggregate pool model (reinvestment-period CLO):
  - Performing balance starts at $500M
  - Each month: only CDR-driven defaults erode performing balance
  - Prepayments and amortisation are reinvested (pool stays full minus defaults)
  - Recovery arrives 12 months after default, treated as additional income
  - Note balances are FIXED at original levels throughout
  - OC test: performing / fixed_note_balance vs threshold

CDR distribution: lognormal(mu=ln(2.5%), sigma=0.85), clipped [0.5%, 15%].
Median CDR ~2.5%; only ~6-9% of paths exceed 8% (genuine-crisis territory).

Target outcomes: ~10-15% overall BBB OC breach rate; ~70-90% breach rate
among the subset of paths with CDR > 8% (those paths are well into breach zone).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import N_PATHS, N_PERIODS, TRANCHE_DEFS, TOTAL_NOTIONAL
from waterfall import apply_waterfall, FIXED_NOTE_BALANCES, FIXED_NOTE_COUPONS


AVG_COUPON = 0.085   # approximate blended loan coupon


def run_single_path(cdr, recovery_rate, cpr, rng):
    performing     = float(TOTAL_NOTIONAL)
    recovery_queue = {}
    equity_flows   = []
    monthly_oc_bbb = []
    any_oc_breach  = {d["name"]: False for d in TRANCHE_DEFS}

    monthly_dp = 1.0 - (1.0 - cdr) ** (1.0 / 12)

    for period in range(N_PERIODS):
        # Stochastic defaults: binomial pool model, N equal-weight loans.
        # std = performing * sqrt(p*(1-p)/N)  — correct dollar-variance formula.
        n_loans         = 200
        expected_def    = performing * monthly_dp
        std_def         = performing * np.sqrt(monthly_dp * (1.0 - monthly_dp) / n_loans)
        actual_defaults = float(np.clip(rng.normal(expected_def, std_def), 0.0, performing))

        performing -= actual_defaults
        recovery_queue[period + 12] = recovery_queue.get(period + 12, 0.0) + actual_defaults * recovery_rate

        recovery_cash = recovery_queue.pop(period, 0.0)
        interest_cash = performing * AVG_COUPON / 12

        result = apply_waterfall(
            interest_cash      = interest_cash,
            performing_balance = performing,
            recovery_cash      = recovery_cash,
        )

        for name, breached in result["oc_breached"].items():
            if breached:
                any_oc_breach[name] = True

        cum_notes = sum(FIXED_NOTE_BALANCES.values())
        bbb_oc    = performing / cum_notes if cum_notes > 0 else 999.0
        monthly_oc_bbb.append(bbb_oc)
        equity_flows.append(result["equity"])

    return {
        "cdr":             cdr,
        "recovery_rate":   recovery_rate,
        "cpr":             cpr,
        "oc_breached_aaa": any_oc_breach["AAA"],
        "oc_breached_aa":  any_oc_breach["AA"],
        "oc_breached_a":   any_oc_breach["A"],
        "oc_breached_bbb": any_oc_breach["BBB"],
        "min_bbb_oc":      min(monthly_oc_bbb),
        "final_bbb_oc":    monthly_oc_bbb[-1],
        "equity_total":    sum(equity_flows),
        "monthly_oc_bbb":  monthly_oc_bbb,
        "equity_flows":    equity_flows,
    }


def run_monte_carlo(n_paths=N_PATHS, seed=0):
    rng = np.random.default_rng(seed)

    # Lognormal CDR: median ~3%, right-skewed toward realistic low values.
    # mean=-3.5 => median=exp(-3.5)=3.0%; sigma=0.7 => ~8-10% of paths above 8%.
    CDR_MU    = -3.5
    CDR_SIGMA =  0.7

    all_paths = []
    for _ in tqdm(range(n_paths), desc="Monte Carlo paths"):
        cdr           = float(np.clip(rng.lognormal(CDR_MU, CDR_SIGMA), 0.005, 0.15))
        recovery_rate = rng.uniform(0.30, 0.70)
        cpr           = rng.uniform(0.10, 0.25)
        path = run_single_path(cdr, recovery_rate, cpr, rng)
        all_paths.append(path)

    scalar_keys = [
        "cdr", "recovery_rate", "cpr",
        "oc_breached_aaa", "oc_breached_aa", "oc_breached_a", "oc_breached_bbb",
        "min_bbb_oc", "final_bbb_oc", "equity_total",
    ]
    summary_df = pd.DataFrame([{k: p[k] for k in scalar_keys} for p in all_paths])

    overall_bbb_breach = summary_df["oc_breached_bbb"].mean()
    stress_mask        = summary_df["cdr"] > 0.08
    stress_breach      = summary_df.loc[stress_mask, "oc_breached_bbb"].mean()

    print(f"\n{'-'*50}")
    print(f"Paths simulated            : {n_paths:,}")
    print(f"BBB OC breach rate (all)   : {overall_bbb_breach:.1%}")
    print(f"BBB OC breach rate (CDR>8%): {stress_breach:.1%}")
    print(f"Median equity total        : ${summary_df['equity_total'].median():,.0f}")
    print(f"{'-'*50}\n")

    return summary_df, all_paths
