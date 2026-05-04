import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import N_LOANS, TOTAL_NOTIONAL


def generate_loan_pool(n_loans: int = N_LOANS, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic pool of leveraged loans representing CLO collateral.

    Returns a DataFrame with one row per loan. All monetary values in dollars.
    """
    rng = np.random.default_rng(seed)

    # Principals drawn uniformly, then scaled to hit target notional
    raw = rng.uniform(1_000_000, 50_000_000, n_loans)
    principals = raw / raw.sum() * TOTAL_NOTIONAL

    # Floating-rate coupons: SOFR (~5.3%) + credit spread 80-580bps
    coupons = rng.uniform(0.06, 0.11, n_loans)

    # Loan maturities 3-7 years, expressed in months
    maturity_months = np.round(rng.uniform(3, 7, n_loans) * 12).astype(int)

    ratings = rng.choice(["BB", "B", "CCC"], n_loans, p=[0.50, 0.35, 0.15])

    return pd.DataFrame({
        "loan_id":             np.arange(1, n_loans + 1),
        "original_principal":  principals,
        "remaining_principal": principals.copy(),
        "coupon":              coupons,
        "maturity_months":     maturity_months,
        "months_outstanding":  np.zeros(n_loans, dtype=int),
        "rating":              ratings,
        "status":              "performing",   # performing | defaulted | prepaid | matured
    })
