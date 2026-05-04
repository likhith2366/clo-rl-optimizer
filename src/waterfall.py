"""
CLO cashflow waterfall engine -- interest-only distribution model.

During the reinvestment period (the entire 5-year simulation horizon here),
principal proceeds from prepayments, amortisation, and loan maturities are
recycled back into new performing assets.  Note balances therefore remain
FIXED at their original values throughout the simulation.

Only loan defaults erode the performing balance.  Recovery cash (received
12 months after default) is also treated as reinvestable income.

OC test logic
-------------
After paying each tranche's interest, the engine tests:
    OC ratio = performing_balance / cumulative_note_balance   (fixed)

If the ratio falls below the tranche threshold, ALL remaining interest cash
is trapped (equity receives nothing) and the waterfall halts.  Note balances
are NOT mutated here; they are immutable during the reinvestment period.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import TRANCHE_DEFS, SOFR, SENIOR_FEE_RATE, IC_THRESHOLD, TOTAL_NOTIONAL

# Fixed note balances (read-only reference, never mutated in normal operation)
FIXED_NOTE_BALANCES = {d["name"]: float(d["initial_balance"]) for d in TRANCHE_DEFS}
FIXED_NOTE_COUPONS  = {d["name"]: SOFR + d["spread"]          for d in TRANCHE_DEFS}
OC_THRESHOLDS       = {d["name"]: d["oc_threshold"]           for d in TRANCHE_DEFS}


def apply_waterfall(
    interest_cash:      float,
    performing_balance: float,
    recovery_cash:      float = 0.0,
) -> dict:
    """
    Distribute one period's interest income through the waterfall.

    Note balances are FIXED (reinvestment-period model).  Only interest cash
    and recovery income flow through the waterfall.  Principal proceeds from
    prepayments / amortisation are assumed reinvested and do not appear here.

    Parameters
    ----------
    interest_cash      : coupon income from performing loans this period
    performing_balance : current par value of performing (non-defaulted) loans
    recovery_cash      : default recoveries arriving this period (12-month lag)

    Returns
    -------
    dict with period-level accounting (note balances unchanged)
    """
    result = {
        "senior_fees_paid": 0.0,
        "interest_paid":    {n: 0.0  for n in FIXED_NOTE_BALANCES},
        "oc_ratio":         {n: 0.0  for n in FIXED_NOTE_BALANCES},
        "oc_breached":      {n: False for n in FIXED_NOTE_BALANCES},
        "ic_breached":      False,
        "equity":           0.0,
        "trapped_cash":     0.0,   # redirected from equity on OC breach
    }

    cash = interest_cash + recovery_cash   # combined income pool

    # -- 1. Senior fees -------------------------------------------------------
    fees      = TOTAL_NOTIONAL * SENIOR_FEE_RATE / 12
    fees_paid = min(cash, fees)
    result["senior_fees_paid"] = fees_paid
    cash -= fees_paid

    # -- 2-5. Tranche interest + OC tests -------------------------------------
    cum_notes = 0.0
    tranche_order = [d["name"] for d in TRANCHE_DEFS]

    for name in tranche_order:
        balance    = FIXED_NOTE_BALANCES[name]
        cum_notes += balance

        # Pay interest
        interest_due  = balance * FIXED_NOTE_COUPONS[name] / 12
        interest_paid = min(cash, interest_due)
        result["interest_paid"][name] = interest_paid
        cash -= interest_paid

        # OC test: performing par vs cumulative fixed note balance
        oc_ratio = performing_balance / cum_notes if cum_notes > 0 else 999.0
        result["oc_ratio"][name] = oc_ratio

        if oc_ratio < OC_THRESHOLDS[name]:
            result["oc_breached"][name] = True
            result["trapped_cash"] = cash   # equity locked out; cash held
            result["equity"] = 0.0
            return result

    # -- 6. IC test -----------------------------------------------------------
    total_int_due = sum(
        FIXED_NOTE_BALANCES[n] * FIXED_NOTE_COUPONS[n] / 12 for n in tranche_order
    )
    ic_ratio = (interest_cash / total_int_due) if total_int_due > 0 else 999.0
    if ic_ratio < IC_THRESHOLD:
        result["ic_breached"] = True
        result["trapped_cash"] = cash
        result["equity"] = 0.0
        return result

    # -- 7. Equity residual ---------------------------------------------------
    result["equity"] = max(0.0, cash)
    return result


def check_oc_trigger(performing_balance: float, tranche_name: str) -> bool:
    """Standalone OC check -- True means triggered (breached)."""
    cum = sum(
        FIXED_NOTE_BALANCES[n]
        for n in [d["name"] for d in TRANCHE_DEFS]
        if TRANCHE_DEFS[[d["name"] for d in TRANCHE_DEFS].index(n)]["oc_threshold"]
           <= OC_THRESHOLDS[tranche_name]
    )
    oc_ratio = performing_balance / cum if cum > 0 else 999.0
    return oc_ratio < OC_THRESHOLDS[tranche_name]


def fresh_tranches():
    """Backward-compat shim -- returns read-only tranche info dicts."""
    return [
        {
            "name":            d["name"],
            "initial_balance": d["initial_balance"],
            "outstanding":     float(d["initial_balance"]),   # fixed, not mutated
            "spread":          d["spread"],
            "oc_threshold":    d["oc_threshold"],
            "coupon":          SOFR + d["spread"],
        }
        for d in TRANCHE_DEFS
    ]
