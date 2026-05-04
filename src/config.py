import os
from pathlib import Path

# Load .env from project root if present
_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        if _line.strip() and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

SOFR = 0.053          # ~current SOFR
N_LOANS = 200
TOTAL_NOTIONAL = 500_000_000   # $500M collateral pool

# Tranche definitions
TRANCHE_DEFS = [
    {"name": "AAA", "initial_balance": 200_000_000, "spread": 0.0145, "oc_threshold": 1.20},
    {"name": "AA",  "initial_balance":  60_000_000, "spread": 0.0215, "oc_threshold": 1.10},
    {"name": "A",   "initial_balance":  40_000_000, "spread": 0.0310, "oc_threshold": 1.06},
    {"name": "BBB", "initial_balance":  30_000_000, "spread": 0.0455, "oc_threshold": 1.03},
]

SENIOR_FEE_RATE = 0.0015   # 15 bps annually on total deal
IC_THRESHOLD    = 1.15

# Simulation parameters
N_PATHS   = 2000
N_PERIODS = 60    # 5 years monthly

# LSTM parameters
WINDOW_SIZE = 120   # ~6 months of business days (daily frequency)
LSTM_EPOCHS = 100
LSTM_LR     = 3e-4
BATCH_SIZE  = 64

# Paths
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT_DIR, "data")
MODEL_DIR   = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
