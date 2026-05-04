"""
Step 2 -- Fetch FRED data and train the LSTM regime classifier.

Outputs
-------
data/fred_data.csv          -- raw monthly macro features
models/lstm_model.pt        -- best model weights
models/lstm_scaler.pkl      -- fitted StandardScaler
results/lstm_history.csv    -- train/val loss per epoch
results/lstm_predictions.csv -- test-set predictions vs actuals
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
from config import DATA_DIR, RESULTS_DIR, MODEL_DIR
from lstm_regime import (fetch_fred_data, label_regimes, train_lstm,
                          save_lstm, REGIME_NAMES)

os.makedirs(DATA_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,   exist_ok=True)

if __name__ == "__main__":
    print("=" * 60)
    print("LSTM REGIME CLASSIFIER")
    print("=" * 60)

    # ── 1. Fetch FRED data ────────────────────────────────────────────────────
    fred_path = os.path.join(DATA_DIR, "fred_data.csv")
    if os.path.exists(fred_path):
        print(f"Loading cached FRED data from {fred_path}")
        df = pd.read_csv(fred_path, index_col=0, parse_dates=True)
    else:
        print("Fetching FRED data (2005-2024)…")
        df = fetch_fred_data()
        df.to_csv(fred_path)
        print(f"Saved -> {fred_path}  ({len(df)} monthly observations)")

    print(f"\nDate range: {df.index[0].date()} -> {df.index[-1].date()}")
    print(df.describe().round(2), "\n")

    # ── 2. Train LSTM ─────────────────────────────────────────────────────────
    model, scaler, train_losses, val_losses, preds, labels = train_lstm(df)

    # ── 3. Save artefacts ─────────────────────────────────────────────────────
    save_lstm(model, scaler)

    # Loss history
    history_df = pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses})
    history_path = os.path.join(RESULTS_DIR, "lstm_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"Saved -> {history_path}")

    # Predictions
    preds_df = pd.DataFrame({
        "actual":    [REGIME_NAMES[l] for l in labels],
        "predicted": [REGIME_NAMES[p] for p in preds],
        "actual_idx":    labels,
        "predicted_idx": preds,
    })
    preds_path = os.path.join(RESULTS_DIR, "lstm_predictions.csv")
    preds_df.to_csv(preds_path, index=False)
    print(f"Saved -> {preds_path}")

    print("\nDone. Run `run_rl.py` next.")
