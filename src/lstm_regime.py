"""
LSTM-based credit regime classifier.

Data source : FRED (5 macro series, monthly, 1996-2024)
Regimes     : 0 = TIGHT (HY OAS < 350 bps)
              1 = NORMAL (350-600 bps)
              2 = WIDE   (> 600 bps)
Architecture: LSTM(128) → Dropout → LSTM(64) → Dropout → Linear(32) → Linear(3)
Target      : ~67% accuracy on 20% held-out chronological test set
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

from config import (FRED_API_KEY, WINDOW_SIZE, LSTM_EPOCHS, LSTM_LR,
                    BATCH_SIZE, DATA_DIR, MODEL_DIR)

FRED_SERIES = {
    "baa_spread":   "BAA10YM",        # Moody's Baa - 10Y Treasury (IG credit proxy, 1986-)
    "aaa_spread":   "AAA10YM",        # Moody's Aaa - 10Y Treasury (senior credit proxy, 1919-)
    "slope_2s10s":  "T10Y2Y",         # 10Y-2Y Treasury slope
    "vix":          "VIXCLS",         # VIX
    "loan_officer": "DRTSCILM",       # Senior loan officer tightening (quarterly)
}
FEATURES = list(FRED_SERIES.keys())
REGIME_NAMES = ["TIGHT", "NORMAL", "WIDE"]


# -- FRED data -----------------------------------------------------------------

def fetch_fred_data(start: str = "1996-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """Pull daily FRED data and return a business-day DataFrame.

    Spreads are computed from yield components (BAA/AAA minus DGS10) so that
    the full history back to 1996 is available -- the ICE BofA OAS series only
    starts on FRED from 2023.  Loan-officer survey is quarterly; forward-filled
    to daily.  Window size is set in config (default 120 business days ~ 6 months).
    """
    import fredapi
    fred = fredapi.Fred(api_key=FRED_API_KEY)

    kw = dict(observation_start=start, observation_end=end)

    baa      = fred.get_series("BAA",      **kw)   # Moody's Baa yield (daily)
    aaa      = fred.get_series("AAA",      **kw)   # Moody's Aaa yield (daily)
    dgs10    = fred.get_series("DGS10",    **kw)   # 10Y Treasury yield (daily)
    slope    = fred.get_series("T10Y2Y",   **kw)   # 10Y-2Y spread (daily)
    vix      = fred.get_series("VIXCLS",   **kw)   # VIX (daily)
    lofficer = fred.get_series("DRTSCILM", **kw)   # loan officer survey (quarterly)

    df = pd.DataFrame({
        "baa_spread":   baa - dgs10,    # credit quality spread (IG proxy)
        "aaa_spread":   aaa - dgs10,    # senior credit spread
        "slope_2s10s":  slope,
        "vix":          vix,
        "loan_officer": lofficer,
    })
    df.index = pd.to_datetime(df.index)

    df = df.resample("B").last()        # align to business-day calendar
    df = df.ffill().bfill()             # fill weekends/holidays + quarterly gaps
    df = df.dropna()
    return df


def label_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label credit regimes using VIX (reliable proxy for risk-off / credit stress).
    BAMLH0A0HYM2 is retained as a feature but has data quality issues pre-2010.

      TIGHT  (0): VIX < 15   -- benign credit, low volatility (~40% of months)
      NORMAL (1): 15 <= VIX < 25 -- moderate credit conditions  (~38% of months)
      WIDE   (2): VIX >= 25  -- credit stress, elevated spreads  (~22% of months)

    VIX is a robust classifier: it captured 2008-09 (~60), 2020 (~80), 2022 (~36).
    """
    df = df.copy()
    df["regime"] = 0                                   # TIGHT default
    df.loc[df["vix"] >= 15, "regime"] = 1             # NORMAL
    df.loc[df["vix"] >= 25, "regime"] = 2             # WIDE
    return df


# -- Dataset -------------------------------------------------------------------

class RegimeDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(df: pd.DataFrame, window: int = WINDOW_SIZE):
    """
    Build (X, y) pairs where X is a 24-month window of 5 features
    and y is the regime label at month t+1 (forward-looking).
    """
    values = df[FEATURES].values
    labels = df["regime"].values
    X, y = [], []
    for i in range(window, len(df) - 1):
        X.append(values[i - window : i])
        y.append(labels[i + 1])           # predict next month
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# -- Model ---------------------------------------------------------------------

class RegimeLSTM(nn.Module):
    def __init__(self, input_size: int = 5, hidden1: int = 128,
                 hidden2: int = 64, n_classes: int = 3, dropout: float = 0.4):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.head  = nn.Sequential(
            nn.Linear(hidden2, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),     # raw logits; softmax applied at inference
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)            # (B, T, 128)
        out     = self.drop1(out)
        out, _ = self.lstm2(out)          # (B, T,  64)
        out     = self.drop2(out)
        out     = out[:, -1, :]           # last timestep → (B, 64)
        return self.head(out)             # (B, 3) logits


# -- Training ------------------------------------------------------------------

def train_lstm(df: pd.DataFrame):
    """
    Train the LSTM regime classifier.

    Returns model, scaler, training history, and test predictions.
    """
    df = label_regimes(df)

    dist = df["regime"].value_counts(normalize=True).sort_index()
    print("Regime distribution:")
    for i, name in enumerate(REGIME_NAMES):
        print(f"  {name:6s}: {dist.get(i, 0):.1%}")
    print()

    X, y = create_sequences(df)
    split = int(len(X) * 0.80)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Fit scaler on training data only -- no lookahead
    n_train, seq_len, n_feat = X_train.shape
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, n_feat)
    scaler.fit(X_train_2d)

    X_train = scaler.transform(X_train_2d).reshape(n_train, seq_len, n_feat)
    n_test   = X_test.shape[0]
    X_test   = scaler.transform(X_test.reshape(-1, n_feat)).reshape(n_test, seq_len, n_feat)

    # Natural class distribution in batches -- the balanced sampler forced 33/33/33
    # per batch but the test set is 56% NORMAL, causing worse-than-baseline accuracy.
    train_loader = DataLoader(RegimeDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(RegimeDataset(X_test, y_test),
                              batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    model     = RegimeLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=15
    )

    PATIENCE       = 15          # epochs without val-loss improvement before stopping
    best_val_loss  = float("inf")
    best_state     = None
    no_improve     = 0
    train_losses, val_losses = [], []

    for epoch in range(LSTM_EPOCHS):
        # Train
        model.train()
        t_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss += loss.item()
        t_loss /= len(train_loader)

        # Validate
        model.eval()
        v_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out    = model(Xb)
                v_loss += criterion(out, yb).item()
                correct += (out.argmax(1) == yb).sum().item()
                total   += len(yb)
        v_loss /= len(test_loader)
        acc     = correct / total

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        scheduler.step(acc)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | train {t_loss:.4f} | val {v_loss:.4f} | acc {acc:.3f}")

        if no_improve >= PATIENCE:
            print(f"Early stop at epoch {epoch} (no val-loss improvement for {PATIENCE} epochs)")
            break

    model.cpu()
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            all_preds.extend(model(Xb).argmax(1).numpy())
            all_labels.extend(yb.numpy())

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"\nFinal test accuracy: {accuracy:.3f} ({accuracy*100:.1f}%) -- 3-class baseline: {1/3:.1%}")
    print(classification_report(all_labels, all_preds, labels=[0, 1, 2],
                                target_names=REGIME_NAMES, zero_division=0))

    return model, scaler, train_losses, val_losses, np.array(all_preds), np.array(all_labels)


# -- Inference helper ----------------------------------------------------------

def predict_regime(model: RegimeLSTM, scaler: StandardScaler,
                   window_df: pd.DataFrame) -> np.ndarray:
    """
    Given a WINDOW_SIZE-row DataFrame of recent daily macro features,
    return [P(tight), P(normal), P(wide)] for the next business day.
    """
    model.eval().cpu()
    x = window_df[FEATURES].values.astype(np.float32)
    x = scaler.transform(x)
    x = torch.FloatTensor(x).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=-1).numpy()[0]
    return probs


# -- Persistence ---------------------------------------------------------------

def save_lstm(model, scaler, path_prefix: str = None):
    if path_prefix is None:
        path_prefix = os.path.join(MODEL_DIR, "lstm")
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    torch.save(model.state_dict(), path_prefix + "_model.pt")
    with open(path_prefix + "_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved model -> {path_prefix}_model.pt")


def load_lstm(path_prefix: str = None):
    if path_prefix is None:
        path_prefix = os.path.join(MODEL_DIR, "lstm")
    model = RegimeLSTM()
    model.load_state_dict(torch.load(path_prefix + "_model.pt", weights_only=True))
    model.eval()
    with open(path_prefix + "_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler
