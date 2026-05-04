# CLO-RL Optimizer

**Autonomous CLO tranche allocation using Monte Carlo simulation, LSTM regime classification, and PPO reinforcement learning.**

---

## Results at a Glance

| Metric | Equal-Weight Baseline | PPO Agent |
|--------|-----------------------|-----------|
| Sharpe Ratio (annualized) | 1.89 | **2.55** |
| Avg Max Drawdown | 6.35% | **3.97%** |
| Excess Return | — | **+73 bps/yr** |
| Drawdown Reduction | — | **−37%** |
| Seeds Won (15 tested) | 0 / 15 | **15 / 15** |

LSTM regime classifier: **82.3% accuracy** on held-out test set (2.2× random baseline).  
Monte Carlo engine: **2,000 paths × 60 months**, BBB OC breach rate **12.5%**.

---

## What This Project Does

A CLO (Collateralized Loan Obligation) pools corporate loans into tranches — AAA, AA, A, BBB, and equity — with a strict payment waterfall and overcollateralization (OC) triggers. Equity investors receive the residual after all senior notes are paid, making their returns highly sensitive to default timing and the OC breach event.

This project builds three integrated components to optimize equity returns:

```
FRED Macro Data (daily, 1996-2024)
        │
        ▼
┌─────────────────────────┐
│  LSTM Regime Classifier │  → P(TIGHT), P(NORMAL), P(WIDE)
│  LSTM(128)→LSTM(64)→FC  │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐    ┌──────────────────────────┐
│  CLO Environment        │◄───│  Monte Carlo Waterfall   │
│  (Gymnasium)            │    │  2,000 paths × 60 months │
└─────────────────────────┘    └──────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  PPO RL Agent           │  → Dynamic weights [Cash, AAA, AA, A, BBB]
│  MlpPolicy, 409K steps  │
└─────────────────────────┘
```

---

## CLO Capital Structure

| Tranche | Balance | Spread | All-in Coupon | OC Threshold |
|---------|---------|--------|---------------|--------------|
| AAA | $200M | 1.45% | 6.75% | 1.20× |
| AA | $60M | 2.15% | 7.45% | 1.10× |
| A | $40M | 3.10% | 8.40% | 1.06× |
| BBB | $30M | 4.55% | 9.85% | 1.03× |
| Equity | $170M | residual | residual | — |
| **Total** | **$500M** | | | |

SOFR: 5.30% · Senior fee: 15 bps · Pool: 200 equal-weight loans · Horizon: 60 months

---

## Module 1 — Monte Carlo Waterfall

Simulates 2,000 independent CLO paths with:

- **CDR**: Lognormal(μ=−3.5, σ=0.7), clipped [0.5%, 15%], median ≈ 2.9%
- **Recovery rate**: Uniform(30%, 70%) per path, 12-month lag
- **CPR**: Uniform(10%, 25%) per path
- **Default model**: Binomial across 200 loans per period
- **OC test**: `performing_balance / note_balance` vs 1.03× threshold
- **On breach**: equity cash diverted to senior note paydown

**Key finding**: CDR has near-zero correlation with equity (corr ≈ 0.00). Recovery rate is the dominant driver (corr = +0.67). OC breach shifts the downside tail — p10 equity drops from $91M to $71M in breached paths — but medians are similar ($101M vs $97M) because breach occurs at median month 45, after most equity is paid out.

```bash
python run_waterfall.py   # single path demo
python run_all.py         # full 2,000-path simulation
```

---

## Module 2 — LSTM Regime Classifier

Classifies macroeconomic regime from 5 FRED daily series (1996–2024, 7,567 observations):

| Feature | FRED Series | Role |
|---------|-------------|------|
| Baa corporate spread | BAA10YM | Credit stress |
| Aaa corporate spread | AAA10YM | Investment grade |
| Yield curve slope | T10Y2Y | Recession signal |
| VIX | VIXCLS | Market fear |
| Senior loan officer survey | DRTSCILM | Credit tightening |

**Regime labels** (VIX thresholds):

| Regime | Condition | Samples |
|--------|-----------|---------|
| TIGHT | VIX < 15 | 374 |
| NORMAL | 15 ≤ VIX < 25 | 793 |
| WIDE | VIX ≥ 25 | 323 |

**Architecture**:
```
Input [batch, 120, 5]
→ LSTM(128) → Dropout(0.4)
→ LSTM(64)  → Dropout(0.4)
→ Linear(32) → ReLU → Linear(3)
→ Softmax → P(TIGHT, NORMAL, WIDE)
```

**Training**: Adam lr=3e-4, weight_decay=1e-4, batch=64, max 100 epochs, early stopping patience=15, gradient clip=1.0, ReduceLROnPlateau scheduler.

**Results**:

| Class | Accuracy | Samples |
|-------|----------|---------|
| TIGHT | 70.6% | 374 |
| NORMAL | 86.1% | 793 |
| WIDE | 86.7% | 323 |
| **Overall** | **82.3%** | 1,490 |

```bash
python run_lstm.py        # train LSTM and save model
```

---

## Module 3 — PPO RL Allocation Agent

### Environment (`src/rl_env.py`)

**Observation** (9-dim):
```
[P_tight, P_normal, P_wide]           — LSTM regime probabilities
[OC_AAA, OC_AA, OC_A, OC_BBB] / 2    — normalized OC ratios
[CDR / 0.10]                          — current default rate
[CDR × 10]                            — HY spread proxy
```

**Action** (5-dim, softmax → portfolio weights):
```
[Cash,  AAA,  AA,  A,  BBB]
```

**Asset durations** (for spread shock P&L):
```
Cash: 0.0yr   AAA: 3.0yr   AA: 4.0yr   A: 4.5yr   BBB: 5.0yr
```

**Spread shock dynamics**:
```
spread_shock = p_wide × |N(0.005, 0.003)| − p_tight × |N(0.002, 0.001)| + CDR_z × N(0.001, 0.0005)
monthly_return = weights · (coupons/12 − durations × spread_shock)
```

**Reward**:
```
reward = portfolio_return × 100
       − 0.50  (if BBB OC breach this step)
```

### Training (`run_rl.py`)

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (stable-baselines3) |
| Policy | MlpPolicy (64→64 hidden, Tanh) |
| Learning rate | 3e-4 |
| n_steps | 2,048 |
| batch_size | 64 |
| n_epochs | 10 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.20 |
| Total timesteps | **409,600** (200 updates × 2,048) |
| Device | CPU |

```bash
python run_rl.py          # train PPO and evaluate vs baseline
```

### Regime-Conditional Allocation

| Regime | Cash | AAA | AA | A | BBB |
|--------|------|-----|----|---|-----|
| TIGHT (benign) | 10% | 18% | 22% | 14% | **36%** |
| NORMAL | 16% | 22% | 22% | 16% | 24% |
| WIDE (stress) | **32%** | 28% | 20% | 12% | 8% |
| CRISIS (extreme) | **40%** | 30% | 17% | 9% | 4% |

---

## Robustness & Honest Negatives

**15-seed validation**:
```
PPO   Sharpe: 2.385 ± 0.180   (15/15 seeds beat baseline)
Base  Sharpe: 1.641 ± 0.189
```

**Where PPO fails**: In an always-TIGHT regime (uniform benign spreads), PPO Sharpe = 17.7 vs baseline 19.6. The agent over-rotates to defensive assets even when credit is benign — sacrificing carry.

**Model limitations**:
- No IC (Interest Coverage) trigger — only OC modeled
- Single pool CDR — no loan-level heterogeneity or sector concentration
- Reinvestment at par — real managers trade at market prices in stress

```bash
python seed_variance.py       # 15-seed robustness test
python honest_negatives.py    # regime-level failure analysis
python deep_analysis.py       # all 8 deep-dive metrics
```

---

## Repository Structure

```
clo-rl-optimizer/
├── src/
│   ├── config.py          # global constants (SOFR, tranches, hyperparameters)
│   ├── loan_pool.py       # synthetic $500M loan pool generation
│   ├── waterfall.py       # monthly cashflow waterfall mechanics
│   ├── monte_carlo.py     # 2,000-path stochastic simulation engine
│   ├── lstm_regime.py     # RegimeLSTM model + training loop
│   ├── rl_env.py          # CLOEnv (Gymnasium)
│   └── rl_agent.py        # PPO training + evaluation
├── notebooks/
│   └── results.ipynb      # analysis notebook
├── data/
│   └── fred_data.csv      # 7,567 daily FRED observations (1996-2024)
├── models/
│   ├── lstm_model.pt      # trained LSTM weights
│   ├── lstm_scaler.pkl    # feature scaler
│   └── ppo_clo.zip        # trained PPO policy
├── results/
│   ├── mc_summary.csv     # 2,000-row Monte Carlo output
│   ├── rl_summary.txt     # authoritative Sharpe + drawdown
│   ├── rl_baseline.csv    # 500 baseline episodes
│   ├── rl_policy.csv      # 500 PPO episodes
│   ├── rl_training_log.csv
│   ├── lstm_predictions.csv
│   ├── lstm_history.csv
│   └── seed_variance.csv  # 15-seed comparison
├── run_all.py             # run full Monte Carlo pipeline
├── run_lstm.py            # train LSTM classifier
├── run_rl.py              # train + evaluate PPO agent
├── run_waterfall.py       # single-path waterfall demo
├── deep_analysis.py       # 8 deep-dive analyses
├── seed_variance.py       # 15-seed robustness test
├── honest_negatives.py    # failure mode analysis
├── make_ppt.py            # generate results slide deck
└── requirements.txt
```

---

## Quickstart

```bash
git clone https://github.com/likhith2366/clo-rl-optimizer.git
cd clo-rl-optimizer
pip install -r requirements.txt

# Set FRED API key (free at https://fred.stlouisfed.org/docs/api/api_key.html)
export FRED_API_KEY=your_key_here   # Windows: set FRED_API_KEY=your_key_here

# Run full pipeline
python run_all.py          # Monte Carlo simulation
python run_lstm.py         # Train LSTM regime classifier
python run_rl.py           # Train + evaluate PPO agent

# Analysis
python deep_analysis.py    # 8 analytical deep-dives
python seed_variance.py    # Robustness across 15 seeds
python honest_negatives.py # Where the model fails

# Results deck
python make_ppt.py         # Generates results/StructuredAlpha_v6.pptx
```

> **Note**: Pre-trained models (`models/`) and pre-computed results (`results/`) are included. You can skip directly to `deep_analysis.py` or `make_ppt.py` without retraining.

---

## Tech Stack

| Component | Library |
|-----------|---------|
| RL training | stable-baselines3 (PPO) |
| RL environment | Gymnasium |
| LSTM | PyTorch |
| Monte Carlo | NumPy / SciPy |
| Data | pandas, FRED API |
| Visualization | matplotlib, seaborn |
| Presentation | python-pptx |

---

## License

MIT
