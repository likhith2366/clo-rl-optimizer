"""
CLO portfolio allocation environment for PPO.

Aggregate pool model (same assumptions as monte_carlo.py):
  - Performing balance decays from CDR only; prepays reinvested
  - Note balances fixed at originals (reinvestment-period model)
  - OC test: performing / fixed_note_balance vs threshold

State (9-dim)
  [0-2]  P(tight), P(normal), P(wide) from LSTM or random
  [3-6]  AAA/AA/A/BBB OC ratios, normalised to [0,1] (OC=2x -> 1.0)
  [7]    Current CDR (normalised by /0.10)
  [8]    HY spread proxy (CDR-based)

Action (5-dim -> softmax -> weights)
  [0] Cash   [1] AAA   [2] AA   [3] A   [4] BBB

Reward
  portfolio_return*100 - 0.5*max(0, drawdown-7%)*100 - 0.30*(OC breach)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import TRANCHE_DEFS, SOFR, N_PERIODS, TOTAL_NOTIONAL
from waterfall import apply_waterfall, FIXED_NOTE_BALANCES, FIXED_NOTE_COUPONS, OC_THRESHOLDS

AVG_COUPON = 0.085


class CLOEnv(gym.Env):
    metadata = {"render_modes": []}

    _ASSET_COUPONS = np.array(
        [SOFR] + [SOFR + d["spread"] for d in TRANCHE_DEFS],
        dtype=np.float64,
    )
    # Modified duration (years): [cash, AAA, AA, A, BBB]
    _DURATIONS = np.array([0.0, 3.0, 4.0, 4.5, 5.0], dtype=np.float64)

    def __init__(self, regime_probs=None, n_periods=N_PERIODS):
        super().__init__()
        self.n_periods    = n_periods
        self.regime_probs = regime_probs

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)
        self.action_space      = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.cdr           = self.np_random.uniform(0.01, 0.10)
        self.recovery_rate = self.np_random.uniform(0.30, 0.70)
        self.cpr           = self.np_random.uniform(0.10, 0.25)

        self.performing     = float(TOTAL_NOTIONAL)
        self.recovery_queue = {}
        self.period         = 0

        self.portfolio_value = 1.0
        self.peak_value      = 1.0

        self._sample_regime()
        return self._obs(), {}

    def _sample_regime(self):
        if self.regime_probs is not None:
            self.current_regime = np.array(self.regime_probs, dtype=np.float64)
            return
        r = self.np_random.random()
        if r < 0.42:
            self.current_regime = np.array([0.85, 0.12, 0.03])
        elif r < 0.80:
            self.current_regime = np.array([0.10, 0.80, 0.10])
        else:
            self.current_regime = np.array([0.05, 0.25, 0.70])

    def _oc_norm(self):
        cum = 0.0
        oc_norm = []
        for d in TRANCHE_DEFS:
            cum += FIXED_NOTE_BALANCES[d["name"]]
            oc = self.performing / cum if cum > 0 else 2.0
            oc_norm.append(min(oc / 2.0, 1.0))
        return np.array(oc_norm, dtype=np.float32)

    def _obs(self):
        oc = self._oc_norm()
        return np.array([
            self.current_regime[0],
            self.current_regime[1],
            self.current_regime[2],
            *oc,
            min(self.cdr / 0.10, 1.0),
            min(self.cdr * 10.0, 1.0),
        ], dtype=np.float32)

    def step(self, action):
        a       = np.array(action, dtype=np.float64)
        e_a     = np.exp(a - a.max())
        weights = e_a / e_a.sum()

        # Simulate one month (aggregate model)
        monthly_dp      = 1.0 - (1.0 - self.cdr) ** (1.0 / 12)
        actual_defaults = float(np.clip(
            self.np_random.normal(self.performing * monthly_dp, self.performing * monthly_dp * 0.1),
            0, self.performing
        ))
        self.performing -= actual_defaults
        self.recovery_queue[self.period + 12] = (
            self.recovery_queue.get(self.period + 12, 0.0) + actual_defaults * self.recovery_rate
        )

        recovery_cash = self.recovery_queue.pop(self.period, 0.0)
        interest_cash = self.performing * AVG_COUPON / 12

        wf = apply_waterfall(
            interest_cash      = interest_cash,
            performing_balance = self.performing,
            recovery_cash      = recovery_cash,
        )

        # Stochastic spread dynamics: regime + CDR drive spread widening/compression.
        # Price return = -duration * monthly_spread_change (standard bond math).
        p_wide  = float(self.current_regime[2])
        p_tight = float(self.current_regime[0])
        cdr_z   = self.cdr / 0.10   # normalised 0-1

        spread_shock = (
            p_wide  * abs(self.np_random.normal(0.005, 0.003))
          - p_tight * abs(self.np_random.normal(0.002, 0.001))
          + cdr_z   * self.np_random.normal(0.001, 0.0005)
        )

        monthly_returns = self._ASSET_COUPONS / 12 - self._DURATIONS * spread_shock
        if wf["oc_breached"].get("BBB", False):
            monthly_returns[4] = -0.02

        port_return = float(np.dot(weights, monthly_returns))

        self.portfolio_value *= (1.0 + port_return)
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value

        # Reward = raw monthly return (spread shocks already encode risk).
        # OC breach adds a discrete credit-event penalty (~0.5% monthly equivalent).
        reward = port_return * 100.0
        if any(wf["oc_breached"].values()):
            reward -= 0.50

        noise = self.np_random.normal(0, 0.02, 3)
        new_r = np.clip(self.current_regime + noise, 0.01, 0.98)
        self.current_regime = new_r / new_r.sum()

        self.period += 1
        done      = self.period >= self.n_periods
        truncated = False

        info = {
            "portfolio_value": self.portfolio_value,
            "port_return":     port_return,
            "drawdown":        drawdown,
            "oc_breached":     any(wf["oc_breached"].values()),
            "weights":         weights.tolist(),
        }
        return self._obs(), reward, done, truncated, info
