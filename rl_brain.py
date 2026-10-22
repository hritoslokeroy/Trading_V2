"""Offline, single-instrument RL simulation and Stable-Baselines3 policies.

Each episode covers one exchange-local trading day. Orders execute at
the NEXT candle's open, with adverse slippage and fees. Actions are 0=Hold,
1=Buy one fixed lot, 2=Sell the existing long lot; shorting is not supported.
"""
from __future__ import annotations

import json
from pathlib import Path
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO, DQN

from src.execution.risk_manager import DEFAULT_SETTINGS_PATH, RiskGatekeeper

def _finite(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0) or (not positive and result < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result

class TradingEnvironment(gym.Env):
    """Four observations, three actions, auditable decomposed step rewards."""
    metadata = {"render_modes": []}
    OBSERVATION_NAMES = ("lstm_score", "cash", "position_count", "open_mtm")

    def __init__(
        self, candles: pd.DataFrame, *, symbol: str,
        settings_path: str | Path = DEFAULT_SETTINGS_PATH,
        initial_cash: float = 100_000.0, lot_size: int = 1,
        fee_bps: float = 5.0, slippage_bps: float = 2.0,
        turnover_penalty: float = 0.0001, overtrade_penalty: float = 0.001,
        risk_penalty: float = 0.01, max_trades_per_episode: int = 10,
    ):
        super().__init__()
        self.settings_path = Path(settings_path)
        if "timestamp" in candles.columns:
            self.df = candles.reset_index(drop=True)
        elif isinstance(candles.index, pd.DatetimeIndex):
            # step() reads the local clock from a "timestamp" column; keep the index as that column.
            self.df = candles.rename_axis("timestamp").reset_index()
        else:
            raise ValueError("candles need a DatetimeIndex or a timestamp column")
        self.initial_cash = _finite(initial_cash, "initial_cash", positive=True)
        
        if isinstance(lot_size, bool) or not isinstance(lot_size, int) or lot_size <= 0:
            raise ValueError("lot_size must be a positive integer")
        if isinstance(max_trades_per_episode, bool) or not isinstance(max_trades_per_episode, int) or max_trades_per_episode < 0:
            raise ValueError("max_trades_per_episode must be a nonnegative integer")
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be nonempty")
            
        self.symbol, self.lot_size = symbol.strip().upper(), lot_size
        self.fee_rate = _finite(fee_bps, "fee_bps") / 10_000
        self.slippage = _finite(slippage_bps, "slippage_bps") / 10_000
        self.turnover_penalty = _finite(turnover_penalty, "turnover_penalty")
        self.overtrade_penalty = _finite(overtrade_penalty, "overtrade_penalty")
        self.risk_penalty = _finite(risk_penalty, "risk_penalty")
        self.max_trades = max_trades_per_episode
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, 0.0, 0.0, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, 1.0, np.inf], dtype=np.float32),
            dtype=np.float32
        )
        
        with self.settings_path.open(encoding="utf-8") as stream:
            settings = json.load(stream)
        self.timezone = settings.get("data", {}).get("timezone", "Asia/Kolkata")
        
        # Initial compilation verification check
        RiskGatekeeper(self.settings_path)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.gatekeeper = RiskGatekeeper(self.settings_path)
        self.current_step = 0
        self.cash = self.initial_cash
        self.position = 0
        self.entry_price = 0.0
        self.trade_count = 0
        self.daily_pnl = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        score = float(self.df.loc[self.current_step, "lstm_score"]) if "lstm_score" in self.df.columns else 0.0
        close = float(self.df.loc[self.current_step, "close"])
        open_mtm = (close - self.entry_price) * self.lot_size if self.position > 0 else 0.0
        return np.array([score, self.cash, float(self.position), open_mtm], dtype=np.float32)

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        info = {"risk_violations": 0, "order_rejected": False, "equity": self.cash, "daily_pnl": self.daily_pnl, "cash": self.cash, "buy_count": 0, "sell_count": 0, "forced_exit_count": 0, "total_executed_trades": 0, "trade_count": self.trade_count, "reason": None}
        
        current_close = float(self.df.loc[self.current_step, "close"])
        timestamp_str = str(self.df.loc[self.current_step, "timestamp"])
        time_only_str = timestamp_str.split()[-1][:5]
        
        next_idx = min(self.current_step + 1, len(self.df) - 1)
        next_open = float(self.df.loc[next_idx, "open"])
        
        initial_equity = self.cash + ((current_close - self.entry_price) * self.lot_size if self.position > 0 else 0.0)
        
        if self.gatekeeper.check_time_cutoff(time_only_str) or not self.gatekeeper.check_daily_loss(self.daily_pnl):
            if self.position > 0:
                fill_price = next_open * (1.0 - self.slippage)
                exit_value = fill_price * self.lot_size
                self.cash += exit_value - (exit_value * self.fee_rate)
                self.position = 0
                info["forced_exit_count"] = 1
                info["total_executed_trades"] += 1
            terminated = True
            info["reason"] = "risk_cutoff_breach"
            return self._get_obs(), -self.risk_penalty, terminated, truncated, info

        if action == 1:
            if self.position == 0:
                order_value = next_open * self.lot_size
                if self.gatekeeper.validate_new_order(self.symbol, order_value, self.position):
                    fill_price = next_open * (1.0 + self.slippage)
                    cost = fill_price * self.lot_size
                    self.cash -= cost + (cost * self.fee_rate)
                    self.entry_price = fill_price
                    self.position = 1
                    self.trade_count += 1
                    info["buy_count"] = 1
                    info["total_executed_trades"] += 1
                    reward -= self.turnover_penalty
                    if self.trade_count > self.max_trades:
                        reward -= self.overtrade_penalty
                else:
                    info["order_rejected"] = True
                    reward -= self.risk_penalty
            else:
                info["risk_violations"] = 1
                reward -= self.risk_penalty

        elif action == 2:
            if self.position > 0:
                fill_price = next_open * (1.0 - self.slippage)
                revenue = fill_price * self.lot_size
                self.cash += revenue - (revenue * self.fee_rate)
                trade_pnl = (fill_price - self.entry_price) * self.lot_size
                self.daily_pnl += trade_pnl
                self.position = 0
                self.entry_price = 0.0
                info["sell_count"] = 1
                info["total_executed_trades"] += 1
                reward -= self.turnover_penalty
            else:
                info["risk_violations"] = 1
                reward -= self.risk_penalty

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            terminated = True
            if self.position > 0:
                revenue = next_open * (1.0 - self.slippage) * self.lot_size
                self.cash += revenue - (revenue * self.fee_rate)
                trade_pnl = (next_open * (1.0 - self.slippage) - self.entry_price) * self.lot_size
                self.daily_pnl += trade_pnl
                self.position = 0
                info["forced_exit_count"] = 1
                info["total_executed_trades"] += 1
            info["reason"] = "end_of_data"

        final_close = float(self.df.loc[min(self.current_step, len(self.df)-1), "close"])
        final_equity = self.cash + ((final_close - self.entry_price) * self.lot_size if self.position > 0 else 0.0)
        equity_change = (final_equity - initial_equity) / self.initial_cash
        reward += equity_change

        info["equity"] = final_equity
        info["cash"] = self.cash
        info["daily_pnl"] = self.daily_pnl
        info["trade_count"] = self.trade_count

        return self._get_obs(), float(reward), terminated, truncated, info

class RLTradingBrain:
    """Wrapper mapping configuration algorithms smoothly to Gym targets."""
    ALGORITHMS = ("PPO", "DQN")
    def __init__(self, env: gym.Env, algorithm: str = "PPO", seed: int = 42, device: str = "cpu", model_kwargs: dict | None = None):
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unknown RL target framework: {algorithm}")
        kwargs = model_kwargs or {}
        if algorithm == "PPO":
            self.model = PPO("MlpPolicy", env, verbose=0, seed=seed, device=device, **kwargs)
        else:
            self.model = DQN("MlpPolicy", env, verbose=0, seed=seed, device=device, **kwargs)
