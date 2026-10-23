"""Offline NSE cash intraday simulator; next-open execution, INR accounting.

Observations use completed candles. Buy/Sell actions fill at the next open.
Protective exits handle opening gaps before actions and intrabar low crossings.
Minute bars cannot establish tick-by-tick fill certainty; fills are estimates.
"""
from __future__ import annotations

import json
from pathlib import Path
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO, DQN
from charges import IntradayCharges, IntradayChargeLedger
from risk_manager import DEFAULT_SETTINGS_PATH, RiskGatekeeper


def _finite(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


class TradingEnvironment(gym.Env):
    """Single long position; optional paper-only margin and whole-share sizing."""
    metadata = {"render_modes": []}
    OBSERVATION_NAMES = ("lstm_score", "cash", "position_count", "open_mtm")

    def __init__(self, candles: pd.DataFrame, *, symbol: str,
                 settings_path: str | Path = DEFAULT_SETTINGS_PATH,
                 initial_cash: float | None = None, lot_size: int = 1,
                 fee_bps: float | None = None, slippage_bps: float = 2.0,
                 turnover_penalty: float = 0.0001, overtrade_penalty: float = 0.001,
                 risk_penalty: float = 0.01, max_trades_per_episode: int = 10):
        super().__init__()
        self.settings_path = Path(settings_path)
        self.settings = json.loads(self.settings_path.read_text())
        self.timezone = self.settings.get('data', {}).get('timezone', 'Asia/Kolkata')
        if self.timezone != 'Asia/Kolkata':
            raise ValueError('NSE intraday simulation requires Asia/Kolkata')
        self.charges = IntradayCharges.from_settings(self.settings)
        self.fee_rate = None if fee_bps is None else _finite(fee_bps, 'fee_bps') / 10000
        if self.fee_rate is not None and self.fee_rate >= 1:
            raise ValueError('fee_bps must be below 10000')
        simulation = self.settings.get('simulation', {})
        if not isinstance(simulation, dict):
            raise ValueError('simulation settings must be an object')
        self.leverage = _finite(simulation.get('leverage', 1), 'leverage', positive=True)
        self.position_sizing = simulation.get('position_sizing', 'one_share')
        if self.position_sizing not in ('one_share', 'max_notional') or not 1 <= self.leverage <= 5:
            raise ValueError('Use one_share/max_notional sizing and leverage between 1 and 5')
        self.simulator_version = 4 if self.leverage != 1 or self.position_sizing != 'one_share' else 2
        if self.simulator_version == 4 and self.settings.get('execution', {}).get('mode') not in ('paper', 'simulation'):
            raise ValueError('Leveraged/variable sizing is restricted to fake-money modes')
        balance = simulation.get('initial_cash_inr', 100000) if initial_cash is None else initial_cash
        self.initial_cash = _finite(balance, 'initial_cash', positive=True)
        if isinstance(lot_size, bool) or not isinstance(lot_size, int) or lot_size != 1:
            raise ValueError('Current model and live adapter require lot_size=1')
        if isinstance(max_trades_per_episode, bool) or not isinstance(max_trades_per_episode, int) or max_trades_per_episode < 0:
            raise ValueError('max_trades_per_episode must be a nonnegative integer')
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError('symbol must be nonempty')
        self.symbol, self.lot_size = symbol.strip().upper(), lot_size
        self.slippage = _finite(slippage_bps, 'slippage_bps') / 10000
        if self.slippage >= 1:
            raise ValueError('slippage_bps must be below 10000')
        self.turnover_penalty = _finite(turnover_penalty, 'turnover_penalty')
        self.overtrade_penalty = _finite(overtrade_penalty, 'overtrade_penalty')
        self.risk_penalty = _finite(risk_penalty, 'risk_penalty')
        self.max_trades = max_trades_per_episode
        self.max_position = _finite(self.settings['risk_limits']['max_position_value_inr'], 'max_position_value_inr', positive=True)
        RiskGatekeeper(self.settings_path)
        self.df = self._validate(candles)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, 0, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, 1, np.inf], dtype=np.float32), dtype=np.float32)
        self._done = True

    def _validate(self, candles):
        if not isinstance(candles, pd.DataFrame) or candles.columns.has_duplicates:
            raise ValueError('candles must be a DataFrame with unique columns')
        df = candles.copy()
        if 'timestamp' in df:
            stamps = pd.DatetimeIndex(pd.to_datetime(df.pop('timestamp'), errors='raise'))
        elif isinstance(df.index, pd.DatetimeIndex):
            stamps = df.index
        else:
            raise ValueError('candles need a DatetimeIndex or timestamp column')
        stamps = stamps.tz_localize(self.timezone) if stamps.tz is None else stamps.tz_convert(self.timezone)
        if len(df) < 2 or stamps.hasnans or stamps.has_duplicates or not stamps.is_monotonic_increasing:
            raise ValueError('At least two sorted, unique candles are required')
        if (len(set(stamps.date)) != 1 or not (np.diff(stamps.as_unit('ns').asi8) == 60_000_000_000).all()
                or (stamps.second != 0).any() or (stamps.microsecond != 0).any() or (stamps.nanosecond != 0).any()
                or (stamps.dayofweek >= 5).any()
                or (stamps.strftime('%H:%M') < '09:15').any() or (stamps.strftime('%H:%M') > '15:29').any()):
            raise ValueError('Use one contiguous one-minute NSE weekday session, 09:15–15:29 IST')
        required = ['open', 'high', 'low', 'close']
        if not set(required).issubset(df):
            raise ValueError('candles require open, high, low and close')
        if 'lstm_score' not in df:
            df['lstm_score'] = 0.0
        numeric = required + ['lstm_score']
        df[numeric] = df[numeric].apply(pd.to_numeric, errors='raise')
        values = df[numeric].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (df[required] <= 0).any().any():
            raise ValueError('Prices must be positive and features finite')
        if ((df.high < df[required].max(axis=1)) | (df.low > df[required].min(axis=1))).any():
            raise ValueError('Inconsistent OHLC bounds')
        df = df.reset_index(drop=True)
        df['timestamp'] = stamps
        return df

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.gatekeeper = RiskGatekeeper(self.settings_path)
        self.current_step = 0
        self.cash = self.initial_cash
        self.position = 0
        self.quantity = 0
        self.margin_reserved = 0.0
        self.entry_price = 0.0
        self.trade_count = 0
        self.daily_pnl = 0.0
        self.charge_ledger = IntradayChargeLedger(self.charges)
        self.total_charges = 0.0
        self.charge_breakdown = {}
        self.executions = []
        self._done = False
        return self._get_obs(), self._info()

    def _equity(self, price):
        return self.cash + self.margin_reserved + self.quantity * (price - self.entry_price)

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        mtm = (row.close - self.entry_price) * self.quantity if self.position else 0.0
        if self.simulator_version == 4:
            # Fixed, causal units: percentage-point return prediction and account
            # fractions. No fitting on evaluation/practice observations.
            values = np.array([row.lstm_score * 100, self.cash / self.initial_cash,
                               self.position, mtm / self.initial_cash], dtype=np.float64)
        else:
            values = np.array([row.lstm_score, self.cash, self.position, mtm], dtype=np.float64)
        if not np.isfinite(values).all() or (np.abs(values) > np.finfo(np.float32).max).any():
            raise ValueError('Account or feature observations overflow float32')
        return values.astype(np.float32)

    def _fees(self, value, side):
        if self.fee_rate is not None:
            return {'legacy_fee': value * self.fee_rate, 'total': value * self.fee_rate}
        return self.charge_ledger.quote(value, side)

    def _size_entry(self, price):
        cap = min(self.max_position, float(self.settings['risk_limits']['max_order_value_inr']))
        upper = min(int(cap // price), int(max(0, self.cash) * self.leverage // price))
        if self.position_sizing == 'one_share':
            upper = min(upper, 1)
        low, high = 0, upper
        # Reserve initial margin plus actual entry fees, never spend borrowed
        # proceeds as equity. Fees apply to full turnover, not margin capital.
        while low < high:
            quantity = (low + high + 1) // 2
            value = quantity * price
            if value / self.leverage + self._fees(value, 'BUY')['total'] <= self.cash:
                low = quantity
            else:
                high = quantity - 1
        return low

    @property
    def observation_schema(self):
        return 'account_fractions_return_percent_v1' if self.simulator_version == 4 else 'raw_account_v1'

    def execution_profile(self):
        return {'position_sizing': self.position_sizing, 'leverage': self.leverage,
                'observation_schema': self.observation_schema,
                'initial_cash_inr': self.initial_cash,
                'max_order_value_inr': float(self.settings['risk_limits']['max_order_value_inr']),
                'max_position_value_inr': self.max_position}

    def _execute(self, side, price, info, *, forced=False, reason=None, quantity=None):
        quantity = (1 if quantity is None else quantity) if side == 'BUY' else self.quantity
        value = price * quantity
        fees = (self._fees(value, side) if self.fee_rate is not None
                else self.charge_ledger.record(value, side))
        if side == 'BUY':
            self.margin_reserved = value / self.leverage
            self.cash -= self.margin_reserved + fees['total']
            self.quantity = quantity
        else:
            self.cash += self.margin_reserved + quantity * (price - self.entry_price) - fees['total']
            self.margin_reserved = 0.0
            self.quantity = 0
        if self.fee_rate is None:
            summary = self.charge_ledger.summary()
            self.total_charges = summary['total']
            self.charge_breakdown = {k: v for k, v in summary.items() if k != 'total'}
        else:
            self.total_charges += fees['total']
            self.charge_breakdown['legacy_fee'] = self.total_charges
        self.position = 1 if side == 'BUY' else 0
        self.entry_price = price if self.position else 0.0
        if not forced:
            self.trade_count += 1
        key = 'forced_exit_count' if forced else ('buy_count' if side == 'BUY' else 'sell_count')
        info[key] += 1
        info['total_executed_trades'] += 1
        self.executions.append({'side': side, 'price': price, 'quantity': quantity,
                                'notional_inr': value, 'leverage': self.leverage,
                                'timestamp': self.df.iloc[self.current_step].timestamp.isoformat(),
                                'charges': fees, 'forced': forced, 'reason': reason})

    def _info(self):
        equity = self._equity(float(self.df.iloc[self.current_step].close))
        self.daily_pnl = equity - self.initial_cash
        return {'risk_violations': 0, 'order_rejected': False, 'equity': equity,
                'daily_pnl': self.daily_pnl, 'cash': self.cash,
                'quantity': self.quantity, 'margin_reserved_inr': self.margin_reserved,
                'position_value_inr': self.quantity * float(self.df.iloc[self.current_step].close),
                'buy_count': 0,
                'sell_count': 0, 'forced_exit_count': 0, 'total_executed_trades': 0,
                'trade_count': self.trade_count, 'total_charges': self.total_charges,
                'charge_breakdown': self.charge_breakdown.copy(), 'reason': None}

    def _exit(self, price, info, reason):
        if self.position:
            self._execute('SELL', price * (1 - self.slippage), info, forced=True, reason=reason)

    @property
    def done(self):
        return self._done

    def append_candles(self, candles):
        """Append new completed bars to a streaming episode without resetting money."""
        if self._done:
            raise RuntimeError('Cannot append to a halted episode')
        incoming = candles.copy()
        if 'timestamp' in incoming:
            incoming = incoming.set_index('timestamp')
        current = self.df.set_index('timestamp')
        self.df = self._validate(pd.concat([current, incoming]))

    def finish(self, reason='paper_session_finished'):
        """Close simulated exposure at the last known close; no broker operations."""
        info = self._info()
        if not self._done:
            self._exit(float(self.df.iloc[self.current_step].close), info, reason)
            self._done = True
        final = self._info()
        for key in ('equity', 'daily_pnl', 'cash', 'trade_count', 'total_charges', 'charge_breakdown',
                    'quantity', 'margin_reserved_inr', 'position_value_inr'):
            info[key] = final[key]
        info['reason'] = reason
        return info

    def step(self, action, *, end_of_data=True):
        if self._done:
            raise RuntimeError('Call reset before stepping a finished episode')
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError('action must be 0 (Hold), 1 (Buy), or 2 (Sell)')
        if self.current_step + 1 >= len(self.df):
            raise ValueError('Wait for the next completed candle before stepping')
        initial_equity = self._equity(float(self.df.iloc[self.current_step].close))
        info = self._info()
        penalty = 0.0
        now = self.df.iloc[self.current_step].timestamp
        # At a known close, act on the risk state immediately, without advancing time.
        if self.gatekeeper.check_time_cutoff(now.strftime('%H:%M:%S')) or not self.gatekeeper.check_daily_loss(self.daily_pnl):
            self._exit(float(self.df.iloc[self.current_step].close), info, 'risk_cutoff_breach')
            self._done = True
            info['reason'] = 'risk_cutoff_breach'
            penalty += self.risk_penalty
        else:
            self.current_step += 1
            row = self.df.iloc[self.current_step]
            opening = float(row.open)
            cutoff = self.gatekeeper.check_time_cutoff(row.timestamp.strftime('%H:%M:%S'))
            gap_loss = not self.gatekeeper.check_daily_loss(self._equity(opening) - self.initial_cash)
            if cutoff or gap_loss:
                self._exit(opening, info, 'time_cutoff' if cutoff else 'daily_loss')
                self._done = True
                info['reason'] = 'risk_cutoff_breach'
            else:
                # A gap below the protective stop takes precedence over an action.
                stopped = False
                if self.position and opening <= float(self.gatekeeper.calculate_stop_loss(self.entry_price, 'BUY')):
                    self._exit(opening, info, 'stop_loss_gap')
                    stopped = True
                if action == 1 and not stopped:
                    price = opening * (1 + self.slippage)
                    quantity = self._size_entry(price) if not self.position else 0
                    value = price * quantity
                    if (not self.position and quantity > 0
                            and value <= self.max_position
                            and self.gatekeeper.validate_new_order(self.symbol, value, self.position)):
                        self._execute('BUY', price, info, quantity=quantity)
                    else:
                        info['order_rejected'] = True
                        penalty += self.risk_penalty
                elif action == 2 and self.position:
                    self._execute('SELL', opening * (1 - self.slippage), info)
                # Use the higher protective trigger: 1% stop or net MTM loss cap.
                # A bar low reveals a crossing, not the exact tick or guaranteed fill.
                if self.position:
                    stop = float(self.gatekeeper.calculate_stop_loss(self.entry_price, 'BUY'))
                    max_loss = float(self.settings['risk_limits']['max_daily_loss_inr'])
                    loss_trigger = (self.initial_cash - max_loss - self.cash - self.margin_reserved) / self.quantity + self.entry_price
                    trigger = max(stop, loss_trigger)
                    if float(row.low) <= trigger:
                        reason = 'daily_loss' if loss_trigger >= stop else 'stop_loss'
                        self._exit(min(opening, trigger), info, reason)
                        if reason == 'daily_loss':
                            self.gatekeeper.halt_trading()
                            self._done = True
                            info['reason'] = 'risk_cutoff_breach'
                if not self.gatekeeper.check_daily_loss(self._equity(float(row.close)) - self.initial_cash):
                    self._exit(float(row.close), info, 'daily_loss')
                    self._done = True
                    info['reason'] = 'risk_cutoff_breach'
                if end_of_data and self.current_step == len(self.df) - 1:
                    # End of data is a close auction estimate, not a second fill at its open.
                    self._exit(float(row.close), info, 'end_of_data')
                    self._done = True
                    info['reason'] = info['reason'] or 'end_of_data'
        penalty += self.turnover_penalty * (info['buy_count'] + info['sell_count'])
        if self.trade_count > self.max_trades and info['buy_count'] + info['sell_count']:
            penalty += self.overtrade_penalty
        # Always synchronize financial fields after all fills, including forced exits.
        final = self._info()
        for key in ('equity', 'daily_pnl', 'cash', 'trade_count', 'total_charges', 'charge_breakdown',
                    'quantity', 'margin_reserved_inr', 'position_value_inr'):
            info[key] = final[key]
        info['equity_change_inr'] = info['equity'] - initial_equity
        info['reward_penalty'] = penalty
        reward = info['equity_change_inr'] / self.initial_cash - penalty
        return self._get_obs(), float(reward), self._done, False, info


class RLTradingBrain:
    """Stable-Baselines3 policy construction for offline training."""
    ALGORITHMS = ('PPO', 'DQN')

    def __init__(self, env, algorithm='PPO', seed=42, device='cpu', model_kwargs=None):
        if algorithm not in self.ALGORITHMS:
            raise ValueError('Unknown RL target framework: ' + str(algorithm))
        cls = PPO if algorithm == 'PPO' else DQN
        self.model = cls('MlpPolicy', env, verbose=0, seed=seed, device=device, **(model_kwargs or {}))
