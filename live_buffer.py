"""Bounded, single-instrument Groww thread-locked snapshot aggregation buffer.

Slices raw incoming JSON stream packets into 1-minute OHLCV candles, handles
volume deduplication using deltas, updates Wilder RSI scales across array evictions,
and generates the structured tensors required for LSTM inference.
"""
from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from numbers import Real
from pathlib import Path
from threading import RLock
import numpy as np
import pandas as pd
import torch
from quant_agent import QuantLSTMBrain


class LiveCandleBuffer:
    """Store at most 500 completed bars plus one mutable active candle bar inside memory."""

    MAX_HISTORY = 500
    FEATURES = QuantLSTMBrain.FEATURES
    TIMEZONE = "Asia/Kolkata"

    def __init__(
        self,
        instrument_token: str,
        *,
        exchange: str = "NSE",
        segment: str = "CASH",
        max_history: int = 500,
        rsi_period: int = 14,
        volume_mode: str = "cumulative",
        timestamp_unit: str = "ms",
        scaler_path: str | Path | None = None,
        device: str | torch.device | None = None,
    ):
        if not isinstance(instrument_token, str) or not instrument_token.strip():
            raise ValueError("instrument_token must be the nonempty Groww subscription token")
        if isinstance(max_history, bool) or not isinstance(max_history, int) or not 60 <= max_history <= 500:
            raise ValueError("max_history must be an integer from 60 through 500")
        if isinstance(rsi_period, bool) or not isinstance(rsi_period, int) or not 1 <= rsi_period < 500:
            raise ValueError("rsi_period must be an integer from 1 through 499")
        if volume_mode not in {"cumulative", "incremental"}:
            raise ValueError("volume_mode must be cumulative or incremental")
        if timestamp_unit not in {"s", "ms", "us", "ns"}:
            raise ValueError("timestamp_unit must be s, ms, us, or ns")

        self.instrument_token = instrument_token.strip()
        self.exchange = exchange
        self.segment = segment
        self.rsi_period = rsi_period
        self.volume_mode = volume_mode
        self.timestamp_unit = timestamp_unit

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._mean = None
        self._scale = None

        if scaler_path is not None:
            with np.load(scaler_path, allow_pickle=False) as saved:
                mean = saved["mean"].copy()
                scale = saved["scale"].copy()

            if (mean.shape != (7,) or scale.shape != (7,) or not np.isfinite(mean).all() or 
                not np.isfinite(scale).all() or (scale <= 0).any()):
                raise ValueError("Scaler must contain seven finite means and positive scales")
            self._mean = mean
            self._scale = scale

        self._lock = RLock()
        self._rows = deque(maxlen=max_history)

        self._frame = pd.DataFrame(
            columns=self.FEATURES,
            index=pd.DatetimeIndex([], tz=self.TIMEZONE, name="timestamp"),
        )

        self._current = None
        self._watermark = None
        self._last_tick = None
        self._last_signature = None
        self._last_cumulative = None
        self._volume_date = None
        self._session = None
        self.late_ticks_dropped = 0

        self._turnover = 0.0
        self._volume = 0.0
        self._previous_close = None
        self._gain = None
        self._loss = None
        self._seed_gains = []
        self._seed_losses = []

    def _timestamp(self, value, *, unit=None) -> pd.Timestamp:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("Invalid tick timestamp")
        try:
            if unit is not None or isinstance(value, Real):
                number = float(value)
                if not np.isfinite(number):
                    raise ValueError("Nonfinite timestamp")
                stamp = pd.Timestamp(pd.to_datetime(value, unit=unit or self.timestamp_unit, utc=True))
            elif isinstance(value, (str, datetime, pd.Timestamp)):
                stamp = pd.Timestamp(value)
            else:
                raise ValueError("Unsupported timestamp format")

            if pd.isna(stamp):
                raise ValueError("Missing timestamp data")
            return stamp.tz_localize(self.TIMEZONE) if stamp.tzinfo is None else stamp.tz_convert(self.TIMEZONE)
        except (ValueError, TypeError, OverflowError):
            raise ValueError("Invalid tick timestamp parsing configuration") from None

    @staticmethod
    def _number(value, name, *, positive=False) -> float:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"Invalid {name}")
        try:
            number = float(value)
        except (ValueError, TypeError, OverflowError):
            raise ValueError(f"Invalid {name}") from None
        if not np.isfinite(number) or (number <= 0 if positive else number < 0):
            raise ValueError(f"Invalid boundaries for {name}")
        return number

    @staticmethod
    def _session_start(stamp):
        opening = stamp.normalize() + timedelta(hours=9, minutes=15)
        return opening if stamp >= opening else opening - timedelta(days=1)

    def _roll_session(self, stamp):
        session = self._session_start(stamp)
        if session != self._session:
            self._session = session
            self._turnover = 0.0
            self._volume = 0.0
            self._last_cumulative = None

    def _next_rsi(self, close):
        if self._previous_close is None:
            self._previous_close = close
            return np.nan

        change = close - self._previous_close
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        self._previous_close = close

        if self._gain is None:
            self._seed_gains.append(gain)
            self._seed_losses.append(loss)
            if len(self._seed_gains) < self.rsi_period:
                return np.nan
            self._gain = np.mean(self._seed_gains)
            self._loss = np.mean(self._seed_losses)
            self._seed_gains.clear()
            self._seed_losses.clear()
        else:
            self._gain = (self._gain * (self.rsi_period - 1) + gain) / self.rsi_period
            self._loss = (self._loss * (self.rsi_period - 1) + loss) / self.rsi_period

        if self._gain == self._loss == 0:
            return 50.0
        if self._loss == 0:
            return 100.0
        return 100 - 100 / (1 + self._gain / self._loss)

    def _finalize(self):
        bar = self._current.copy()
        self._roll_session(bar["timestamp"])

        typical = bar["high"] / 3 + bar["low"] / 3 + bar["close"] / 3
        turnover = self._turnover + typical * bar["volume"]
        volume = self._volume + bar["volume"]

        if not np.isfinite([turnover, volume]).all():
            raise ValueError("Session turnover/volume math overflow bounds")

        self._turnover = turnover
        self._volume = volume
        bar["rsi"] = self._next_rsi(bar["close"])
        bar["vwap"] = turnover / volume if volume > 0 else np.nan

        self._rows.append(bar)
        self._frame = pd.DataFrame.from_records(list(self._rows)).set_index("timestamp")
        self._frame.index = pd.DatetimeIndex(self._frame.index, name="timestamp")
        self._watermark = bar["timestamp"] + timedelta(minutes=1)
        self._current = None

    def on_tick(self, payload: Mapping) -> bool:
        """Consume one decoded broker snapshot dictionary packet."""
        if not isinstance(payload, Mapping):
            raise TypeError("Tick payload must be a mapping object dictionary")

        with self._lock:
            stamp = self._timestamp(payload["tsInMillis"], unit="ms") if "tsInMillis" in payload else self._timestamp(payload.get("timestamp"))
            price = self._number(payload.get("ltp", payload.get("price")), "price", positive=True)
            raw_volume = self._number(payload.get("volume"), "volume")

            signature = (stamp, price, raw_volume)
            if self.volume_mode == "cumulative" and signature == self._last_signature:
                return False

            if (self._last_tick is not None and stamp < self._last_tick) or (self._watermark is not None and stamp < self._watermark):
                # A tick stamped just before a minute boundary can legitimately arrive after the
                # timer closed that bar. Drop it (counted) instead of raising, because the feed
                # callback treats any exception as a fatal circuit trip for the whole session.
                self.late_ticks_dropped += 1
                return False

            same_session = self._session_start(stamp) == self._session
            previous = self._last_cumulative if (same_session and stamp.date() == self._volume_date) else None

            if self.volume_mode == "cumulative":
                if previous is not None and raw_volume < previous:
                    raise ValueError("Cumulative running total volume decreased paradoxically")
                delta = raw_volume - previous if previous is not None else 0.0
            else:
                delta = raw_volume

            start = stamp.floor("min")
            if self._current is not None and start > self._current["timestamp"]:
                self._finalize()

            self._roll_session(stamp)
            if self._current is None:
                self._current = {"timestamp": start, "open": price, "high": price, "low": price, "close": price, "volume": delta}
            else:
                volume = self._current["volume"] + delta
                if not np.isfinite(volume):
                    raise ValueError("Bar candle volume tracking overflow")
                self._current.update(high=max(self._current["high"], price), low=min(self._current["low"], price), close=price, volume=volume)

            self._last_tick = stamp
            self._last_signature = signature
            self._last_cumulative = raw_volume
            self._volume_date = stamp.date()

            return True

    def on_groww_update(self, metadata: Mapping, feed) -> bool:
        """Select target matching instrument identifier signatures from the callback hook stream."""
        if not isinstance(metadata, Mapping):
            raise TypeError("Groww callback metadata must be a dictionary configuration map")
        identity = (metadata.get("exchange"), metadata.get("segment"), str(metadata.get("feed_key")))
        if identity != (self.exchange, self.segment, self.instrument_token) or metadata.get("feed_type") != "ltp":
            return False
        snapshots = feed.get_ltp()
        tick = snapshots.get(self.exchange, {}).get(self.segment, {}).get(self.instrument_token)
        return False if tick is None else self.on_tick(tick)

    def flush_completed(self, now=None) -> bool:
        """Force close idle bar candles via separate chronological system timer execution loops."""
        with self._lock:
            stamp = self._timestamp(pd.Timestamp.now(tz="UTC") if now is None else now)
            if (self._last_tick is not None and stamp < self._last_tick) or (self._watermark is not None and stamp < self._watermark):
                raise ValueError("Cannot move system buffer clock backwards")
            closed = self._current is not None and stamp >= (self._current["timestamp"] + timedelta(minutes=1))
            if closed:
                self._finalize()
            self._roll_session(stamp)
            self._watermark = stamp.floor("min")
            return bool(closed)

    def get_dataframe(self) -> pd.DataFrame:
        """Return a detached memory snapshot DataFrame copy of completed indicators."""
        with self._lock:
            return self._frame.copy(deep=True)

    @property
    def session_totals(self) -> dict:
        with self._lock:
            return {
                "session_start": self._session,
                "turnover": self._turnover,
                "volume": self._volume,
                "vwap": (self._turnover / self._volume if self._volume else 0.0),
            }

    def get_current_state_tensor(self, model_features_list: Sequence[str]) -> torch.Tensor:
        """Export scaled rolling matrices formatted as (1, 60, 7) float32 system indicators."""
        if isinstance(model_features_list, str) or tuple(model_features_list) != self.FEATURES:
            raise ValueError("Feature vector formatting order sequence mismatch detected")
        with self._lock:
            if self._mean is None:
                raise RuntimeError("Load training parameters scaler before querying real-time evaluation steps")
            if len(self._frame) < 60:
                raise RuntimeError("At least 60 fully closed sequential candle bars are required")
            window = self._frame.iloc[-60:].copy(deep=True)
            if self._session is None or window.index[0] < self._session:
                raise RuntimeError("Need 60 completed candles inside the current active market session timeline")
            contiguous = (np.diff(window.index.as_unit("ns").asi8) == 60_000_000_000).all()
            if not contiguous:
                raise RuntimeError("Latest completed data rows are not consecutive 1-minute historical candles")
            values = window.loc[:, list(self.FEATURES)].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise RuntimeError("Indicators are not finite or hold un-warmed padding NaNs")
            values = (values - self._mean) / self._scale
            if not np.isfinite(values).all() or (np.abs(values) > np.finfo(np.float32).max).any():
                raise ValueError("Standardized tensor mapping value boundaries overflow system thresholds")
            state_tensor = torch.tensor(values, dtype=torch.float32, device=self.device).unsqueeze(0)
            return state_tensor


class CompletedCandleBuffer(LiveCandleBuffer):
    """Features from completed historical OHLCV; LTP is used only for prices.

    Each poll must include this entire session so Wilder RSI/VWAP keep the same
    seed as training. Missing, stale or revised candles stop the session.
    """

    def update_completed(self, candles, now):
        from data_loader import prepare_intraday_frame
        stamp = self._timestamp(now)
        opening = stamp.normalize() + timedelta(hours=9, minutes=15)
        expected_last = stamp.floor('min') - timedelta(minutes=1)
        if candles.empty:
            if expected_last >= opening:
                raise ValueError('Completed market candles are unavailable')
            return
        frame = prepare_intraday_frame(candles, rsi_period=self.rsi_period)
        if (frame.index[0] != opening or frame.index[-1] != expected_last
                or not frame.index.is_monotonic_increasing
                or (np.diff(frame.index.as_unit('ns').asi8) != 60_000_000_000).any()):
            raise ValueError('Completed candles are stale, incomplete or outside this session')
        with self._lock:
            if not self._frame.empty:
                old = self._frame.loc[:, ['open', 'high', 'low', 'close', 'volume']]
                if not old.index.isin(frame.index).all() or not np.array_equal(
                        old.to_numpy(), frame.loc[old.index, old.columns].to_numpy()):
                    raise ValueError('Previously consumed candles changed')
            self._frame = frame.loc[:, list(self.FEATURES)].tail(self._rows.maxlen).copy()
            self._session = opening
            self._watermark = expected_last + timedelta(minutes=1)

    def on_tick(self, payload):
        raise ValueError('LTP has no candle volume; use completed OHLCV')
