"""Automated historical data ingestion and technical indicator computation.

Handles file validation, explicit timezone normalization to Asia/Kolkata,
Wilder's RSI smoothing arrays, and the daily 09:15 AM VWAP rolling accumulation reset.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import numpy as np
import pandas as pd

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.json"


@dataclass(frozen=True)
class GrowwCredentials:
    api_key: str | None = None
    api_secret: str | None = None
    access_token: str | None = None
    totp_secret: str | None = None


def load_settings(path: str | Path = DEFAULT_SETTINGS_PATH) -> dict:
    """Read non-secret configuration layout without modifying the file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found at: {p}")
    with p.open(encoding="utf-8") as stream:
        settings = json.load(stream)
    if not isinstance(settings, dict):
        raise ValueError("Settings must be a JSON object")
    if not isinstance(settings.get("groww"), dict):
        raise ValueError("Settings must contain a groww object")
    return settings


def load_credentials(
    settings_path: str | Path = DEFAULT_SETTINGS_PATH,
    *,
    environ: Mapping[str, str] | None = None,
    required: tuple[str, ...] = ("access_token",),
) -> GrowwCredentials:
    """Resolve configured variable names; report missing names, never secret values."""
    config = load_settings(settings_path)["groww"]
    source = os.environ if environ is None else environ
    fields = ("api_key", "api_secret", "access_token", "totp_secret")
    if any(name not in fields for name in required):
        raise ValueError("Unknown credential field requested")
    values = {}
    missing = []
    for name in fields:
        variable = config.get(f"{name}_env")
        if not isinstance(variable, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", variable):
            raise ValueError(f"Invalid environment variable mapping for {name}")
        value = source.get(variable)
        values[name] = value if value and value.strip() else None
        if name in required and values[name] is None:
            missing.append(variable)
    if missing:
        raise ValueError("Missing required environment variables: " + ", ".join(missing))
    return GrowwCredentials(**values)


def _wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    """Seed Wilder smoothing with the first period's mean price changes."""
    changes = close.diff()
    gains = changes.clip(lower=0).to_numpy()
    losses = (-changes.clip(upper=0)).to_numpy()
    result = np.full(len(close), np.nan)
    if len(close) <= period:
        return pd.Series(result, index=close.index)
    gain, loss = gains[1:period + 1].mean(), losses[1:period + 1].mean()
    for i in range(period, len(close)):
        if i > period:
            gain = (gain * (period - 1) + gains[i]) / period
            loss = (loss * (period - 1) + losses[i]) / period
        result[i] = 50.0 if gain == loss == 0 else (100.0 if loss == 0 else 100 - 100 / (1 + gain / loss))
    return pd.Series(result, index=close.index)


def load_intraday_csv(
    path: str | Path,
    *,
    timezone: str = "Asia/Kolkata",
    rsi_period: int = 14,
    column_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Return sorted OHLCV candles with rolling indicators, safe from lookahead leaks."""
    if isinstance(rsi_period, bool) or not isinstance(rsi_period, int) or rsi_period < 1:
        raise ValueError("rsi_period must be a positive integer")
    try:
        ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        raise ValueError(f"Invalid timezone configuration: {timezone}")
        
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Intraday data sheet not found at: {p}")
        
    frame = pd.read_csv(p)
    frame.columns = frame.columns.str.strip().str.lower()
    if column_map:
        frame = frame.rename(columns={k.strip().lower(): v.strip().lower() for k, v in column_map.items()})
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names noticed in file fields")
    columns = ["open", "high", "low", "close", "volume"]
    if not {"timestamp", *columns}.issubset(frame.columns):
        raise ValueError("CSV requires timestamp, open, high, low, close, volume columns")
    if frame.empty:
        raise ValueError("CSV contains no candles to process")
        
    raw_times = frame["timestamp"]
    if not raw_times.map(lambda value: isinstance(value, str)).all():
        raise ValueError("Timestamps must be strings, not numeric epochs or missing values")
        
    try:
        timestamps = pd.DatetimeIndex(pd.to_datetime(raw_times, format="mixed", errors="raise"))
        if timestamps.hasnans:
            raise ValueError("Missing timestamps")
        if timestamps.tz is None:
            timestamps = timestamps.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
        else:
            timestamps = timestamps.tz_convert(timezone)
    except (ValueError, TypeError):
        raise ValueError("Invalid, mixed-timezone, or ambiguous timestamps") from None
        
    frame = frame[columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("OHLCV must contain finite numeric values")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any() or (frame.volume < 0).any():
        raise ValueError("Prices must be positive and volume nonnegative")
    if ((frame.high < frame[["open", "low", "close"]].max(axis=1)) |
        (frame.low > frame[["open", "high", "close"]].min(axis=1))).any():
        raise ValueError("Candle high/low bounds are inconsistent")
        
    frame.index = timestamps.rename("timestamp")
    if frame.index.has_duplicates:
        raise ValueError("Duplicate candle timestamps; load each instrument separately")
        
    frame = frame.sort_index()
    frame["rsi"] = _wilder_rsi(frame.close, rsi_period)
    sessions = frame.index.normalize()
    typical = (frame.high + frame.low + frame.close) / 3
    cumulative_volume = frame.volume.groupby(sessions).cumsum()
    frame["vwap"] = (typical * frame.volume).groupby(sessions).cumsum() / cumulative_volume.replace(0, np.nan)
    return frame
