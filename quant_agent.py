"""Two-layer LSTM for a next-candle scalar regression score.

Weights start untrained. Define the next-candle target during training (for
example, next close or next return). Fit any feature/target scaler on training
data only and reuse it at inference; no scaling is learned from a live window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn


class QuantLSTMBrain(nn.Module):
    """Map (batch, 60, 7) float tensors to (batch, 1) unbounded scores."""

    FEATURES = ("open", "high", "low", "close", "volume", "rsi", "vwap")
    SEQUENCE_LENGTH = 60

    def __init__(self, hidden_size: int = 64, dropout: float = 0.0):
        super().__init__()
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.lstm = nn.LSTM(
            input_size=len(self.FEATURES),
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        if not isinstance(sequences, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if sequences.ndim != 3 or tuple(sequences.shape[1:]) != (self.SEQUENCE_LENGTH, len(self.FEATURES)) or sequences.shape[0] == 0:
            raise ValueError("Expected nonempty input with shape (batch, 60, 7)")
        parameter = self.linear.weight
        if sequences.dtype != parameter.dtype or sequences.device != parameter.device:
            raise ValueError("Input dtype and device must match the model")
        if not torch.isfinite(sequences).all():
            raise ValueError("Input features must be finite; remove RSI warm-up before constructing windows")
        outputs, _ = self.lstm(sequences)
        return self.linear(outputs[:, -1, :])

    def prepare_sequence(self, candles: pd.DataFrame) -> torch.Tensor:
        """Convert the last 60 contiguous one-minute candles to (1, 60, 7)."""
        if not isinstance(candles, pd.DataFrame):
            raise TypeError("candles must be a pandas DataFrame")
        if candles.columns.has_duplicates or not set(self.FEATURES).issubset(candles.columns):
            raise ValueError("Expected unique columns including open/high/low/close/volume/rsi/vwap")
        if len(candles) < self.SEQUENCE_LENGTH:
            raise ValueError("At least 60 candles are required")
        if not isinstance(candles.index, pd.DatetimeIndex) or candles.index.hasnans or not candles.index.is_monotonic_increasing or candles.index.has_duplicates:
            raise ValueError("Candles require a sorted, unique DatetimeIndex without missing timestamps")
        window = candles.iloc[-self.SEQUENCE_LENGTH:]
        if not (np.diff(window.index.as_unit("ns").asi8) == 60_000_000_000).all():
            raise ValueError("Last 60 candles must have consecutive one-minute timestamps")
        try:
            values = window.loc[:, list(self.FEATURES)].to_numpy(dtype=np.float64, copy=True)
        except (ValueError, TypeError):
            raise ValueError("Features must be numeric") from None
        if not np.isfinite(values).all():
            raise ValueError("Last 60 candles contain missing/nonfinite indicators or prices")
        parameter = self.linear.weight
        tensor = torch.as_tensor(values, dtype=parameter.dtype, device=parameter.device).unsqueeze(0)
        if not torch.isfinite(tensor).all():
            raise ValueError("Features overflow the model dtype")
        return tensor
