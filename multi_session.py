"""Day-isolated data, forecast scaling and RL episodes for offline research."""
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from data_loader import prepare_intraday_frame
from quant_agent import QuantLSTMBrain
from rl_brain import TradingEnvironment

TRANSFORM = {'output_units': 'basis_points', 'divide_by': 10000}


def forecast_divisor(metadata):
    transform = metadata.get('forecast_transform')
    if transform is None:
        if metadata.get('training_pipeline') == 'multi_session_v1':
            raise ValueError('Multi-session model is missing its forecast transform')
        return 1.0
    if transform != TRANSFORM:
        raise ValueError('Unsupported forecast transform; retrain rather than guess units')
    return 10000.0


def load_sessions(csv_path, settings):
    raw = pd.read_csv(csv_path)
    if 'timestamp' not in raw:
        raise ValueError('CSV needs timestamp,open,high,low,close,volume')
    stamps = pd.DatetimeIndex(pd.to_datetime(raw.timestamp, format='mixed', errors='raise'))
    if stamps.tz is None:
        stamps = stamps.tz_localize('Asia/Kolkata')
    else:
        stamps = stamps.tz_convert('Asia/Kolkata')
    if stamps.has_duplicates or stamps.hasnans or not stamps.is_monotonic_increasing:
        raise ValueError('CSV timestamps must be sorted and unique')
    sessions = []
    for day in sorted(set(stamps.date)):
        frame = prepare_intraday_frame(raw.loc[stamps.date == day],
            rsi_period=settings.get('data', {}).get('rsi_period', 14))
        # Indicators reset separately each day; overnight prices never seed RSI.
        ready = frame.dropna(subset=list(QuantLSTMBrain.FEATURES))
        if len(ready) < 120:
            raise ValueError(f'{day}: need at least 120 indicator-ready consecutive minute bars')
        if (len(set(frame.index.date)) != 1 or (frame.index.dayofweek >= 5).any()
                or (frame.index.strftime('%H:%M') < '09:15').any()
                or (frame.index.strftime('%H:%M') > '15:29').any()
                or (frame.index.second != 0).any() or (frame.index.microsecond != 0).any()
                or (frame.index.nanosecond != 0).any()
                or not (np.diff(frame.index.as_unit('ns').asi8) == 60_000_000_000).all()):
            raise ValueError(f'{day}: use one contiguous regular-session minute section; do not fill gaps')
        if not (frame.index.strftime('%H:%M') < '15:15').any():
            raise ValueError(f'{day}: no bars before the trading cutoff')
        sessions.append(ready.copy())
    return sessions


def split_sessions(sessions):
    if len(sessions) < 4:
        raise ValueError('Need at least four distinct sessions: fit, validation, RL and evaluation')
    fit_end = max(1, int(len(sessions) * .5))
    val_end = fit_end + max(1, int(len(sessions) * .15))
    rl_end = val_end + max(1, int(len(sessions) * .2))
    if rl_end >= len(sessions):
        raise ValueError('Need a later evaluation session outside all training groups')
    return {'fit': sessions[:fit_end], 'validation': sessions[fit_end:val_end],
            'rl': sessions[val_end:rl_end], 'evaluation': sessions[rl_end:]}


class SessionWindows(Dataset):
    """Lazy CPU windows; neither features nor next-close targets cross a day."""
    def __init__(self, sessions, scaler):
        self.arrays, self.targets, self.samples = [], [], []
        length = QuantLSTMBrain.SEQUENCE_LENGTH
        for session_id, frame in enumerate(sessions):
            values = scaler.transform(frame.loc[:, list(QuantLSTMBrain.FEATURES)].to_numpy(dtype=float)).astype(np.float32)
            targets = (frame.close.shift(-1) / frame.close - 1).to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError('Feature scaling overflow')
            self.arrays.append(values)
            self.targets.append(targets)
            self.samples.extend((session_id, end) for end in range(length-1, len(frame)-1))
        if not self.samples:
            raise ValueError('No same-session supervised windows')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        session, end = self.samples[index]
        length = QuantLSTMBrain.SEQUENCE_LENGTH
        return (torch.from_numpy(self.arrays[session][end-length+1:end+1]),
                torch.tensor([self.targets[session][end] * 10000], dtype=torch.float32))


def score_session(model, frame, scaler, device):
    values = scaler.transform(frame.loc[:, list(QuantLSTMBrain.FEATURES)].to_numpy(dtype=float)).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Feature scaling overflow')
    length = QuantLSTMBrain.SEQUENCE_LENGTH
    scores = []
    model.eval()
    with torch.inference_mode():
        for first in range(0, len(frame)-length+1, 32):
            batch = np.stack([values[start:start+length] for start in
                              range(first, min(first+32, len(frame)-length+1))])
            predictions = model(torch.as_tensor(batch, device=device)).cpu().numpy().reshape(-1) / 10000
            if not np.isfinite(predictions).all():
                raise ValueError('Nonfinite forecast')
            scores.extend(predictions.tolist())
    result = frame.iloc[length-1:].copy()
    result['lstm_score'] = scores
    return result


class SessionEnvironment(gym.Env):
    """Rotate training days, resetting virtual money/risk only between episodes."""
    def __init__(self, sessions, symbol, settings_path):
        super().__init__()
        self.sessions, self.symbol, self.settings_path = sessions, symbol, settings_path
        self.index = -1
        self.env = None
        probe = TradingEnvironment(sessions[0], symbol=symbol, settings_path=settings_path)
        self.action_space, self.observation_space = probe.action_space, probe.observation_space
        self.profile, self.simulator_version = probe.execution_profile(), probe.simulator_version
        probe.close()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.env is not None:
            self.env.close()
        self.index = (self.index+1) % len(self.sessions)
        self.env = TradingEnvironment(self.sessions[self.index], symbol=self.symbol,
                                      settings_path=self.settings_path)
        return self.env.reset(seed=seed)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        if self.env is not None:
            self.env.close()
