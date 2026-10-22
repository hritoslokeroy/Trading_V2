"""Offline training smoke test: synthetic one-session CSV -> train_pipeline -> checkpoint checks.

Run from the repository root:
    python test_training_smoke.py

Uses a temporary settings file and temporary output folders. No network, no broker.
This checks that the pipeline runs and its artifacts are loadable and compatible with the
live adapter's checkpoint checks. It says nothing about strategy quality.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN, PPO

REPOSITORY_ROOT = Path(__file__).resolve().parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from quant_agent import QuantLSTMBrain
from main import train_pipeline


def _write_session_csv(path: Path, candles: int = 300) -> None:
    rng = np.random.default_rng(7)
    stamps = pd.date_range("2026-10-20 09:15:00", periods=candles, freq="min")
    close = 100 + np.cumsum(rng.normal(0, 0.15, candles))
    open_ = np.r_[close[0], close[:-1]]
    frame = pd.DataFrame({
        "timestamp": stamps.strftime("%Y-%m-%d %H:%M:%S"),
        "open": open_,
        "high": np.maximum(open_, close) + 0.05,
        "low": np.minimum(open_, close) - 0.05,
        "close": close,
        "volume": rng.integers(500, 3000, candles),
    })
    frame.to_csv(path, index=False)


class TrainingSmokeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

        self.csv = self.root / "session.csv"
        _write_session_csv(self.csv)

        self.settings = self.root / "settings.json"
        self.settings.write_text(json.dumps({
            "groww": {
                f"{name}_env": f"UNUSED_{name.upper()}"
                for name in ("api_key", "api_secret", "access_token", "totp_secret")
            },
            "risk_limits": {
                "trading_enabled": True,
                "max_order_value_inr": 1000,
                "max_position_value_inr": 1000,
                "max_daily_loss_inr": 500,
                "max_open_positions": 1,
                "mandatory_stop_loss_pct": 1.0,
                "hard_cutoff_time": "15:15",
            },
            "data": {
                "timezone": "Asia/Kolkata",
                "rsi_period": 14,
                "supported_symbols": ["RELIANCE"],
            },
        }))

    def test_pipeline_runs_and_artifacts_are_loadable(self):
        for algorithm, loader in (("PPO", PPO), ("DQN", DQN)):
            with self.subTest(algorithm=algorithm):
                out = self.root / f"out_{algorithm}"
                report = train_pipeline(
                    self.csv, self.settings, out,
                    symbol="RELIANCE", algorithm=algorithm,
                    lstm_epochs=1, rl_timesteps=64, seed=3,
                )

                for name in ("lstm.pt", "feature_scaler.npz", "rl_policy.zip", "run_metrics.json"):
                    self.assertTrue((out / name).exists(), name)

                self.assertEqual(report["simulator_version"], 2)
                self.assertEqual(report["timezone"], "Asia/Kolkata")
                self.assertEqual(report["algorithm"], algorithm)
                self.assertEqual(report["features"], list(QuantLSTMBrain.FEATURES))
                self.assertGreater(report["evaluation"]["steps"], 0)

                # Same checks the live adapter applies when loading a checkpoint.
                brain = QuantLSTMBrain()
                brain.load_state_dict(torch.load(out / "lstm.pt", weights_only=True))
                policy = loader.load(out / "rl_policy.zip", device="cpu")
                self.assertEqual(policy.observation_space.shape, (4,))
                self.assertEqual(policy.action_space.n, 3)

                with np.load(out / "feature_scaler.npz", allow_pickle=False) as saved:
                    self.assertEqual(saved["mean"].shape, (7,))
                    self.assertTrue((saved["scale"] > 0).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
