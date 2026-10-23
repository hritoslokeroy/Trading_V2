"""Offline training entry point: python main.py --help.

Partitions indicators data sequentially into 60% LSTM / 20% RL / 20% Rollout Eval
to maintain absolute zero out-of-sample data leakage boundaries.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import random
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from quant_agent import QuantLSTMBrain
from rl_brain import RLTradingBrain, TradingEnvironment
from risk_manager import DEFAULT_SETTINGS_PATH
from data_loader import load_intraday_csv, load_settings


def train_pipeline(
    csv_path: str | Path, settings_path: str | Path, output_dir: str | Path,
    *, symbol: str, algorithm: str = "PPO", lstm_epochs: int = 10,
    rl_timesteps: int = 2048, seed: int = 42,
) -> dict:
    """Train, evaluate, and export models, scaling metrics, and logs snapshot."""
    for name, value in (("lstm_epochs", lstm_epochs), ("rl_timesteps", rl_timesteps)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    if algorithm not in RLTradingBrain.ALGORITHMS:
        raise ValueError("algorithm must be PPO or DQN")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Choose a new output directory; existing parameters are preserved")

    settings = load_settings(settings_path)
    if settings.get("execution", {}).get("mode", "simulation") != "simulation":
        raise ValueError("Training requires simulation settings; use simulation_settings.json")
    if settings["risk_limits"].get("trading_enabled") is not True:
        raise ValueError("Use a separate simulation settings file with explicit limits and trading_enabled=true")

    data_config = settings.get("data", {})
    frame = load_intraday_csv(csv_path, timezone=data_config.get("timezone", "Asia/Kolkata"),
                              rsi_period=data_config.get("rsi_period", 14))

    ready = frame.dropna(subset=list(QuantLSTMBrain.FEATURES)).copy()
    if len(ready) < 120:
        raise ValueError("At least 120 indicator-ready candles are required")

    contiguous = (np.diff(ready.index.as_unit("ns").asi8) == 60_000_000_000).all()
    if len(set(ready.index.date)) != 1 or not contiguous:
        raise ValueError("This baseline requires one contiguous one-minute local trading session")

    train_end = int(len(ready) * 0.6)
    rl_end = int(len(ready) * 0.8)

    probe = ready.iloc[train_end:rl_end].assign(lstm_score=0.0)
    TradingEnvironment(probe, symbol=symbol, settings_path=settings_path).close()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuantLSTMBrain().to(device)

    features = ready.loc[:, list(model.FEATURES)].to_numpy(dtype=np.float64)
    scaler = StandardScaler().fit(features[:train_end])
    scaled = scaler.transform(features).astype(np.float32)

    if not np.isfinite(scaled).all():
        raise ValueError("Scaled features are not finite")

    sequence_length = model.SEQUENCE_LENGTH
    closes = ready.close.iloc[:train_end].to_numpy(dtype=np.float64)
    ends = np.arange(sequence_length - 1, len(closes) - 1)

    feature_tensor = torch.as_tensor(scaled, dtype=torch.float32, device=device)
    x = torch.stack([feature_tensor[end - sequence_length + 1:end + 1] for end in ends])

    targets = closes[ends + 1] / closes[ends] - 1
    y = torch.as_tensor(targets, dtype=torch.float32, device=device).unsqueeze(1)

    expected_x_shape = (train_end - sequence_length, sequence_length, len(model.FEATURES))
    if x.shape != expected_x_shape or y.shape != (len(x), 1):
        raise ValueError("Supervised window/target alignment failed")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(lstm_epochs):
        model.train()
        batches = torch.randperm(len(x), device=device).split(32)
        for indices in batches:
            optimizer.zero_grad()
            predictions = model(x[indices])
            loss = torch.nn.functional.mse_loss(predictions, y[indices])
            if not torch.isfinite(loss):
                raise ValueError("LSTM training diverged")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()

    model.eval()
    with torch.inference_mode():
        supervised_mse = torch.nn.functional.mse_loss(model(x), y).item()
        scores = []
        for end in range(sequence_length - 1, len(ready)):
            inputs = feature_tensor[end - sequence_length + 1:end + 1].unsqueeze(0)
            score = model(inputs).item()
            if not np.isfinite(score):
                raise ValueError("LSTM inference produced a nonfinite score")
            scores.append(score)

    ready["lstm_score"] = np.nan
    ready.loc[ready.index[sequence_length - 1:], "lstm_score"] = scores

    training_env = TradingEnvironment(ready.iloc[train_end:rl_end], symbol=symbol, settings_path=settings_path)
    evaluation_env = TradingEnvironment(ready.iloc[rl_end:], symbol=symbol, settings_path=settings_path)

    if algorithm == "PPO":
        model_kwargs = {"n_steps": 32, "batch_size": 16, "n_epochs": 2}
    else:
        model_kwargs = {
            "learning_starts": min(100, rl_timesteps // 4),
            "buffer_size": max(1000, rl_timesteps),
            "batch_size": 32,
            "train_freq": 1,
            "gradient_steps": 1,
        }

    brain = RLTradingBrain(training_env, algorithm=algorithm, seed=seed, device=str(device), model_kwargs=model_kwargs)
    rl_agent = brain.model
    rl_agent.learn(total_timesteps=rl_timesteps)

    observation, info = evaluation_env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    
    buy_count = 0
    sell_count = 0
    forced_exit_count = 0
    total_executed_trades = 0

    while True:
        action, _ = rl_agent.predict(observation, deterministic=True)
        action = int(np.asarray(action).item())
        
        observation, reward, terminated, truncated, info = evaluation_env.step(action)
        total_reward += reward
        steps += 1
        buy_count += info["buy_count"]
        sell_count += info["sell_count"]
        forced_exit_count += info["forced_exit_count"]
        total_executed_trades += info["total_executed_trades"]
        if terminated or truncated:
            break

    report = {
        "simulator_version": training_env.simulator_version,
        "execution_profile": training_env.execution_profile(), "timezone": "Asia/Kolkata",
        "seed": seed, "symbol": symbol, "algorithm": algorithm, "device": str(device),
        "target": "next_close_fractional_return", "features": list(model.FEATURES),
        "raw_candles": len(frame), "indicator_ready_candles": len(ready),
        "split_counts": {"lstm": train_end, "rl": rl_end - train_end, "evaluation": len(ready) - rl_end},
        "supervised_samples": len(x), "lstm_train_mse": supervised_mse, "lstm_scored_rows": len(scores),
        "sequence_length": sequence_length, "rl_requested_timesteps": rl_timesteps, "rl_actual_timesteps": rl_agent.num_timesteps,
        "evaluation": {
            "steps": steps, "equity": info["equity"], "daily_pnl": info["daily_pnl"], "cash": info["cash"],
            "buy_trades": buy_count, "sell_trades": sell_count, "forced_exits": forced_exit_count,
            "total_executed_trades": total_executed_trades, "reward": total_reward, "voluntary_trades": info["trade_count"],
            "total_charges": info["total_charges"], "charge_breakdown": info["charge_breakdown"],
            "executions": evaluation_env.executions,
        },
        "charge_assumptions": {"rates": asdict(evaluation_env.charges),
                               "aggregation": "session", "rounding": "paise_half_up_per_component",
                               "status": "estimate; verify current tariff and contract-note rounding"},
        "boundaries": {
            "lstm_end": ready.index[train_end - 1].isoformat(), "rl_start": ready.index[train_end].isoformat(),
            "rl_end": ready.index[rl_end - 1].isoformat(), "evaluation_start": ready.index[rl_end].isoformat(),
        }
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    torch.save({name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}, output_dir / "lstm.pt")
    np.savez(output_dir / "feature_scaler.npz", mean=scaler.mean_, scale=scaler.scale_)
    rl_agent.save(output_dir / "rl_policy")

    (output_dir / "run_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    training_env.close()
    evaluation_env.close()
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS_PATH)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--algorithm", choices=["PPO", "DQN"], default="PPO")
    parser.add_argument("--lstm-epochs", type=int, default=10)
    parser.add_argument("--rl-timesteps", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    try:
        report = train_pipeline(args.csv, args.settings, args.output_dir, symbol=args.symbol,
                                algorithm=args.algorithm, lstm_epochs=args.lstm_epochs,
                                rl_timesteps=args.rl_timesteps, seed=args.seed)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Training failed: {exc}\n")

    evaluation = report["evaluation"]
    print(f"Final simulated cash balance: INR {evaluation['cash']:.2f}")
    print(f"Executed trades: {evaluation['total_executed_trades']} (buys={evaluation['buy_trades']}, sells={evaluation['sell_trades']}, forced exits={evaluation['forced_exits']}, voluntary fills={evaluation['voluntary_trades']})")
    print(f"Estimated charges: INR {evaluation['total_charges']:.2f}")
    print("Charges are estimates; verify the current broker tariff before live use.")
    print(f"Artifacts saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
