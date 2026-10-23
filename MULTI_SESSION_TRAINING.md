# Multiple-session fake-money training

Use this workflow after the single-session diagnostic runs. It preserves the
₹25,000 virtual starting equity, 5× simulated leverage, ₹85,000 entry cap,
1% price stop, ₹1,000 daily-loss threshold and 15:15 IST cutoff. Real trading
stays disabled. It does not add a trade-count limit, cooldown or profit target.

## Update the project

Download main from GitHub and replace project source files while preserving
`.venv`, `data` and `artifacts`. Include the new `multi_session.py`,
`prepare_training_data.py` and `train_multisession.py` files. Core dependencies
have not changed; recent public-data downloading optionally uses `yfinance`.
Do not reinstall CPU requirements over a working CUDA torch installation.

Open PowerShell inside the project folder. If yfinance is not installed:

```powershell
.\.venv\Scripts\python.exe -m pip install yfinance
```

## Prepare data without Groww

```powershell
.\.venv\Scripts\python.exe prepare_training_data.py --symbol RELIANCE --output-dir data/multi-01
```

This requests Yahoo's recent seven-day window of one-minute data. Access and
history depth are not guaranteed, and seven-day history is not long-term training
coverage. Current/future days and weekends are excluded. Each completed day uses
its longest real contiguous regular-session section of at least 150 bars. No
prices or missing minutes are filled. Invalid/duplicate rows stop preparation.
The latest eligible day is reserved as `practice_multi.csv`; the earlier days
are saved as `multi_sessions.csv`. `data_manifest.json` records the source,
selected ranges and partial sessions. Indicators start at the selected section.
The command preserves previous files by refusing an existing output directory.

At least **five eligible completed sessions** are needed: four training-pipeline
groups and a separate practice day. If unavailable, keep collecting data or use
an existing longer, single-stock one-minute OHLCV CSV:

```powershell
.\.venv\Scripts\python.exe prepare_training_data.py --csv data/longer_history.csv --output-dir data/multi-02
```

Do not concatenate different stocks, mix bar intervals, or label five-minute bars
as consecutive one-minute bars. Yahoo and Groww prices/volumes may differ.
Dates already examined during earlier experiments are development data, not an
untouched final test. Reserve later, unexamined sessions for final assessment.

## Validate forecasts, then train a policy

```powershell
.\.venv\Scripts\python.exe train_multisession.py --csv data/multi-01/multi_sessions.csv --settings simulation_settings.json --output-dir artifacts/model-multi-01 --symbol RELIANCE --lstm-epochs 30 --rl-timesteps 10000
```

Whole sessions are divided chronologically into four disjoint groups:

1. **Forecast fitting:** first ~50% of available training sessions. Features are
   scaled on these days only. LSTM windows/targets never cross days. Next-close
   return targets use basis points, and the output head starts at a zero forecast.
2. **Forecast validation:** next ~15%, at least one day. Checkpoint selection and
   early stopping use only this group. No training-day or test-day score chooses
   an epoch. Checkpoints include epoch zero so an unhelpful training update need
   not replace the no-change initialization.
3. **Policy learning:** next ~20%, at least one later day. Forecast weights/scaler
   stay frozen. PPO (or optional DQN) rotates these days as independent episodes,
   with fresh virtual equity and risk state at each episode reset.
4. **Evaluation:** remaining later days. This data does not fit the scaler,
   forecast or policy, or select checkpoints. Evaluation carries virtual equity
   between days, resets the daily loss gate each day, and records per-day results.

Four to five sessions are a pipeline check, not adequate evidence of general
performance. With only five training sessions the policy group has one day;
additional training steps repeat that day rather than add market history.

Forecast validation must beat **both** zero-return and fitting-data mean-return
baselines in MSE. If it fails, the command saves `forecast_report.json`,
`lstm.pt` and `feature_scaler.npz`, prints why policy training was skipped, and
exits with code 2. This is a deliberate quality gate, not an installation error.
There is no PPO file to paper-trade in that case. Share the forecast report:

```powershell
Get-Content .\artifacts\model-multi-01\forecast_report.json
```

A diagnostic-only `--allow-weak-forecast` override exists for research/tests. It
is not recommended to bypass failed validation merely to obtain a trading model.
Passing the gate does not prove a profitable forecast or strategy.

If policy training completes, inspect `run_metrics.json`. Forecast metrics are
reported in fractional-return units, despite fitting in basis points. The saved
`forecast_transform` makes paper inference divide the LSTM output by 10,000,
exactly as policy training did. Existing version-4 models without this transform
continue using their original fractional-return outputs.

Windows with a GTX 1650 use batches of 32 and keep sequence datasets on CPU;
only the active batch is transferred to the GPU. No whole multi-day window tensor
is stored on the GPU. All phases can also run on CPU.

## Paper-test the reserved session

Only after policy training succeeds:

```powershell
.\.venv\Scripts\python.exe paper_trade.py --csv data/multi-01/practice_multi.csv --model-dir artifacts/model-multi-01 --symbol RELIANCE --output-dir artifacts/paper-multi-01
Get-Content .\artifacts\model-multi-01\run_metrics.json
Get-Content .\artifacts\paper-multi-01\paper_report.json
```

Review net results after costs, drawdown, action counts, actual quantities and
forced exits. The strategy may still prefer inactivity or lose money. Nothing
in this workflow forces purchases or increases risk limits, and no paper result
switches on real trading. These leveraged models remain incompatible with the
current one-share live adapter.
