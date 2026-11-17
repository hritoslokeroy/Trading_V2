# Paper PPO with independent market inputs

The previous saved LSTM was the neutral epoch-zero checkpoint. With four inputs,
a flat account then supplied the same `[0, 1, 0, 0]` state on every decision. A
policy could not condition its entry decision on market changes. Increasing
capital or repeating that training did not correct this design limitation.

Simulator version 5 adds nine causal inputs independently of the LSTM score:
1-minute return, 5-minute return, trailing 10-minute volatility, candle body and
range, VWAP distance, log-relative volume, centered RSI and session time. It
retains normalized forecast, cash, position count and open P&L: **13 inputs**.
Prices and volume are derived from completed bars only. Indicators/context are
prepared separately for a whole day before forecast warmup or partition slicing.
This prevents overnight windows, future normalization and lost lookback history.
Percentage/relative units use fixed definitions, not scalers fitted on practice.

Rewards, charges, penalties, stops and sizing are unchanged. There is no reward
for merely buying, penalty added for inactivity, or forced purchase rule. Zero
trades can still be the result; varying observations are not proof of an edge.
The gatekeeper remains independent of the model's proposed actions.

## Update files

Copy updated source files from main while preserving `.venv`, `data`, and
`artifacts`. Include `market_context.py`, `simulation_market_50k_settings.json`
and `paper_market_50k_settings.json`. No dependency reinstall is needed.
Keep previous configurations and models as comparisons. Old 4-input checkpoints
cannot run in the 13-input preset. Metadata version, observation names, dimensions
and profile must match; otherwise the paper runner refuses startup.

The new preset retains INR 50,000 virtual capital, 5x simulated leverage,
INR 250,000 maximum entry notional, one long position, 1% stop, INR 1,000 daily
net-loss threshold and 15:15 IST cutoff. Entry margin plus fees must fit within
cash. The live adapter cannot use this version; real trading remains disabled.

## Train a new diagnostic model

Use the existing frozen snapshot so the change can be compared with earlier
results. This does not make previously inspected evaluation dates untouched.

```powershell
.\.venv\Scripts\python.exe train_multisession.py --csv data/multi-02/multi_sessions.csv --settings simulation_market_50k_settings.json --output-dir artifacts/model-market50k-01 --symbol RELIANCE --lstm-epochs 30 --rl-timesteps 10000 --allow-weak-forecast
```

The weak-forecast override remains explicit because the auxiliary LSTM previously
failed validation. Independent market context now reaches PPO even if that score
is zero. This is a diagnostic policy experiment, not a validation pass or live
activation. The command does not weaken risk controls or make the paper runner
learn online. Omit the override to keep the original strict forecast gate.

Look for `simulator_version: 5`, 13 policy observation names, and
`policy_input_diagnostics`. Its `varying_market_inputs` and ranges are calculated
from RL fitting sessions only. `forecast_identically_zero` can be true while
market inputs vary: that is precisely the limitation this version addresses.

## Paper replay on the existing development comparison day

After successful training:

```powershell
.\.venv\Scripts\python.exe paper_trade.py --csv data/multi-02/practice_multi.csv --model-dir artifacts/model-market50k-01 --settings paper_market_50k_settings.json --symbol RELIANCE --output-dir artifacts/paper-market50k-01
Get-Content .\artifacts\paper-market50k-01\paper_report.json
```

Use both matching new settings files. Existing model and paper directories are
preserved. Choose fresh names if any proposed output path exists.

Inspect action proposals separately from actual orders and `sell_while_flat`,
plus costs, net P&L and drawdown. Independent inputs allow market-dependent
decisions; they do not guarantee purchases or profitable behaviour. Preserve a
newer, unexamined session before any final performance claim. Continue the data
archive separately; don't fabricate candles or repeatedly tune on a claimed
untouched test. No Groww credentials are required for this CSV workflow.
