# 50,000 INR diagnostic paper experiment

This is a separate preset, leaving the previous 25,000/85,000 configuration
available to reproduce earlier results. The new settings are:

| Setting | Value |
| --- | ---: |
| Virtual starting equity | INR 50,000 |
| Simulated leverage | 5x |
| Maximum entry notional | INR 250,000 |
| Positions | One long RELIANCE position |
| Quantity | Maximum affordable whole shares within the entry cap |
| Price stop | 1% |
| Daily net-loss threshold | INR 1,000 |
| Cutoff | 15:15 IST |

The 250,000 cap uses the full capital-times-leverage interpretation of the
request for more shares. It is a new explicit preset, not a silent edit of the
previous 85,000 cap. Entry margin and charges must fit within virtual capital,
so actual notional is slightly below the theoretical maximum. Exit proceeds
may exceed the entry cap after appreciation; exits are never blocked by it.

The daily threshold remains 1,000, not 5,000. At near 250,000 exposure a 0.4%
adverse move is already about 1,000 before costs, so the daily-loss control can
exit sooner than the 1% stop. Gaps, slippage and exit costs can exceed the loss
threshold. These are simulated controls, not broker-guaranteed maximum losses.
No leverage or limit is changed in real-money settings; real trading stays disabled.

## Update and train offline

Update project source files from main and include `simulation_50k_settings.json`
and `paper_50k_settings.json`. Preserve `.venv`, `data` and `artifacts`. There are
no new dependencies. Use the existing snapshot, keeping the reserved later day
outside fitting and policy learning.

The current forecasts failed validation. Changing capital does not fix that.
If you explicitly want to practise the policy/accounting workflow despite that
result, the following is a **diagnostic fake-money experiment** using the existing
`--allow-weak-forecast` override. It does not assert a validated strategy or bypass
any real-money activation checks. It never rewards trades simply for happening.

```powershell
.\.venv\Scripts\python.exe train_multisession.py --csv data/multi-02/multi_sessions.csv --settings simulation_50k_settings.json --output-dir artifacts/model-paper50k-01 --symbol RELIANCE --lstm-epochs 30 --rl-timesteps 10000 --allow-weak-forecast
```

To retain the strict forecast gate, omit `--allow-weak-forecast`. In that case
PPO will be skipped if forecast validation fails again. Learning occurs here in
offline simulation. The subsequent paper runner freezes model weights; it does
not continuously learn from each new trade or automatically switch to real money.

## Replay with the matching paper preset

After policy training completes successfully:

```powershell
.\.venv\Scripts\python.exe paper_trade.py --csv data/multi-02/practice_multi.csv --model-dir artifacts/model-paper50k-01 --settings paper_50k_settings.json --symbol RELIANCE --output-dir artifacts/paper50k-01
Get-Content .\artifacts\model-paper50k-01\run_metrics.json
Get-Content .\artifacts\paper50k-01\paper_report.json
```

Both commands must use the new explicit settings. Existing models trained for
the 85,000 entry cap are refused by the 250,000 paper profile rather than silently
resized. Use new output folders; do not overwrite previous experiments.
Reports include initial capital, leverage, position caps, actual quantities,
costs and losses. Paper reports now also disclose `forecast_validation_passed`
and `forecast_validation_override_used`. Missing validation information in older
artifacts is reported as unknown, not invented as a pass.

More shares do not make a weak model profitable. A zero-trade, losing or
validation-bypassed result remains research evidence only. These leveraged
version-4 artifacts cannot activate the current one-share live adapter. No
Groww credentials or connection are required for the CSV workflow.
