# Cost-aware holding-horizon research

This command is a research tool, not a live trading program or a deployable AI.
It tests a simple regularized forecasting model before committing to another
neural-network/PPO experiment. It does not require Groww access.

## Run

Use the exported fitting/validation-only file. Preserve all later evaluation
and practice data and use a fresh output folder:

```powershell
.\.venv\Scripts\python.exe horizon_experiment.py --csv data/forecast_review.csv --settings simulation_settings.json --fit-end 2026-10-29 --validation-end 2026-11-03 --output-dir artifacts/horizon-research-01
Get-Content .\artifacts\horizon-research-01\horizon_report.json
```

Core dependencies are unchanged. Copy `horizon_experiment.py` when updating source
files. The command creates a report only. No model is automatically connected to
paper/live execution, and no real account calls or orders occur.

## Fixed protocol

- Compare 1-minute control, 5-minute and 15-minute holding periods.
- Use causal `relative_v1` inputs and a StandardScaler/Ridge model with fixed
  alpha 1,000, identical across horizons. There is no alpha/threshold search.
- Predict entry-open to exit-open **market return**. At a completed close, entry
  would fill at the next minute's open, and planned exit is H minutes after that
  entry. Targets never cross overnight or extend after the 15:15 cutoff.
- Fit only earlier original fitting dates; score the final three fitting dates
  as expanding chronological development folds. Scalers, means and model weights
  never fit that fold's scoring day.
- A forecast sample is not a trade. Overlapping forecast labels are allowed for
  regression but are never summed as if they were independent trading profits.
- Replay at most one long position. Entry requires forecast gross return to
  exceed a known-price estimate of round-trip charges plus slippage. Flat/negative
  forecasts do not cause purchases. The estimate uses the currently known close,
  not future entry or exit prices.
- The existing simulator performs actual estimated fills, whole-share sizing,
  session-ledger charges, 2 bps per-leg slippage, 1% stop, daily net-loss checks,
  and cutoff/end-of-data liquidation. A stop or loss halt can exit before H.
- Replay equity carries across scoring days, while the daily-loss gate resets
  each day. Capital stays ₹25,000 initially, simulated leverage 5×, entry cap
  ₹85,000, and daily-loss threshold ₹1,000. Those settings are read from the
  supplied simulation configuration; the script does not edit them.

The model predicts a future market move, not net P&L. Regression metrics therefore
compare gross-return errors to zero and training-mean forecasts. Trading results
separately include costs and stops and use no-trade P&L zero as a financial
benchmark. These measures must not be confused.

## Selection and limitations

A 5/15-minute candidate must beat both forecast baselines in pooled fold MSE,
win at least two of three folds, produce positive total net P&L, have at least
two positive scoring days and complete at least three round trips. This is a
prespecified development screen, not statistical proof. Among qualifying
candidates, highest development net P&L chooses a horizon before examining later
comparison dates. The 1-minute control cannot become the selected replacement.

If no candidate qualifies, the script reports that outcome and does not score the
later comparison. If one qualifies, it is fitted using all original fitting dates
and compared on later original validation dates. No later final evaluation or
reserved practice data is permitted in the input. A passing later comparison
supports further research only; it does not activate trading.

The data was already inspected in earlier research, so these are development
comparisons, not a wholly independent final test. Costs are indicative. Public
candles do not establish execution liquidity, bid/ask spread, depth, broker margin
eligibility or accepted stops. Actual entry gaps may invalidate the known-price
cost estimate. Minute-bar fills/stops are estimates and loss thresholds can be
exceeded during gaps/charges. No financing or additional broker service fees are
modelled. Do not infer profits from zero trades or remove the hurdle to make the
model trade.

A strategy is a separate requirement from software correctness. Existing tests
and a report-producing pipeline do not establish live readiness. Groww account
access/response validation and the unsupported leveraged live adapter are also
separate unfinished work; the current real adapter remains one-share and disabled.
