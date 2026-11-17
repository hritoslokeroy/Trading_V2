# Compare forecast inputs before another policy training run

The CSV audit found substantial price/volume changes between fitting and validation
periods. That makes a price-relative input experiment reasonable; it does not prove
that a new representation will forecast well or trade profitably.

This tool compares the existing `raw_v1` representation with `relative_v1`:

- One-minute close return in percentage points.
- Candle body and range relative to prices.
- Close distance from the causal session VWAP.
- Log of one plus volume relative to its trailing 20-bar median.
- Centered RSI and elapsed regular-session time.

Inputs remain seven-dimensional. No candle values, actual trade prices, charges,
capital limits or rewards are changed. All feature computations use only current
and earlier data in that session. A scaler is fitted only on the fold's fitting
days. Indicators and relative features are calculated before warmup rows are
removed, identically during multi-session fitting and paper inference.

## Run the development experiment

Update source files from main, preserving `.venv`, `data` and `artifacts`. Include
`forecast_features.py` and `forecast_experiment.py`. No new dependencies are needed.
Use the previously exported **fitting/validation-only** CSV, not the full archive:

```powershell
.\.venv\Scripts\python.exe forecast_experiment.py --csv data/forecast_review.csv --settings simulation_settings.json --fit-end 2026-10-29 --validation-end 2026-11-03 --output-dir artifacts/forecast-experiment-01 --lstm-epochs 15
Get-Content .\artifacts\forecast-experiment-01\comparison_report.json
```

The experiment uses the last three original fitting days as expanding-date fold
scoring days. In each fold, the preceding day is reserved for early stopping and
checkpoint selection; only days before that fit the model and scaler. The scoring
day does not select its own checkpoint. Raw and relative candidates use paired
seeds and the same architecture, loss, optimizer and maximum epochs.

Candidate choice uses pooled scoring error across those folds. After that choice
is frozen, the chosen representation is refitted using the last original fitting
day for checkpoint selection, and compared on the three original later validation
days. Those later comparison results cannot switch the selected candidate.
Neither policy-training dates nor final evaluation/practice data are used. The
tool rejects a CSV containing dates after `--validation-end`.

`further_research_supported` requires the selected candidate to beat both zero
and fitting-mean baselines in pooled fold MSE, win at least two of three folds,
and also beat both baselines on the later comparison. This is a development
screen, not statistical proof of a tradable advantage. These dates have already
been examined in earlier experiments; they are not an untouched final test.

The command writes only `comparison_report.json`, not a PPO model. It never
submits broker orders or trains a trading policy. If the screen fails, preserve
the report and do not bypass validation to force another policy model.

## Representation support in multi-session and paper code

The multi-session trainer now accepts an explicit `--feature-set raw_v1` or
`--feature-set relative_v1`. Its default remains raw for reproducibility. Do not
switch inputs merely because the option exists: review the comparison first.

Saved metadata includes `feature_transform` and `forecast_features`. The paper
runner reads that schema and applies the same representation and saved scaler;
it refuses unknown or inconsistent metadata rather than guessing input units.
Old artifacts without the new input marker retain their original raw behaviour.
The basis-point output transform remains separate and unchanged. Leveraged models
still cannot activate the one-share live adapter.

If a later separately specified experiment is justified, use new model/data
folders and the existing multi-session validation gate. Do not overwrite earlier
models, retune repeatedly on a claimed untouched test, remove costs or force Buy
actions. Real-money mode stays disabled.
