# Accumulate history before the next training experiment

The current model has not demonstrated a profitable entry/exit policy. Repeating
training or paper replay on the same days does not add evidence. Keep those
models and reports as baselines. Do not enable real trading or force purchases.

## Start a local archive

Update project files from main, preserving `.venv`, `data` and `artifacts`.
The optional downloader is the same `yfinance` used previously. No Groww API,
credentials or new core dependencies are needed.

In PowerShell inside the project folder:

```powershell
.\.venv\Scripts\python.exe archive_market_data.py --symbol RELIANCE --archive data/archive/RELIANCE_1m.csv
```

This fetches the recent seven-day one-minute window from Yahoo. It appends new
**completed weekday dates**, excluding the current day even after market close.
Run the same command each trading-day morning to capture prior completed days
before they roll out of the source's short history. It does not run automatically
or need to remain open all day. Public data availability is not guaranteed.

Repeated runs do not duplicate candles. Existing stored dates are immutable:
later revisions or additional minutes for an already stored date are not merged
silently. The command reports overlapping revisions and preserves the original
snapshot. Use a separate archive path if researching corrected source history.
No missing prices/volume are filled. CSV updates use an atomic file replacement;
an unsuccessful download or write does not intentionally erase previous history.
Backups remain useful for any local data collection.

The archive identifies the stock, one-minute interval and source in its columns.
Do not mix stocks or import five-minute candles as one-minute data. Plain CSV
imports cannot prove ticker identity; verify the file matches `--symbol`.

## Inspect progress without downloading

```powershell
.\.venv\Scripts\python.exe archive_market_data.py --symbol RELIANCE --archive data/archive/RELIANCE_1m.csv --status
```

The count that matters is **usable continuous sessions**, not repeated downloads.
A usable date contains at least 150 real consecutive regular-session minute bars.
The next development checkpoint is **20 usable sessions**. This is a practical
starting dataset target, not statistical proof of an edge or a live-readiness gate.
More independent market history and evaluation periods will still be needed.
With only the six currently available completed dates, accumulating genuinely new
dates takes several weeks; rerunning today cannot create additional history.

## Import genuine longer history if available

A longer one-stock OHLCV CSV from an appropriate data source can be imported:

```powershell
.\.venv\Scripts\python.exe archive_market_data.py --symbol RELIANCE --archive data/archive/RELIANCE_1m.csv --csv data/longer_history.csv
```

It must contain `timestamp,open,high,low,close,volume`, in NSE regular-session
one-minute bars. Naive timestamps are interpreted as IST; aware ones are converted.
Previously saved dates are preserved, even for imports. This command does not
purchase data or establish a provider's entitlement/accuracy. Broker history can
also be used once the Groww HTTP 403 issue is resolved.

If Yahoo is unavailable, existing experiment CSVs can seed a new archive:

```powershell
.\.venv\Scripts\python.exe archive_market_data.py --symbol RELIANCE --archive data/archive/RELIANCE_saved_1m.csv --csv data/multi-01/multi_sessions.csv
.\.venv\Scripts\python.exe archive_market_data.py --symbol RELIANCE --archive data/archive/RELIANCE_saved_1m.csv --csv data/multi-01/practice_multi.csv
```

These preserve already seen data; they do not create new training examples or
untouched holdouts.

## Prepare the next development dataset after collection

When sufficient new sessions are available:

```powershell
.\.venv\Scripts\python.exe prepare_training_data.py --symbol RELIANCE --csv data/archive/RELIANCE_1m.csv --output-dir data/multi-02
```

This snapshots the archive into separate model-development and later practice
files. It selects a real contiguous section per day and does not alter the
archive. Use the multi-session workflow in [MULTI_SESSION_TRAINING.md](MULTI_SESSION_TRAINING.md)
with new output folders, after reviewing the dates. Reserve later sessions not
used for tuning for final assessment.

## Clearer action diagnostics

Updated training/evaluation/paper reports include `sell_while_flat`: Sell
proposals made when the simulated account had no shares at decision time. Such
proposals are no-ops, not sell executions. `executed_orders`, actual fills and
charges remain the trading measures. This diagnostic does not change rewards,
force trades, relax risk limits or require retraining saved models. Existing
reports remain unchanged; new runs include the additional count.

Virtual capital remains ₹25,000 with 5× simulated leverage, an ₹85,000 entry cap,
a 1% price stop and ₹1,000 daily-loss threshold. Real trading stays disabled.
