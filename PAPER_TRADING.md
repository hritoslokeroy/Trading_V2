# Practice with virtual money first

Paper trading uses **₹1,00,000 virtual cash**, **one share per position**, an INR
5,000 paper order/position cap, the configured 1% stop, estimated Groww charges,
and a ₹500 daily simulated loss limit. These are in `paper_settings.json`.
Your real account balance and holdings are never used by paper mode.

The workflow is: download market data → train on a past session → practice on a
later session → review results → choose separately whether to start real trading.
Paper mode never enables real trading or modifies a trained model. Learning happens
in `main.py`; `paper_trade.py` exercises a frozen checkpoint. Improvement requires
retraining and a new independent practice period, not simply leaving paper mode on.

## 1. Prepare your Windows PowerShell

Download the latest repository ZIP or pull `main`. Open PowerShell in the project
folder. Reuse your installed `.venv`; the commands below invoke its Python directly
so PowerShell activation permissions do not get in the way. If Python/dependencies
are not installed, follow README first. The existing CUDA PyTorch installation can
be reused; reinstalling the CPU-pinned requirements replaces it with CPU PyTorch.

**Never paste API secrets into chat, code, reports, or GitHub.** Set credentials
only in your local PowerShell environment. For a Groww API-key/secret account:

```powershell
$env:GROWW_API_KEY = [System.Net.NetworkCredential]::new('', (Read-Host 'Groww API key' -AsSecureString)).Password
$env:GROWW_API_SECRET = [System.Net.NetworkCredential]::new('', (Read-Host 'Groww API secret' -AsSecureString)).Password
```

If your account uses TOTP authentication, set `GROWW_TOTP_SECRET` to its Base32 seed
instead of API secret. The seed is different from a temporary six-digit code.
Alternatively set `GROWW_ACCESS_TOKEN` securely to an already-issued valid access
token. Selection order is access token, API key + TOTP seed, API key + secret;
remove stale variables for an unused route. Credentials expire according to Groww.
The app does not load `.env` files automatically. Closing PowerShell clears these
process-scoped variables. No API credentials are needed for CSV replay or training.

## 2. Download two independent historical sessions

Pick two completed NSE trading dates: one for training, a **later date** for practice.
Check the exchange holiday calendar. The downloader uses market-data requests only.
These PowerShell prompts ask for dates, not secrets:

```powershell
$trainingDay = Read-Host 'Training date (YYYY-MM-DD)'
$practiceDay = Read-Host 'Later practice date (YYYY-MM-DD)'
.\.venv\Scripts\python.exe download_history.py --symbol RELIANCE --start $trainingDay --end $trainingDay --out data/train.csv --settings paper_settings.json
.\.venv\Scripts\python.exe download_history.py --symbol RELIANCE --start $practiceDay --end $practiceDay --out data/practice.csv --settings paper_settings.json
```

A CSV needs `timestamp,open,high,low,close,volume`. Each run needs one contiguous
minute session. The current trainer requires at least 120 indicator-ready candles.
Broker data availability/permissions must be tested on your PC; no real account
connection was tested in the cloud. If you already have matching CSVs, skip download.

## 3. Train and replay with fake money

Training still uses `simulation_settings.json`. Review its symbol/order caps for
your dataset before training: orders above a cap are rejected rather than enlarged.
Default simulation and paper order/position caps are both ₹5,000. Keep those
fake-money limits consistent when comparing training with practice. This does
not change the disabled real-money settings or their separate ₹1,000 caps.

```powershell
.\.venv\Scripts\python.exe main.py --csv data/train.csv --settings simulation_settings.json --output-dir artifacts/model-01 --symbol RELIANCE --algorithm PPO --lstm-epochs 10 --rl-timesteps 2048
.\.venv\Scripts\python.exe paper_trade.py --csv data/practice.csv --model-dir artifacts/model-01 --symbol RELIANCE --output-dir artifacts/paper-replay-01
```

Use PPO if you intend to use the current live adapter later. DQN works for offline
training and paper trading, but the real-money adapter currently accepts PPO only.
Use a new output folder for each run; existing results are never overwritten.
To choose different **virtual** starting funds, add `--cash 25000`.

Each paper folder contains:

- `paper_report.json`: virtual cash/equity, net P&L after fees, estimated charges,
  drawdown, decision/fill counts, training overlap and model fingerprint.
- `executions.json`: simulated buys/sells, forced exits and cost components.
- `decisions.json`: model proposals and rejected entries.
- `equity_curve.csv`: virtual equity and P&L over time.

`overlaps_training: true` marks an in-sample replay, which is not an independent
performance test. Zero trades do not demonstrate entry/exit performance. Costs and
slippage can make a flat-price strategy lose money. Each run begins a new virtual
session; carry forward virtual funds explicitly with `--cash` if desired.

## 4. Practice on current Groww market data

During an NSE weekday session, with credentials set locally:

```powershell
.\.venv\Scripts\python.exe paper_trade.py --live-data --model-dir artifacts/model-01 --symbol RELIANCE --output-dir artifacts/paper-market-01
```

This reads Groww's **completed one-minute candles**. It never submits, cancels,
or modifies broker orders and never reads real positions or margin. Startup
history is warm-up only: it starts at the latest completed candle and follows new
candles. The LSTM requires about 74 completed bars to warm up if starting at open.

These are delayed, simulated next-open fills, observed only after the next minute
completes. They do not establish tick-level fill prices or realistic liquidity.
The runner stops on stale/missing/rewritten data, day rollover or the cutoff. On
Ctrl+C or data failure it closes **fake exposure** at the last known close with
slippage and saves the report. That price can be stale; its exit reason is recorded.
No real position is being liquidated. It does not backfill missed minutes as trades.

Poll interval defaults to 10 seconds; data errors stop the run rather than retrying
indefinitely. Check data subscription costs and request limits with Groww. No
real-money transition occurs when paper profit reaches a target.

## 5. Real money is a separate, explicit decision

Review multiple independent paper sessions, costs, rejected entries, drawdowns and
failures first. Paper performance and passing tests do not establish profitability.
The real account's response schemas, feed, stop-order acceptance, credentials,
instrument token/tick size and account permissions still need independent validation.
The current reference decoder deliberately stops on an unfamiliar response.

`live_trade.py` provides a manual entry point. It requires all of:

- A separate local settings copy with live mode, trading enabled, and reviewed
  charge rates; neither shipped settings nor paper settings enable real trading.
- A completed, out-of-sample PPO paper report with simulated fills, for the exact
  same checkpoint and symbol. A no-trade, failed, unknown-overlap, DQN or changed-model
  report is refused. Local reports are review aids, not tamper-proof attestations.
- `--confirm-real-money` on the command, plus the correct instrument token, tick
  size and a fresh journal. Running paper mode cannot supply this flag automatically.

When you deliberately choose real trading, copy settings to `live_settings.local.json`
(which is ignored by Git), review its limits, and then update the three live gates.
See `python live_trade.py --help` for the complete arguments. There is no timed or
profit-based automatic activation. The adapter supports one long share and one
writer to the account; reconcile existing orders/positions before starting.
An interrupted/uncertain live write needs broker reconciliation, not an automatic
restart. Starting a real session is intentionally not part of the paper commands.
