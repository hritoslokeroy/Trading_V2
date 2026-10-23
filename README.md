# NSE intraday trading research

Offline LSTM + PPO/DQN training and fake-money trading for one NSE cash
instrument, plus a separately disabled, one-share Groww live adapter. Amounts are INR. Exchange
clocks use **Asia/Kolkata (IST, UTC+05:30)**, including on UTC cloud machines.
The simulator accepts contiguous weekday one-minute candles from 09:15 to
15:29 and prevents entries at or after the configured **15:15 IST** cutoff.
Paper/training defaults: ₹25,000 virtual capital, 5× simulated leverage, whole
shares sized up to ₹85,000 entry notional. Real trading defaults stay disabled
and do not inherit this leverage or sizing. It does not implement F&O, shorts,
delivery, real broker leverage, a holiday calendar,
or trailing stops.

## Install and test

Use Python 3.12 and run commands from this repository's root. Modules intentionally
live in this directory; no temporary `src` layout or PYTHONPATH is required.
The default dependency file installs CPU PyTorch. GPU-specific installation is
outside this tested configuration.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -v
```

On the prepared cloud machine the interpreter is
`/workspace/trading-test-env/bin/python`. With uv, install using
`uv pip install --index-strategy unsafe-best-match -r requirements.txt` so it can
resolve the explicitly pinned CPU torch wheel on the additional PyTorch index.
Keep TLS verification enabled.

Tests use synthetic candles and mocked broker/feed/credentials. They exercise
PPO and DQN training, checkpoint reloading, financial accounting, tariff
arithmetic, stop and loss limits, gap fills, Kolkata cutoff times, order counts,
and simulated live fault recovery. They place no live orders.

## Paper trading before real money

See [PAPER_TRADING.md](PAPER_TRADING.md) for the Windows workflow: train, replay
with virtual money, practice on read-only Groww candles, and review before a
separate manual real-money launch. Paper mode never switches to real trading.

## Multiple-session forecast validation

See [MULTI_SESSION_TRAINING.md](MULTI_SESSION_TRAINING.md) for the recommended
next workflow: recent public or local CSV data, day-isolated basis-point forecast
training, validation against zero/mean baselines, separate policy-training days,
and later evaluation. A failed forecast gate stops before PPO training.
`main.py` below remains the original single-session diagnostic entry point.

## Offline training

Supply a CSV with `timestamp,open,high,low,close,volume`. Each training run currently
requires one contiguous minute session with at least 120 indicator-ready candles
(after RSI warmup). A multi-day downloader export must be split into sessions
before training. No historical dataset is included.

```bash
python main.py --csv data/session.csv --settings simulation_settings.json \
  --output-dir artifacts/experiment-01 --symbol RELIANCE \
  --algorithm PPO --lstm-epochs 10 --rl-timesteps 2048
```

Choose a new output directory for every run. The pipeline partitions data
chronologically 60% LSTM / 20% RL / 20% evaluation and fits scaling on the LSTM
partition only. It exports `lstm.pt`, `feature_scaler.npz`, `rl_policy.zip`, and
`run_metrics.json`. Metrics include **actual fills**, forced exits, net P&L,
charge components, rate assumptions and an execution ledger. Reward is net equity
change / initial capital, less the documented turnover and risk penalties.

Equity is free cash plus reserved initial margin plus open P&L. Fake-money buys
reserve notional/leverage plus entry fees; exits release margin and realize
price movement times actual quantity. Fees use full traded notional. Buy checks
enforce cash, order and position-value limits. Stops handle opening gaps at the
open and intrabar breaches at the protective trigger, with adverse slippage.
The daily-loss trigger uses net mark-to-market including costs already charged;
after liquidation all returned balances and counters are refreshed. The final
bar closes remaining positions at its close. Minute bars cannot establish exact
tick fills or guarantee that a loss will stay within a limit during a gap.

**Retrain for this profile.** Leveraged whole-share sizing uses simulator version
4; paper runs reject mismatched sizing and observation profiles. Both the old
one-share version-2 models and unnormalized leveraged version-3 models require
retraining for the new defaults. The live adapter stays
one-share/version-2 PPO only and refuses these leveraged models and reports.
DQN remains available for offline training and paper trading.
No test or short training run establishes strategy profitability.

### Policy inputs and diagnostics

Simulator version 4 keeps the four-input policy interface but changes units to
`[predicted_return * 100, free_cash / initial_capital, position_count,
open_pnl / initial_capital]`. These fixed scales are applied identically during
training and paper inference; no practice/evaluation data fit the normalization.
The separate market-feature scaler still uses only the LSTM training partition.
Risk controls and accounting continue using actual INR amounts and quantities.

`run_metrics.json` records training action counts, fills, rejected entries,
episode counts and reward penalties, plus deterministic evaluation action counts.
Forecast diagnostics compare next-close return MSE against a zero-return forecast
for each partition, excluding targets outside that partition. Training MSE is
not validation accuracy. Repeated training episodes are not additional market
history; their summed changes must not be read as a portfolio return.

No reward is added for making a trade. Costs, overtrading penalties and risk
limits stay active. An all-Hold evaluation is explicitly reported; more actions
or lower training error alone do not establish improvement. The original `main.py` diagnostic still accepts one continuous session.
The new `train_multisession.py` path supports separate chronological session
groups; use untouched later data for final strategy assessment.

## Charges for NSE cash intraday

`charges.py` estimates NSE cash intraday costs using a session charge ledger. Both settings files expose
the rate assumptions in `charges.rates`; rates in JSON are **fractions of turnover**.
These are indicative defaults, not independently verified current tariffs.
Access to Groww's pricing pages was denied by the cloud network policy during
implementation. Verify the rates against [Groww pricing](https://groww.in/pricing),
the [brokerage calculator](https://groww.in/calculators/brokerage-calculator), and
your contract notes before using them for real-money decisions.

| Component | Indicative default |
| --- | --- |
| Brokerage, each executed order | Lower of ₹20, max(₹5, 0.1% of value), and 2.5% of value |
| Intraday STT | 0.025% of sells |
| Stamp duty | 0.003% of buys |
| NSE exchange charge | 0.00297% on both sides |
| SEBI charge | 0.0001% on both sides |
| IPFT assumption | 0.0001% on both sides; confirm current treatment |
| GST | 18% on brokerage + exchange + SEBI + IPFT |

Brokerage is calculated and rounded to paise **per executed order**, with its
own minimum/maximum and small-order cap. Other heads retain Decimal precision
across the session; each component and GST is rounded to paise on the cumulative
amount. GST uses the unrounded taxable components and per-order brokerage.
Executions debit only the increase in that session bill, so their charges sum
to the reported total. Quotes do not change the ledger, rejected orders incur
no charges, and reset starts a new session ledger. Keep segments/sessions separate.

The default is `charges.aggregation="session"` and
`charges.rounding="paise_half_up_per_component"`. STT and stamp duty are **not**
forced to whole rupees. This is an estimate policy consistent with paise amounts
in the supplied statement, not a claim that its exact underlying aggregation
rules can be inferred from the summary. Actual broker charges take precedence.
The statement's equity brokerage below ₹5 does not on its own establish a new
minimum: order details are needed, including small-order caps.

Example calculations under the configured rates (one buy order and one sell order):

| Quantity and prices | Charges | Net P&L |
| --- | --- | --- |
| 100 shares, ₹200 → ₹215 | ₹54.73 | ₹1,445.27 |
| 100 shares, ₹200 → ₹185 | ₹52.10 | −₹1,552.10 |
| 1 share, ₹1,000 → ₹1,000 | ₹12.15 | −₹12.15 |

These examples exclude slippage. Calculate offline, without broker credentials:

```bash
python charges.py --quantity 100 --buy 200 --sell 215
python charges.py --quantity 100 --buy 200 --sell 185
```

The calculator accepts quantities independently of the trading model;
it does not increase live sizing or risk limits. Do not present each fill of one
order as a new executed order: partial-fill aggregation is outside this simulator.

For actual statement reconciliation, `reconcile_statement(obligation, charges,
net_amount)` uses signed amounts exactly as reported and returns a difference.
It does not substitute an estimate, invent turnover from a settlement obligation,
or interpret that obligation as trading profit/loss. The supplied equity summary
reconciles **−₹1,220.98 − ₹9.19 = −₹1,230.17**. F&O separately reconciles
**−₹10,700 − ₹117.43 = −₹10,817.43**; combined charges are **₹126.62** and net
payable is **₹12,047.60**. Actual-statement arithmetic supports these columns;
it does not add F&O tariff estimation or F&O trading to this equity program.
IGST replaces the CGST/SGST split on the supplied statement; do not charge both.
IPFT remains an explicit component and is included in the GST base, whether the
broker displays it separately or folds it into another line. Avoid double counting.

Charges are deducted from simulated cash, included in net P&L and used in loss
checks. A legacy `fee_bps` override uses its original flat-fee calculation;
normal training uses the session ledger. Retraining is recommended after changing
rates or the charge model. The exact broker aggregation/rounding policy and current
rates still require dated trade/order details and the applicable tariff.

These NSE rates do not vary merely because the user lives in Kolkata. GST may
be split into CGST/SGST or IGST on the actual invoice; the estimator uses the
combined rate. The model excludes income tax, delivery DP charges, API/data
subscriptions, exchange penalties and broker auto-square-off charges. Its 15:15
exit is strategy initiated, ahead of broker auto-square-off. Add any applicable
subscriptions to your business cost assessment; they are not order turnover fees.

## Configuration and live execution

`simulation_settings.json` enables **offline simulation** only. `settings.json`
disables trading and live execution by default. The live adapter refuses to start
unless all three are explicit: `execution.mode="live"`,
`risk_limits.trading_enabled=true`, and `charges.rates_verified=true`. The last
flag records the operator's rate review; it does not verify prices automatically.
Never pass simulation settings to a live session.

Credential mappings refer to `GROWW_API_KEY`, `GROWW_TOTP_SECRET`,
`GROWW_API_SECRET`, and `GROWW_ACCESS_TOKEN`. Authentication accepts an existing access token, an API key with a TOTP seed,
or an API key with an API secret, in that priority order. Supply credentials securely through your environment;
keep them out of source control and logs. No real credentials are needed for tests.
The default settings file resolves beside the module, regardless of the shell's
working directory. Explicit paths are recommended for experiments.

The live decoder uses documented `clear_cash`, available MIS balance,
`brokerage_and_charges`, position `quantity`, `net_price`, and `realised_pnl`.
Buying cash is capped by the smaller of clear cash and available MIS balance.
Open P&L is marked with a fresh LTP; reported margin charges are deducted once
from realized plus open P&L. Estimated fees separately reserve entry cash.
An isolated NSE CASH MIS account is required: CNC/carry-forward positions,
unmanaged open symbols, missing risk inputs and unknown fee signs are refused.
Validate that margin charges cover today's session and that closed position
rows retain today's realized P&L against the actual account before enabling live.
The documentation examples and offline tests cannot establish those semantics.

Production LTP updates carry prices/timestamps only. Model features use completed
one-minute historical OHLCV, matching training RSI/VWAP. Missing, stale, revised,
or gapped candles halt the session; current-day historical availability/latency
still needs a read-only account check. NSE CASH permissions and instrument
buy/sell/reserved flags, token, lot and unmodified tick size are checked at startup.
Pending cancellation/modification/delivery states remain working until confirmed.

Shared per-session API budgets cover documented authentication, order writes,
quotes and account/status reads. Exceeding a budget halts without retrying an
uncertain write or delaying reconciliation behind a minute-long queue. Budgets
are in memory, not shared with other programs or persisted across restarts;
broker-side limits remain authoritative. Historical quotas were not specified
in the reviewed table and are not inferred from quote limits. Actual broker stop
acceptance, response semantics and account access remain independently unverified.

The live adapter has no automatic restart: inspect previous journals and
reconcile broker positions and orders after any uncertain write. Live market
access has not been exercised by these tests.
