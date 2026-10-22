# NSE intraday trading research

Offline LSTM + PPO/DQN training and an opt-in Groww execution adapter for one
fully funded long share of one NSE cash instrument. Amounts are INR. Exchange
clocks use **Asia/Kolkata (IST, UTC+05:30)**, including on UTC cloud machines.
The simulator accepts contiguous weekday one-minute candles from 09:15 to
15:29 and prevents entries at or after the configured **15:15 IST** cutoff.
It does not implement F&O, shorts, delivery, margin leverage, a holiday calendar,
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

Equity is cash plus the owned share's market value. Buy checks reserve fees and
enforce cash, order and position-value limits. Stops handle opening gaps at the
open and intrabar breaches at the protective trigger, with adverse slippage.
The daily-loss trigger uses net mark-to-market including costs already charged;
after liquidation all returned balances and counters are refreshed. The final
bar closes remaining positions at its close. Minute bars cannot establish exact
tick fills or guarantee that a loss will stay within a limit during a gap.

**Retrain old checkpoints.** Simulator version 2 fixes the earlier accounting
and risk behavior; the live loader rejects checkpoints without that version.
The live adapter accepts PPO only; DQN remains an offline training option.
No test or short training run establishes strategy profitability.

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

The calculator accepts quantities independently of the one-share trading model;
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
`GROWW_API_SECRET`, and `GROWW_ACCESS_TOKEN`. The current TOTP authentication path
requires the first two. Supply credentials securely through your environment;
keep them out of source control and logs. No real credentials are needed for tests.
The default settings file resolves beside the module, regardless of the shell's
working directory. Explicit paths are recommended for experiments.

Live account P&L comes from broker snapshots; estimated fees are used to reserve
entry cash, not subtracted a second time from broker-reported P&L. Actual Groww
response schemas, broker stop acceptance, feed behavior, and account permissions
must still be checked independently. The decoder in `groww_env.py` is an explicit
reference contract and fails closed on missing or unfamiliar data. Mocked tests
do not validate a real account's schema.

The live adapter has no automatic restart: inspect previous journals and
reconcile broker positions and orders after any uncertain write. Live market
access has not been exercised by these tests.
