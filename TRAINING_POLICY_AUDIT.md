# Check the final policy on its original training days

Training action/execution totals include stochastic exploration throughout
learning. They do not show how the final saved deterministic policy behaves on
those same days. This command replays it without learning or touching later data.

```powershell
.\.venv\Scripts\python.exe audit_training_policy.py --csv data/rl_training_review.csv --model-dir artifacts/model-market50k-01 --settings paper_market_50k_settings.json --symbol RELIANCE --output-dir artifacts/frozen-training-audit-01
Get-Content .\artifacts\frozen-training-audit-01\training_policy_audit.json
```

Update source files and include `audit_training_policy.py`. Existing dependencies,
models, data and settings are reused. No Python/CUDA reinstall is needed. Use a
fresh output folder; no earlier report is overwritten. The input must contain
exactly the dates in the checkpoint's `session_splits.rl`, and virtual starting
capital must match its training profile. Evaluation and practice dates are refused.

Indicators restart each day as during training. Every day starts with the
configured INR 50,000 virtual cash, matching independent RL episode resets;
profits/losses are not carried between these diagnostic episodes. The summary
reports **summed independent daily P&L**, not the return of a continuously funded
portfolio. Costs, slippage, stops, sizing and daily loss controls remain active.
The model and scaler are frozen. There are no broker requests/orders or learning.

If final training-day behaviour is also inactivity, the policy has not found a
useful entry strategy even on familiar data. If it trades on training dates but
not later dates, investigate generalization and state distributions. Neither
outcome alone identifies a single cause or justifies changing rewards blindly.
In-sample profits would still require independent later testing and would not
establish readiness for real money. Keep existing artifacts as baselines.

This diagnostic does not add entry incentives, remove fees, change risk limits,
activate real trading or retrain the policy. Its purpose is to avoid another
parameter change before distinguishing exploration from the saved policy.
