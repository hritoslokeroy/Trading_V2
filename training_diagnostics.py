"""Offline learning diagnostics; never reward activity or change risk limits."""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TrainingDiagnostics(BaseCallback):
    def __init__(self):
        super().__init__()
        self.action_counts = dict.fromkeys(('hold','buy','sell'),0)
        self.steps = self.episodes = self.rejected_entries = 0
        self.buys = self.sells = self.forced_exits = 0
        self.equity_change = self.penalties = 0.0

    def _on_step(self):
        for action in np.asarray(self.locals['actions']).reshape(-1):
            self.action_counts[('hold','buy','sell')[int(action)]] += 1
            self.steps += 1
        for info in self.locals['infos']:
            self.rejected_entries += int(info.get('order_rejected',False))
            self.buys += info.get('buy_count',0)
            self.sells += info.get('sell_count',0)
            self.forced_exits += info.get('forced_exit_count',0)
            self.equity_change += info.get('equity_change_inr',0)
            self.penalties += info.get('reward_penalty',0)
        self.episodes += int(np.asarray(self.locals['dones']).sum())
        return True

    def summary(self):
        return {'steps':self.steps, 'completed_episodes':self.episodes,
                'action_counts':self.action_counts.copy(),
                'buy_executions':self.buys, 'sell_executions':self.sells,
                'forced_exits':self.forced_exits, 'rejected_entries':self.rejected_entries,
                'summed_equity_changes_inr':self.equity_change,
                'summed_reward_penalties':self.penalties,
                'note':'Training revisits the same session. Summed changes are not a portfolio return.'}


def forecast_metrics(frame):
    """Next-close return errors, excluding targets outside this supplied window.

    Zero-return forecasts are a simple baseline. Evaluation metrics are only
    reported after fitting; they never adjust models, scalers or rewards.
    """
    targets = frame.close.shift(-1) / frame.close - 1
    mask = targets.notna() & frame.lstm_score.notna()
    targets = targets[mask].to_numpy(dtype=float)
    scores = frame.loc[mask,'lstm_score'].to_numpy(dtype=float)
    if not len(targets):
        return {'samples':0, 'model_mse':None, 'zero_return_mse':None}
    return {'samples':len(targets), 'model_mse':float(np.mean((scores-targets)**2)),
            'zero_return_mse':float(np.mean(targets**2))}
