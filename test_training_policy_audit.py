"""Frozen in-sample audit: strict date scope, isolated balances and no learning."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd
from audit_training_policy import audit
from data_loader import prepare_intraday_frame
from market_context import prepare_market_context
from quant_agent import QuantLSTMBrain
from rl_brain import TradingEnvironment
from test_multisession import market_days

ROOT=Path(__file__).resolve().parent


class TrainingPolicyAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.raw=market_days(4)
        self.csv=self.root/'training.csv'; self.raw.to_csv(self.csv,index=False)
        self.settings=ROOT/'paper_market_50k_settings.json'
        sample=prepare_market_context(prepare_intraday_frame(market_days(1))).dropna().assign(lstm_score=0)
        profile=TradingEnvironment(sample,symbol='RELIANCE',settings_path=self.settings).execution_profile()
        self.model=Mock()
        self.model.metadata={'algorithm':'PPO','simulator_version':5,'execution_profile':profile,
            'session_splits':{'rl':sorted(self.raw.timestamp.str[:10].unique().tolist())}}
        self.model.fingerprint='offline-frozen-model'
        self.model.action.return_value=2
        self.model.score.side_effect=lambda frame:prepare_market_context(frame).dropna(
            subset=list(QuantLSTMBrain.FEATURES)).iloc[59:].assign(lstm_score=0)
        self.model.learn.side_effect=AssertionError('No learning allowed')
        self.factory=patch('audit_training_policy.FrozenPolicy',return_value=self.model)
        self.factory.start(); self.addCleanup(self.factory.stop)
        self.http=patch('requests.sessions.Session.request',side_effect=AssertionError('No network'))
        self.http.start(); self.addCleanup(self.http.stop)

    def test_replays_all_training_dates_with_no_learning_and_fresh_cash(self):
        output=self.root/'audit'
        result=audit(self.csv,self.root/'model',self.settings,output)
        self.assertFalse(result['is_out_of_sample'])
        self.assertFalse(result['uses_real_money'])
        self.assertFalse(result['learning_during_audit'])
        self.assertEqual(result['executed_orders'],0)
        self.assertGreater(result['sell_while_flat'],0)
        self.assertEqual(result['initial_cash_per_day_inr'],50000)
        self.assertEqual(result['summed_independent_daily_net_pnl_inr'],0)
        for day in result['dates']:
            report=json.loads((output/day/'paper_report.json').read_text())
            self.assertEqual(report['initial_cash_inr'],50000)
        self.model.learn.assert_not_called()
        self.assertTrue((output/'training_policy_audit.json').exists())

    def test_wrong_dates_are_rejected_before_writing_reports(self):
        self.raw.iloc[:180].to_csv(self.csv,index=False)
        with self.assertRaisesRegex(ValueError,'exactly'):
            audit(self.csv,self.root/'model',self.settings,self.root/'bad')
        self.assertFalse((self.root/'bad').exists())
        later=self.raw.copy(); later.loc[0,'timestamp']='2026-12-01 09:15:00'
        later.sort_values('timestamp').to_csv(self.csv,index=False)
        with self.assertRaisesRegex(ValueError,'exactly'):
            audit(self.csv,self.root/'model',self.settings,self.root/'future')

    def test_changed_virtual_capital_cannot_be_mislabelled_as_training_replay(self):
        self.model.metadata['execution_profile']['initial_cash_inr']=10000
        with self.assertRaisesRegex(ValueError,'starting capital'):
            audit(self.csv,self.root/'model',self.settings,self.root/'wrong-capital')
        self.assertFalse((self.root/'wrong-capital').exists())

    def test_previous_outputs_and_duplicate_bars_are_preserved_or_rejected(self):
        output=self.root/'existing';output.mkdir()
        marker=output/'saved.txt';marker.write_text('keep')
        with self.assertRaises(FileExistsError):audit(self.csv,self.root/'model',self.settings,output)
        self.assertEqual(marker.read_text(),'keep')
        pd.concat([self.raw,self.raw.iloc[:1]]).sort_values('timestamp').to_csv(self.csv,index=False)
        with self.assertRaisesRegex(ValueError,'unique'):
            audit(self.csv,self.root/'model',self.settings,self.root/'duplicate')


if __name__=='__main__': unittest.main()
