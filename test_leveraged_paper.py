"""Leveraged fake money: independent accounting identities and broker isolation."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

from rl_brain import TradingEnvironment
from paper_trade import FrozenPolicy, PaperRun
from live_trade import verify_paper_report
from main import train_pipeline
from data_loader import load_intraday_csv
from test_training_smoke import _write_session_csv


class LeveragedAccountingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.path=self.root/'settings.json'
        self.config=json.loads(Path(__file__).with_name('paper_settings.json').read_text())
        self.net=patch('requests.sessions.Session.request',side_effect=AssertionError('No HTTP'))
        self.net.start()
        self.addCleanup(self.net.stop)

    def env(self, prices=None, **kwargs):
        self.path.write_text(json.dumps(self.config))
        prices=prices or [200]*6
        frame=pd.DataFrame({'open':prices,'high':prices,'low':prices,'close':prices},
            index=pd.date_range('2026-10-21 10:00',periods=len(prices),freq='min',tz='Asia/Kolkata'))
        env=TradingEnvironment(frame,symbol='RELIANCE',settings_path=self.path,
                               slippage_bps=0,**kwargs)
        env.reset()
        return env

    def test_shipped_training_and_paper_settings_match_requested_capital_and_leverage(self):
        for name in ('paper_settings.json','simulation_settings.json'):
            config=json.loads(Path(__file__).with_name(name).read_text())
            self.assertEqual(config['simulation'],{
                'initial_cash_inr':25000,'leverage':5,'position_sizing':'max_notional'})
            self.assertEqual(config['risk_limits']['max_order_value_inr'],85000)
            self.assertEqual(config['risk_limits']['max_position_value_inr'],85000)
            self.assertEqual(config['risk_limits']['max_daily_loss_inr'],1000)
        live=json.loads(Path(__file__).with_name('settings.json').read_text())
        self.assertEqual(live['execution']['mode'],'disabled')
        self.assertFalse(live['risk_limits']['trading_enabled'])
        self.assertEqual(live['risk_limits']['max_order_value_inr'],1000)

    def test_normalized_inputs_preserve_signal_and_actual_position_pnl(self):
        env=self.env([200,200,201,201,201,201])
        env.df['lstm_score']=.001
        np.testing.assert_allclose(env._get_obs(),[.1,1,0,0],rtol=1e-6)
        env.step(1,end_of_data=False)
        observation,_,_,_,_=env.step(0,end_of_data=False)
        self.assertAlmostEqual(float(observation[1]),env.cash/25000,places=6)
        self.assertEqual(observation[2],1)
        self.assertAlmostEqual(float(observation[3]),425/25000,places=6)
        env.df.loc[env.current_step,'lstm_score']=.002
        changed=env._get_obs()
        self.assertAlmostEqual(float(changed[0]),.2,places=6)
        np.testing.assert_array_equal(changed[1:],observation[1:])
        self.assertEqual(env.observation_schema,'account_fractions_return_percent_v1')

    def test_max_notional_whole_shares_and_margin_reservation(self):
        env=self.env()
        _,_,_,_,info=env.step(1,end_of_data=False)
        self.assertEqual(env.initial_cash,25000)
        self.assertEqual(env.quantity,425)
        self.assertEqual(info['position_value_inr'],85000)
        self.assertEqual(info['margin_reserved_inr'],17000)
        self.assertAlmostEqual(env.cash,8000-env.total_charges)
        self.assertAlmostEqual(info['equity'],25000-env.total_charges)
        self.assertEqual(env.executions[0]['quantity'],425)
        self.assertEqual(env.executions[0]['notional_inr'],85000)
        # Cannot add to the open position on a second buy.
        env.step(1,end_of_data=False)
        self.assertEqual(len(env.executions),1)

    def test_profit_and_charges_apply_to_full_quantity_without_double_leverage(self):
        env=self.env([200,200,201,201,201,201])
        env.step(1,end_of_data=False)
        _,_,_,_,info=env.step(2,end_of_data=False)
        expected_gross=425*(201-200)
        self.assertEqual(env.quantity,0)
        self.assertEqual(env.margin_reserved,0)
        self.assertAlmostEqual(env.cash,25000+expected_gross-env.total_charges)
        expected=env.charges.estimate_orders([(85000,'BUY'),(85425,'SELL')])
        self.assertAlmostEqual(env.total_charges,expected['total'])
        self.assertAlmostEqual(info['daily_pnl'],expected_gross-expected['total'])

    def test_cash_limited_quantity_reserves_fees_and_never_borrows_entry_margin(self):
        env=self.env(initial_cash=100)
        env.step(1,end_of_data=False)
        self.assertEqual(env.quantity,2)  # Three shares need at least 120 margin.
        self.assertGreaterEqual(env.cash,0)
        self.assertLessEqual(env.margin_reserved+env.total_charges,100)
        env.finish()
        self.assertAlmostEqual(env.cash,100-env.total_charges)

    def test_rounding_cannot_exceed_cap_even_with_slippage(self):
        self.path.write_text(json.dumps(self.config))
        frame=pd.DataFrame({'open':[200]*4,'high':[201]*4,'low':[200]*4,'close':[200]*4},
            index=pd.date_range('2026-10-21 10:00',periods=4,freq='min',tz='Asia/Kolkata'))
        env=TradingEnvironment(frame,symbol='RELIANCE',settings_path=self.path)
        env.reset()
        env.step(1,end_of_data=False)
        self.assertEqual(env.quantity,424)
        self.assertLessEqual(env.executions[0]['notional_inr'],85000)
        self.assertGreater(env.executions[0]['price']*(env.quantity+1),85000)

    def test_price_above_cap_rejected_and_no_fractional_shares(self):
        env=self.env([90000]*6)
        _,_,_,_,info=env.step(1,end_of_data=False)
        self.assertTrue(info['order_rejected'])
        self.assertEqual(env.quantity,0)
        self.assertEqual(env.cash,25000)

    def test_gap_loss_releases_margin_and_halts_without_retry_or_new_entry(self):
        env=self.env([200,200,190,190,190,190])
        env.step(1,end_of_data=False)
        _,_,done,_,info=env.step(1,end_of_data=False)
        self.assertTrue(done)
        self.assertEqual(env.quantity,0)
        self.assertEqual(len(env.executions),2)
        self.assertFalse(env.gatekeeper.trading_enabled)
        self.assertAlmostEqual(info['daily_pnl'],-4250-env.total_charges)
        self.assertEqual(env.margin_reserved,0)

    def test_daily_loss_trigger_scales_with_quantity_and_closes_early(self):
        self.config['risk_limits']['max_daily_loss_inr']=500
        env=self.env()
        env.step(1,end_of_data=False)
        env.df.loc[2,'low']=198
        _,_,done,_,info=env.step(0,end_of_data=False)
        self.assertTrue(done)
        self.assertEqual(env.executions[-1]['reason'],'daily_loss')
        self.assertGreater(env.executions[-1]['price'],198)  # Before 1% price stop.
        self.assertLessEqual(info['daily_pnl'],-500)
        self.assertEqual(env.quantity,0)

    def test_mandatory_one_percent_stop_uses_all_shares(self):
        self.config['risk_limits']['max_daily_loss_inr']=5000
        env=self.env()
        env.step(1,end_of_data=False)
        env.df.loc[2,'low']=198
        env.step(0,end_of_data=False)
        self.assertEqual(env.executions[-1]['reason'],'stop_loss')
        self.assertEqual(env.executions[-1]['quantity'],425)
        self.assertAlmostEqual(env.cash,25000-850-env.total_charges)

    def test_loss_changes_next_position_margin_budget(self):
        env=self.env(initial_cash=10000)
        self.config['risk_limits']['max_daily_loss_inr']=5000
        # Rewrite before reset to make this a high-loss-budget independent case.
        self.path.write_text(json.dumps(self.config))
        env=self.env([200,200,199,199,199,199],initial_cash=10000)
        env.step(1,end_of_data=False)
        env.step(2,end_of_data=False)
        env.step(1,end_of_data=False)
        self.assertLess(env.executions[-1]['notional_inr'],50000)
        self.assertGreaterEqual(env.cash,0)
        self.assertLessEqual(env.margin_reserved,env.initial_cash)

    def test_leverage_invalid_or_in_real_mode_is_refused(self):
        for leverage in (0,6,True,float('nan')):
            self.config['simulation']['leverage']=leverage
            with self.assertRaises(ValueError): self.env()
        self.config['simulation']['leverage']=5
        self.config['execution']['mode']='live'
        with self.assertRaisesRegex(ValueError,'fake-money'):
            self.env()

    def test_rewards_and_end_of_session_equity_reconcile(self):
        env=self.env([200,200,200.5,201,201,201])
        reward_sum=0
        for action in [1,0,2,0,0]:
            _,reward,done,_,info=env.step(action)
            reward_sum+=reward
            if done: break
        expected=info['daily_pnl']/env.initial_cash-env.turnover_penalty*2
        self.assertAlmostEqual(reward_sum,expected)
        self.assertEqual(env.quantity,0)
        self.assertAlmostEqual(info['cash'],info['equity'])


class LeveragedModelTests(unittest.TestCase):
    def test_model_metadata_paper_replay_and_live_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            csv=root/'training.csv'
            _write_session_csv(csv)
            config=json.loads(Path(__file__).with_name('simulation_settings.json').read_text())
            settings=root/'simulation.json'
            settings.write_text(json.dumps(config))
            model_dir=root/'model'
            with patch('requests.sessions.Session.request',side_effect=AssertionError('No HTTP')):
                metrics=train_pipeline(csv,settings,model_dir,symbol='RELIANCE',
                    lstm_epochs=1,rl_timesteps=64,seed=3)
                self.assertEqual(metrics['simulator_version'],4)
                self.assertEqual(metrics['execution_profile']['initial_cash_inr'],25000)
                self.assertEqual(metrics['execution_profile']['leverage'],5)
                self.assertEqual(metrics['execution_profile']['observation_schema'],
                                 'account_fractions_return_percent_v1')
                diagnostics=metrics['training_diagnostics']
                self.assertEqual(diagnostics['steps'],metrics['rl_actual_timesteps'])
                self.assertEqual(sum(diagnostics['action_counts'].values()),diagnostics['steps'])
                self.assertEqual(sum(metrics['evaluation']['action_counts'].values()),
                                 metrics['evaluation']['steps'])
                self.assertGreater(metrics['forecast_diagnostics']['evaluation_window']['samples'],0)
                self.assertEqual(metrics['forecast_diagnostics']['lstm_training_window']['samples'],
                                 metrics['supervised_samples'])
                model=FrozenPolicy(model_dir,'RELIANCE')
                model.action=Mock(side_effect=[1,2]+[0]*400)
                config['execution']['mode']='paper'
                config['paper']={'initial_cash_inr':25000}
                paper=root/'paper.json'
                paper.write_text(json.dumps(config))
                frame=load_intraday_csv(csv)
                frame.index=frame.index+pd.Timedelta(days=1)
                run=PaperRun(model,paper,root/'report','RELIANCE')
                report=run.replay(frame)
                self.assertEqual(report['initial_cash_inr'],25000)
                self.assertEqual(report['simulator_version'],4)
                self.assertGreater(report['max_executed_quantity'],1)
                self.assertEqual(sum(report['action_counts'].values()),report['decisions'])
                self.assertFalse(report['uses_real_money'])
                self.assertFalse(report['overlaps_training'])
                with self.assertRaisesRegex(ValueError,'one-share live adapter'):
                    verify_paper_report(root/'report'/'paper_report.json',model_dir,'RELIANCE')
                config['simulation']['leverage']=1
                paper.write_text(json.dumps(config))
                mismatch=PaperRun(model,paper,root/'mismatch','RELIANCE')
                with self.assertRaisesRegex(ValueError,'retrain'):
                    mismatch.replay(frame)
                self.assertEqual(json.loads((root/'mismatch'/'paper_report.json').read_text())['status'],'failed')


if __name__=='__main__': unittest.main()
