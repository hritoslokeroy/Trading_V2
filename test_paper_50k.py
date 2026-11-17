"""50k fake-money preset: full 5x sizing, fee reservation and live isolation."""
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd
from rl_brain import TradingEnvironment
from paper_trade import PaperRun
from groww_env import IST, ExecutionHalted, LiveGrowwExecutionAdapter


ROOT=Path(__file__).resolve().parent


class Paper50kTests(unittest.TestCase):
    def frame(self,prices=None):
        prices=prices or [200]*6
        return pd.DataFrame({'open':prices,'high':prices,'low':prices,'close':prices},
            index=pd.date_range('2026-11-16 10:00',periods=len(prices),freq='min',tz='Asia/Kolkata'))

    def env(self,prices=None):
        env=TradingEnvironment(self.frame(prices),symbol='RELIANCE',
                               settings_path=ROOT/'paper_50k_settings.json')
        env.reset()
        return env

    def test_training_and_paper_preset_match_and_preserve_comparison_defaults(self):
        for filename,mode in [('simulation_50k_settings.json','simulation'),('paper_50k_settings.json','paper')]:
            config=json.loads((ROOT/filename).read_text())
            self.assertEqual(config['execution']['mode'],mode)
            self.assertEqual(config['simulation']['initial_cash_inr'],50000)
            self.assertEqual(config['simulation']['leverage'],5)
            self.assertEqual(config['risk_limits']['max_order_value_inr'],250000)
            self.assertEqual(config['risk_limits']['max_position_value_inr'],250000)
            self.assertEqual(config['risk_limits']['max_daily_loss_inr'],1000)
            self.assertEqual(config['risk_limits']['mandatory_stop_loss_pct'],1)
            self.assertEqual(config['risk_limits']['max_open_positions'],1)
        original=json.loads((ROOT/'paper_settings.json').read_text())
        self.assertEqual(original['paper']['initial_cash_inr'],25000)
        live=json.loads((ROOT/'settings.json').read_text())
        self.assertEqual(live['execution']['mode'],'disabled')
        self.assertFalse(live['risk_limits']['trading_enabled'])

    def test_margin_and_entry_fees_fit_inside_50k_and_use_more_whole_shares(self):
        env=self.env()
        env.step(1,end_of_data=False)
        entry=env.executions[0]
        self.assertEqual(env.initial_cash,50000)
        self.assertGreater(env.quantity,425)
        self.assertLessEqual(entry['notional_inr'],250000)
        self.assertGreaterEqual(env.cash,0)
        self.assertAlmostEqual(env.cash+env.margin_reserved+env.total_charges,50000)
        next_value=entry['price']*(env.quantity+1)
        self.assertTrue(next_value>250000 or next_value/5+env._fees(next_value,'BUY')['total']>50000)
        self.assertEqual(entry['quantity'],int(entry['quantity']))

    def test_full_quantity_profit_and_fee_accounting_release_all_margin(self):
        env=self.env([200,200,201,201,201,201])
        env.step(1,end_of_data=False)
        entry=env.executions[0]
        env.step(2,end_of_data=False)
        exit_=env.executions[-1]
        gross=entry['quantity']*(exit_['price']-entry['price'])
        self.assertAlmostEqual(env.cash,50000+gross-env.total_charges)
        self.assertEqual(env.margin_reserved,0)
        self.assertEqual(env.quantity,0)
        fees=env.charges.estimate_orders([(entry['notional_inr'],'BUY'),(exit_['notional_inr'],'SELL')])
        self.assertAlmostEqual(env.total_charges,fees['total'])

    def test_1000_daily_loss_can_trigger_before_one_percent_price_stop(self):
        env=self.env()
        env.step(1,end_of_data=False)
        env.df.loc[2,'low']=198
        _,_,done,_,info=env.step(0,end_of_data=False)
        self.assertTrue(done)
        self.assertFalse(env.gatekeeper.trading_enabled)
        self.assertEqual(env.executions[-1]['reason'],'daily_loss')
        self.assertGreater(env.executions[-1]['price'],198.1)
        self.assertEqual(env.quantity,0)
        self.assertLessEqual(info['daily_pnl'],-1000)

    def test_old_85k_profile_model_cannot_silently_use_250k_sizing(self):
        with tempfile.TemporaryDirectory() as tmp:
            model=Mock()
            old=TradingEnvironment(self.frame(),symbol='RELIANCE',settings_path=ROOT/'paper_settings.json')
            model.metadata={'simulator_version':4,'execution_profile':old.execution_profile()}
            run=PaperRun(model,ROOT/'paper_50k_settings.json',Path(tmp)/'report','RELIANCE')
            with self.assertRaisesRegex(ValueError,'retrain'):
                run.initialize(self.frame().assign(lstm_score=0))

    def test_training_and_paper_reports_use_50k_preset_and_disclose_diagnostic_flag(self):
        from train_multisession import train_multisession
        from test_multisession import market_days
        from paper_trade import FrozenPolicy
        from data_loader import load_intraday_csv
        with tempfile.TemporaryDirectory() as tmp, \
             patch('requests.sessions.Session.request',side_effect=AssertionError('No external HTTP')), \
             patch('torch.cuda.is_available',return_value=False):
            root=Path(tmp)
            data=root/'training.csv'
            market_days(8).to_csv(data,index=False)
            model_dir=root/'model'
            report=train_multisession(data,ROOT/'simulation_50k_settings.json',model_dir,
                symbol='RELIANCE',lstm_epochs=1,rl_timesteps=32,seed=3,allow_weak_forecast=True)
            self.assertEqual(report['execution_profile']['initial_cash_inr'],50000)
            self.assertEqual(report['execution_profile']['max_position_value_inr'],250000)
            model=FrozenPolicy(model_dir,'RELIANCE')
            # Replay a genuinely later synthetic session, without online learning.
            raw=market_days(9)
            later=root/'practice.csv'
            raw[raw.timestamp.str.startswith(raw.timestamp.str[:10].max())].to_csv(later,index=False)
            result=PaperRun(model,ROOT/'paper_50k_settings.json',root/'paper','RELIANCE').replay(load_intraday_csv(later))
            self.assertEqual(result['initial_cash_inr'],50000)
            self.assertFalse(result['uses_real_money'])
            self.assertFalse(result['learning_during_session'])
            self.assertEqual(result['forecast_validation_passed'],report['forecast_diagnostics']['validation_passed'])
            self.assertEqual(result['forecast_validation_override_used'],
                             report['forecast_diagnostics']['validation_passed'] is False)


class Paper50kLiveIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_fake_presets_cannot_authenticate_live_adapter(self):
        for filename in ('paper_50k_settings.json','simulation_50k_settings.json'):
            with tempfile.TemporaryDirectory() as tmp:
                token=Mock(side_effect=AssertionError('No broker authentication allowed'))
                adapter=LiveGrowwExecutionAdapter(tmp,Path(tmp)/'journal',tick_size='.05',
                    settings_path=ROOT/filename,token_provider=token,alert=lambda _:None,
                    clock=lambda:datetime(2026,11,16,10,tzinfo=IST))
                with self.assertRaises(ExecutionHalted):
                    await adapter.run_market_session('2885','RELIANCE')
                token.assert_not_called()


if __name__=='__main__': unittest.main()
