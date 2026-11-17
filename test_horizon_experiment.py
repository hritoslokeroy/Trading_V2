"""Research horizon alignment, cost hurdle, risk enforcement and temporal guard."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from data_loader import prepare_intraday_frame
from forecast_features import RELATIVE, prepare_features, feature_matrix
from horizon_experiment import (examples, fit_model, predict, replay, estimated_cost_hurdle,
                                experiment, summarize_days)
from rl_brain import TradingEnvironment
from test_multisession import market_days


class HorizonTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.config=json.loads(Path(__file__).with_name('simulation_settings.json').read_text())
        self.settings=self.root/'settings.json'; self.settings.write_text(json.dumps(self.config))
        self.network=patch('requests.sessions.Session.request',side_effect=AssertionError('No network'))
        self.network.start(); self.addCleanup(self.network.stop)

    def session(self, raw=None):
        raw=market_days(1) if raw is None else raw
        return prepare_features(prepare_intraday_frame(raw),RELATIVE).dropna()

    def fixed_model(self, value):
        model=Mock()
        model.predict.side_effect=lambda x:np.full(len(x),value*10000)
        return model

    def test_labels_are_entry_next_open_to_open_h_minutes_later(self):
        raw=market_days(1)
        values=100*1.001**np.arange(len(raw))
        raw[['open','high','low','close']]=np.column_stack([values]*4)
        day=self.session(raw)
        for horizon in (1,5,15):
            x,y,dates=examples([day],horizon)
            np.testing.assert_allclose(y,1.001**horizon-1,rtol=1e-10)
            self.assertEqual(len(x),len(day)-horizon-60)
            self.assertEqual(x.shape[1],7)
            self.assertEqual(set(dates),{'2026-10-05'})
        other=day.copy(); other.index=other.index+pd.Timedelta(days=1)
        _,joint,_=examples([day,other],15)
        self.assertEqual(len(joint),2*(len(day)-75))

    def test_future_prices_change_labels_but_not_earlier_input_features(self):
        raw=market_days(1)
        day=self.session(raw)
        x,y,_=examples([day],5)
        # Ready row 65 corresponds to raw row 79 after the 14-row RSI warmup.
        altered=raw.copy()
        altered.loc[79,['open','high','low','close']]*=1.01
        changed_x,changed_y,_=examples([self.session(altered)],5)
        np.testing.assert_array_equal(x[0],changed_x[0])
        self.assertNotEqual(y[0],changed_y[0])

    def test_market_prediction_must_cover_fees_and_both_slippage_legs(self):
        day=self.session()
        env=TradingEnvironment(day,symbol='RELIANCE',settings_path=self.settings)
        env.reset()
        hurdle=estimated_cost_hurdle(env,100)
        self.assertGreater(hurdle,2*env.slippage)
        self.assertEqual(env.total_charges,0)  # Quoting never records a fill.
        empty=replay(self.fixed_model(0),day,5,self.settings,'RELIANCE',25000)
        self.assertEqual(empty['executed_orders'],0)
        self.assertEqual(empty['net_pnl_inr'],0)
        small=replay(self.fixed_model(hurdle/2),day,5,self.settings,'RELIANCE',25000)
        self.assertEqual(small['executed_orders'],0)
        self.assertGreater(small['positive_gross_forecasts'],0)

    def test_positions_do_not_overlap_and_planned_exits_hold_five_minutes(self):
        raw=market_days(1); raw[['open','high','low','close']]=100
        day=self.session(raw)
        result=replay(self.fixed_model(.01),day,5,self.settings,'RELIANCE',25000)
        self.assertGreater(result['round_trips'],0)
        self.assertGreater(result['estimated_charges_inr'],0)
        self.assertLess(result['net_pnl_inr'],0) # Flat market still loses costs/slippage.
        self.assertEqual(result['final_quantity'],0)
        fills=result['executions']
        self.assertTrue(all(fill['side']==('BUY' if i%2==0 else 'SELL') for i,fill in enumerate(fills)))
        entry,exit_=fills[:2]
        self.assertEqual(pd.Timestamp(exit_['timestamp'])-pd.Timestamp(entry['timestamp']),pd.Timedelta(minutes=5))
        self.assertLessEqual(entry['notional_inr'],85000)

    def test_stop_loss_can_exit_before_planned_fifteen_minutes(self):
        raw=market_days(1); raw[['open','high','low','close']]=100
        raw.loc[75,'low']=97
        day=self.session(raw)
        result=replay(self.fixed_model(.01),day,15,self.settings,'RELIANCE',25000)
        first_entry,first_exit=result['executions'][:2]
        self.assertEqual(first_exit['reason'],'stop_loss')
        self.assertTrue(first_exit['forced'])
        self.assertLess(pd.Timestamp(first_exit['timestamp'])-pd.Timestamp(first_entry['timestamp']),pd.Timedelta(minutes=15))
        self.assertEqual(result['final_quantity'],0)

    def test_daily_loss_gate_stops_an_aggressive_forecast(self):
        self.config['risk_limits']['max_daily_loss_inr']=5
        self.settings.write_text(json.dumps(self.config))
        result=replay(self.fixed_model(.01),self.session(),15,self.settings,'RELIANCE',25000)
        self.assertEqual(result['entry_count'],1)
        self.assertEqual(result['final_quantity'],0)
        self.assertLessEqual(result['net_pnl_inr'],-5)
        self.assertEqual(result['forced_exits'],1)

    def test_scaler_does_not_fit_later_days_and_targets_are_same_day(self):
        day=self.session()
        model,mean=fit_model([day],5)
        x,y,_=examples([day],5)
        np.testing.assert_allclose(model[0].mean_,x.mean(axis=0))
        self.assertAlmostEqual(mean,float(y.mean()))
        self.assertEqual(len(predict(model,x)),len(y))

    def test_failed_development_screen_does_not_score_later_comparison(self):
        csv=self.root/'input.csv'; market_days(12).to_csv(csv,index=False)
        candidate={'development_screen_passed':False,'trading':{'net_pnl_inr':0}}
        with patch('horizon_experiment.development_folds',return_value=candidate), \
             patch('horizon_experiment.error_metrics',side_effect=AssertionError('No later scoring allowed')):
            result=experiment(csv,self.settings,self.root/'result',fit_end='2026-10-14',validation_end='2026-10-20')
        self.assertIsNone(result['selected_horizon'])
        self.assertIsNone(result['later_comparison'])
        self.assertFalse(result['further_research_supported'])
        self.assertTrue((self.root/'result'/'horizon_report.json').exists())
        with self.assertRaisesRegex(ValueError,'later evaluation/practice'):
            experiment(csv,self.settings,self.root/'bad',fit_end='2026-10-14',validation_end='2026-10-16')

    def test_day_aggregation_preserves_actual_cash_continuity(self):
        days=[{'net_pnl_inr':-100,'ending_cash_inr':24900,'round_trips':1,'estimated_charges_inr':70,'executed_orders':2},
              {'net_pnl_inr':200,'ending_cash_inr':25100,'round_trips':1,'estimated_charges_inr':70,'executed_orders':2}]
        result=summarize_days(days,25000)
        self.assertEqual(result['net_pnl_inr'],100)
        self.assertEqual(result['ending_cash_inr'],25100)
        self.assertEqual(result['no_trade_benchmark_pnl_inr'],0)


if __name__=='__main__': unittest.main()
