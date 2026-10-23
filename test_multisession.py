"""Offline temporal separation, forecast units, margin resets and end-to-end checks."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from data_loader import load_intraday_csv
from multi_session import (TRANSFORM, SessionEnvironment, SessionWindows, forecast_divisor,
                           load_sessions, score_session, split_sessions)
from paper_trade import FrozenPolicy, PaperRun
from prepare_training_data import prepare, select_sessions
from quant_agent import QuantLSTMBrain
from train_multisession import (evaluate_sessions, fit_forecast, prediction_metrics,
                                train_multisession)


def market_days(count=8, bars=180):
    days=[]
    for day_id, day in enumerate(pd.bdate_range('2026-10-05',periods=count)):
        times=pd.date_range(day+pd.Timedelta(hours=9,minutes=15),periods=bars,freq='min')
        close=100+day_id*10+np.sin(np.arange(bars)/8)*.1
        opening=np.r_[close[0],close[:-1]]
        days.append(pd.DataFrame({'timestamp':times.strftime('%Y-%m-%d %H:%M:%S'),
            'open':opening,'high':np.maximum(opening,close)+.02,
            'low':np.minimum(opening,close)-.02,'close':close,
            'volume':1000+np.arange(bars)%20}))
    return pd.concat(days,ignore_index=True)


class MultiSessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.csv=self.root/'sessions.csv'
        market_days().to_csv(self.csv,index=False)
        self.config=json.loads(Path(__file__).with_name('simulation_settings.json').read_text())
        self.settings=self.root/'settings.json'
        self.settings.write_text(json.dumps(self.config))
        self.network=patch('requests.sessions.Session.request',side_effect=AssertionError('No external HTTP'))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.cpu=patch('torch.cuda.is_available',return_value=False)
        self.cpu.start()
        self.addCleanup(self.cpu.stop)
        self.sessions=load_sessions(self.csv,self.config)

    def test_days_and_targets_never_cross_overnight_boundary(self):
        self.assertEqual(len(self.sessions),8)
        groups=split_sessions(self.sessions)
        all_dates=[frame.index[0].date() for name in ('fit','validation','rl','evaluation')
                   for frame in groups[name]]
        self.assertEqual(all_dates,sorted(set(all_dates)))
        scaler=StandardScaler().fit(self.sessions[0].loc[:,list(QuantLSTMBrain.FEATURES)].to_numpy())
        data=SessionWindows(self.sessions,scaler)
        self.assertEqual(len(data),sum(len(frame)-60 for frame in self.sessions))
        self.assertLess(max(abs(data[index][1].item()) for index in range(len(data))),100)
        for session,end in data.samples:
            self.assertLess(end+1,len(self.sessions[session]))
            self.assertEqual(self.sessions[session].index[end].date(),
                             self.sessions[session].index[end+1].date())
        # RSI is recomputed independently, as for a separately loaded day.
        one=self.root/'one.csv'
        raw=pd.read_csv(self.csv)
        raw[raw.timestamp.str.startswith('2026-10-06')].to_csv(one,index=False)
        expected=load_intraday_csv(one).dropna()
        pd.testing.assert_frame_equal(self.sessions[1],expected)

    def test_missing_minutes_and_duplicate_rows_refused_by_trainer(self):
        raw=pd.read_csv(self.csv)
        raw.drop(100).to_csv(self.csv,index=False)
        with self.assertRaisesRegex(ValueError,'contiguous'):
            load_sessions(self.csv,self.config)
        pd.concat([raw.iloc[:1],raw]).to_csv(self.csv,index=False)
        with self.assertRaisesRegex(ValueError,'unique'):
            load_sessions(self.csv,self.config)
        with self.assertRaisesRegex(ValueError,'four'):
            split_sessions(self.sessions[:3])

    def test_scaler_fits_only_fit_days_and_validation_checkpoint_includes_zero_baseline(self):
        groups=split_sessions(self.sessions)
        torch.manual_seed(3)
        model,scaler,report=fit_forecast(groups['fit'],groups['validation'],1,torch.device('cpu'))
        expected=np.concatenate([day.loc[:,list(QuantLSTMBrain.FEATURES)].to_numpy()
                                 for day in groups['fit']]).mean(axis=0)
        np.testing.assert_allclose(scaler.mean_,expected)
        self.assertLessEqual(report['validation']['model_mse'],report['validation']['zero_return_mse'])
        self.assertIn(report['selected_epoch'],(0,1))
        # Changing future RL/test inputs cannot change forecast fitting.
        altered=copy.deepcopy(groups)
        for name in ('rl','evaluation'):
            for day in altered[name]: day['close']*=10
        torch.manual_seed(3)
        other,other_scaler,other_report=fit_forecast(altered['fit'],altered['validation'],1,torch.device('cpu'))
        np.testing.assert_array_equal(scaler.mean_,other_scaler.mean_)
        self.assertEqual(report,other_report)
        for key,value in model.state_dict().items():
            self.assertTrue(torch.equal(value,other.state_dict()[key]))

    def test_basis_points_roundtrip_and_causal_scoring(self):
        scaler=StandardScaler().fit(self.sessions[0].loc[:,list(QuantLSTMBrain.FEATURES)].to_numpy())
        model=QuantLSTMBrain()
        with torch.no_grad():
            model.linear.weight.zero_(); model.linear.bias.fill_(25)
        scored=score_session(model,self.sessions[0],scaler,torch.device('cpu'))
        np.testing.assert_allclose(scored.lstm_score,.0025)
        prefix=score_session(model,self.sessions[0].iloc[:120],scaler,torch.device('cpu'))
        np.testing.assert_array_equal(prefix.lstm_score,scored.loc[prefix.index].lstm_score)
        self.assertEqual(forecast_divisor({'forecast_transform':TRANSFORM}),10000)
        self.assertEqual(forecast_divisor({}),1)
        with self.assertRaises(ValueError): forecast_divisor({'forecast_transform':{'divide_by':100}})
        with self.assertRaises(ValueError): forecast_divisor({'training_pipeline':'multi_session_v1'})

    def test_training_episodes_reset_margin_positions_and_loss_gate_per_day(self):
        frames=[day.iloc[59:].assign(lstm_score=0) for day in self.sessions[:2]]
        env=SessionEnvironment(frames,'RELIANCE',self.settings)
        env.reset(seed=42)
        first_date=env.env.df.timestamp.iloc[0].date()
        env.step(1)
        self.assertGreater(env.env.quantity,1)
        env.env.gatekeeper.halt_trading()
        observation,_=env.reset()
        self.assertNotEqual(env.env.df.timestamp.iloc[0].date(),first_date)
        self.assertEqual(env.env.quantity,0)
        self.assertEqual(env.env.margin_reserved,0)
        self.assertEqual(env.env.cash,25000)
        self.assertTrue(env.env.gatekeeper.trading_enabled)

    def test_evaluation_carries_equity_instead_of_resetting_daily_losses(self):
        calls=[0]
        def propose(observation,deterministic):
            calls[0]+=1
            return (1 if observation[2]==0 else 2),None
        policy=Mock()
        policy.predict.side_effect=propose
        frames=[day.iloc[59:].assign(lstm_score=0) for day in self.sessions[:2]]
        report=evaluate_sessions(policy,frames,'RELIANCE',self.settings,25000)
        self.assertEqual(report['sessions_evaluated'],2)
        self.assertEqual(report['daily_results'][1]['starting_cash_inr'],
                         report['daily_results'][0]['ending_cash_inr'])
        self.assertAlmostEqual(sum(day['net_pnl_inr'] for day in report['daily_results']),report['net_pnl_inr'])
        self.assertEqual(report['cash'],report['daily_results'][-1]['ending_cash_inr'])

    def test_weak_forecast_gate_skips_policy_and_saves_inspectable_report(self):
        with patch('train_multisession.RLTradingBrain',side_effect=AssertionError('Must not train PPO')):
            report=train_multisession(self.csv,self.settings,self.root/'weak',symbol='RELIANCE',
                                      lstm_epochs=1,rl_timesteps=32,seed=3)
        # Synthetic smooth days commonly fail the strict training-mean baseline.
        self.assertFalse(report['forecast_diagnostics']['validation_passed'])
        self.assertEqual(report['status'],'forecast_validation_failed')
        self.assertTrue((self.root/'weak'/'forecast_report.json').exists())
        self.assertFalse((self.root/'weak'/'rl_policy.zip').exists())
        self.assertNotIn('evaluation',report['forecast_diagnostics'])

    def test_full_training_reload_and_paper_use_identical_prediction_units(self):
        output=self.root/'trained'
        report=train_multisession(self.csv,self.settings,output,symbol='RELIANCE',
            lstm_epochs=1,rl_timesteps=32,seed=3,allow_weak_forecast=True)
        self.assertEqual(report['status'],'trained')
        self.assertEqual(report['forecast_transform'],TRANSFORM)
        self.assertEqual(report['training_diagnostics']['steps'],report['rl_actual_timesteps'])
        self.assertEqual(report['evaluation']['sessions_evaluated'],2)
        policy=FrozenPolicy(output,'RELIANCE')
        raw_model=QuantLSTMBrain()
        raw_model.load_state_dict(torch.load(output/'lstm.pt',weights_only=True))
        groups=split_sessions(self.sessions)
        scaler=StandardScaler().fit(np.concatenate([
            day.loc[:,list(QuantLSTMBrain.FEATURES)].to_numpy() for day in groups['fit']]))
        expected=score_session(raw_model,self.sessions[-1],scaler,torch.device('cpu'))
        actual=policy.score(self.sessions[-1])
        np.testing.assert_allclose(actual.lstm_score,expected.lstm_score,atol=1e-8)
        config=copy.deepcopy(self.config); config['execution']['mode']='paper'
        config['paper']={'initial_cash_inr':25000}
        path=self.root/'paper.json'; path.write_text(json.dumps(config))
        raw=market_days(count=9)
        final_date=raw.timestamp.str[:10].max()
        later=self.root/'later.csv'
        raw[raw.timestamp.str.startswith(final_date)].to_csv(later,index=False)
        paper=PaperRun(policy,path,self.root/'paper','RELIANCE').replay(load_intraday_csv(later))
        self.assertFalse(paper['uses_real_money'])
        self.assertFalse(paper['overlaps_training'])

    def test_preparer_reserves_later_completed_day_and_does_not_fill_gaps(self):
        raw=market_days(count=6)
        # One early bad minute: preparation can select the remaining real section.
        raw=raw.drop(1)
        manifest=prepare(raw,self.root/'data',now=pd.Timestamp('2026-10-13',tz='Asia/Kolkata'))
        training=pd.read_csv(self.root/'data'/'multi_sessions.csv')
        practice=pd.read_csv(self.root/'data'/'practice_multi.csv')
        self.assertEqual(manifest['practice_date'],'2026-10-12')
        self.assertEqual(training.timestamp.str[:10].nunique(),5)
        self.assertEqual(practice.timestamp.str[:10].nunique(),1)
        self.assertLess(training.timestamp.max(),practice.timestamp.min())
        sessions,_=select_sessions(raw,now=pd.Timestamp('2026-10-12',tz='Asia/Kolkata'))
        self.assertEqual(len(sessions),5)  # Current day excluded.
        self.assertEqual(len(sessions[0]),178)
        with self.assertRaises(FileExistsError): prepare(raw,self.root/'data')
        with self.assertRaisesRegex(ValueError,'five'):
            prepare(market_days(count=4),self.root/'short',now=pd.Timestamp('2026-10-13',tz='Asia/Kolkata'))


if __name__=='__main__': unittest.main()
