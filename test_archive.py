"""Offline archive integrity and no-op action diagnostics; no public downloads."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd
from archive_market_data import collect, read_archive, status
from test_multisession import market_days
from paper_trade import PaperRun
from rl_brain import TradingEnvironment
from training_diagnostics import TrainingDiagnostics
from data_loader import load_intraday_csv


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.path=self.root/'RELIANCE.csv'
        self.now=pd.Timestamp('2026-10-13',tz='Asia/Kolkata')
        self.http=patch('requests.sessions.Session.request',side_effect=AssertionError('No network'))
        self.http.start(); self.addCleanup(self.http.stop)

    def test_accumulates_new_days_and_repeated_runs_are_idempotent(self):
        first=collect(market_days(5),self.path,'RELIANCE',now=self.now)
        self.assertEqual(first['new_days'],5)
        self.assertEqual(first['eligible_sessions'],5)
        before=self.path.read_bytes()
        repeated=collect(market_days(5),self.path,'RELIANCE',now=self.now)
        self.assertEqual(repeated['new_rows'],0)
        self.assertEqual(self.path.read_bytes(),before)
        second=collect(market_days(6),self.path,'RELIANCE',now=self.now)
        self.assertEqual(second['new_days'],1)
        self.assertEqual(second['stored_days'],6)
        raw=pd.read_csv(self.path)
        self.assertFalse(raw.timestamp.duplicated().any())
        self.assertTrue(raw.symbol.eq('RELIANCE').all())
        self.assertTrue(raw.interval.eq('1minute').all())

    def test_provider_revisions_and_added_old_day_minutes_do_not_rewrite_history(self):
        raw=market_days(5)
        collect(raw.drop(100),self.path,'RELIANCE',now=self.now)
        before=self.path.read_bytes()
        raw.loc[50,['open','high','low','close']]+=2
        result=collect(raw,self.path,'RELIANCE',now=self.now)
        self.assertEqual(result['source_revisions_ignored'],1)
        self.assertEqual(result['new_rows'],0)
        self.assertEqual(self.path.read_bytes(),before)
        self.assertEqual(result['eligible_sessions'],4)

    def test_current_days_excluded_and_empty_download_cannot_destroy_archive(self):
        raw=market_days(6)
        result=collect(raw,self.path,'RELIANCE',now=pd.Timestamp('2026-10-12',tz='Asia/Kolkata'))
        self.assertEqual(result['stored_days'],5)
        before=self.path.read_bytes()
        result=collect(raw.iloc[:0],self.path,'RELIANCE',now=self.now)
        self.assertEqual(result['new_days'],0)
        self.assertEqual(self.path.read_bytes(),before)
        with self.assertRaisesRegex(ValueError,'No completed'):
            collect(raw.iloc[:0],self.root/'empty.csv','RELIANCE',now=self.now)
        self.assertFalse((self.root/'empty.csv').exists())

    def test_mixed_symbols_intervals_and_clock_regressions_are_rejected(self):
        raw=market_days(5)
        collect(raw,self.path,'RELIANCE',now=self.now)
        before=self.path.read_bytes()
        for altered in (raw.assign(symbol='INFY'), raw.assign(interval='5minute')):
            with self.assertRaises(ValueError): collect(altered,self.path,'RELIANCE',now=self.now)
            self.assertEqual(self.path.read_bytes(),before)
        with self.assertRaisesRegex(ValueError,'check the clock'):
            collect(raw,self.path,'RELIANCE',now=pd.Timestamp('2026-10-06',tz='Asia/Kolkata'))
        self.assertEqual(self.path.read_bytes(),before)
        with self.assertRaisesRegex(ValueError,'another symbol'):
            read_archive(self.path,'INFY',self.now)
        with self.assertRaisesRegex(ValueError,'coarser'):
            collect(raw.iloc[::5],self.path,'RELIANCE',now=self.now)
        self.assertEqual(self.path.read_bytes(),before)

    def test_atomic_write_failure_preserves_existing_archive(self):
        collect(market_days(5),self.path,'RELIANCE',now=self.now)
        before=self.path.read_bytes()
        with patch('archive_market_data.os.replace',side_effect=OSError('Synthetic disk failure')):
            with self.assertRaises(OSError): collect(market_days(6),self.path,'RELIANCE',now=self.now)
        self.assertEqual(self.path.read_bytes(),before)
        self.assertEqual(list(self.root.glob('*.tmp')),[])

    def test_target_counts_real_eligible_sessions_not_download_attempts(self):
        result=collect(market_days(20),self.path,'RELIANCE',
                       now=pd.Timestamp('2026-11-02',tz='Asia/Kolkata'))
        self.assertEqual(result['eligible_sessions'],20)
        self.assertTrue(result['ready_for_next_development_run'])
        reread=status(read_archive(self.path,'RELIANCE',pd.Timestamp('2026-11-02',tz='Asia/Kolkata')))
        self.assertEqual(reread['eligible_sessions'],20)


class FlatSellTests(unittest.TestCase):
    def test_flat_sell_is_counted_without_changing_rewards_cash_or_charges(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); csv=root/'day.csv'
            market_days(1).to_csv(csv,index=False)
            features=load_intraday_csv(csv).dropna()
            env=TradingEnvironment(features,symbol='RELIANCE',
                settings_path=Path(__file__).with_name('simulation_settings.json'))
            env.reset()
            _,reward,_,_,info=env.step(2,end_of_data=False)
            self.assertTrue(info['sell_while_flat'])
            self.assertEqual(reward,0)
            self.assertEqual(env.cash,25000)
            self.assertEqual(env.total_charges,0)
            self.assertEqual(env.executions,[])
            env.step(1,end_of_data=False)
            _,_,_,_,info=env.step(2,end_of_data=False)
            self.assertFalse(info['sell_while_flat'])
            self.assertEqual(info['sell_count'],1)
            callback=TrainingDiagnostics()
            callback.locals={'actions':[2],'dones':[False],
                'infos':[{'sell_while_flat':True,'sell_count':0}]}
            callback._on_step()
            self.assertEqual(callback.summary()['sell_while_flat'],1)
            self.assertEqual(callback.summary()['sell_executions'],0)

    def test_paper_report_distinguishes_flat_sell_decisions_from_orders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); csv=root/'day.csv'
            market_days(1).to_csv(csv,index=False)
            features=load_intraday_csv(csv).dropna()
            model=Mock()
            model.metadata={'algorithm':'PPO','simulator_version':4,
                'execution_profile':TradingEnvironment(features,symbol='RELIANCE',
                    settings_path=Path(__file__).with_name('simulation_settings.json')).execution_profile()}
            model.score.side_effect=lambda data:data.assign(lstm_score=0)
            model.action.return_value=2
            model.fingerprint='offline-test'
            run=PaperRun(model,Path(__file__).with_name('paper_settings.json'),root/'paper','RELIANCE')
            report=run.replay(features)
            self.assertEqual(report['sell_while_flat'],report['decisions'])
            self.assertEqual(report['executed_orders'],0)
            self.assertEqual(report['net_pnl_inr'],0)


if __name__=='__main__': unittest.main()
