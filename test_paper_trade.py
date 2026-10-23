"""Paper execution uses real saved models with no external HTTP or broker orders."""
import copy
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import torch
from data_loader import GrowwCredentials, load_intraday_csv, prepare_intraday_frame
from live_trade import main as live_main, verify_paper_report
from main import train_pipeline
from paper_trade import FrozenPolicy, PaperRun, paper_settings
from paper_market_data import IST, PaperDataError, ReadOnlyGrowwCandles
from test_training_smoke import _write_session_csv


class PaperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.shared = tempfile.TemporaryDirectory()
        cls.root = Path(cls.shared.name)
        cls.csv = cls.root/'session.csv'
        _write_session_csv(cls.csv)
        frame = pd.read_csv(cls.csv)
        frame[['open','high','low','close']] = 100.0
        frame.to_csv(cls.csv,index=False)
        cls.features = load_intraday_csv(cls.csv)
        cls.config = json.loads(Path(__file__).with_name('paper_settings.json').read_text())
        # Preserve one-share regression coverage independently of shipped defaults.
        cls.config.pop('simulation', None)
        cls.config['paper']['initial_cash_inr'] = 100000
        cls.config['risk_limits']['max_order_value_inr'] = 5000
        cls.config['risk_limits']['max_position_value_inr'] = 5000
        train = copy.deepcopy(cls.config)
        train['execution']['mode'] = 'simulation'
        cls.train_settings = cls.root/'simulation.json'
        cls.train_settings.write_text(json.dumps(train))
        cls.models = {}
        for algo in ('PPO','DQN'):
            out = cls.root/algo
            train_pipeline(cls.csv,cls.train_settings,out,symbol='RELIANCE',algorithm=algo,
                           lstm_epochs=1,rl_timesteps=64,seed=3)
            cls.models[algo] = out

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.settings = self.root/'paper.json'
        self.settings.write_text(json.dumps(self.config))
        self.http = patch('requests.sessions.Session.request',side_effect=AssertionError('External HTTP forbidden'))
        self.http.start()
        self.addCleanup(self.http.stop)
        self.orders = patch('growwapi.GrowwAPI.place_order',side_effect=AssertionError('Real order forbidden'))
        self.order_mock = self.orders.start()
        self.addCleanup(self.orders.stop)

    def run_object(self, algo='PPO', **kwargs):
        model = FrozenPolicy(self.models[algo],'RELIANCE')
        return PaperRun(model,self.settings,self.root/'out','RELIANCE',**kwargs)

    def test_both_algorithms_replay_without_credentials_or_learning(self):
        for algo in ('PPO','DQN'):
            with self.subTest(algorithm=algo):
                model = FrozenPolicy(self.models[algo],'RELIANCE')
                run = PaperRun(model,self.settings,self.root/algo,'RELIANCE')
                before = {k:v.clone() for k,v in model.lstm.state_dict().items()}
                with patch.object(model.policy,'learn',side_effect=AssertionError('Frozen paper policy')):
                    report = run.replay(self.features)
                self.assertGreater(report['decisions'],0)
                self.assertFalse(report['uses_real_money'])
                self.assertFalse(report['learning_during_session'])
                self.assertFalse(report['auto_switch_to_live'])
                self.assertEqual(report['mode'],'paper')
                self.assertTrue(report['overlaps_training'])
                for key,value in before.items():
                    self.assertTrue(torch.equal(value,model.lstm.state_dict()[key]))
                self.assertTrue((run.output/'paper_report.json').exists())
                self.assertTrue((run.output/'equity_curve.csv').exists())
        self.order_mock.assert_not_called()

    def test_paper_fills_charges_and_cash_reconcile(self):
        run = self.run_object()
        run.model.action = Mock(side_effect=lambda obs:1 if obs[2]==0 else 2)
        report = run.replay(self.features)
        self.assertGreater(report['executed_orders'],0)
        self.assertGreater(report['estimated_charges_inr'],0)
        self.assertAlmostEqual(report['net_pnl_inr'],report['final_cash_inr']-report['initial_cash_inr'])
        expected = run.env.charges.estimate_orders([(x['price'],x['side']) for x in run.env.executions])
        self.assertAlmostEqual(expected['total'],report['estimated_charges_inr'])
        self.assertEqual(run.env.position,0)
        self.order_mock.assert_not_called()

    def test_future_candles_do_not_change_past_scores(self):
        model = FrozenPolicy(self.models['PPO'],'RELIANCE')
        prefix = model.score(self.features.iloc[:150])
        full = model.score(self.features)
        np.testing.assert_allclose(prefix.lstm_score,full.loc[prefix.index].lstm_score,atol=1e-7)

    def test_paper_refuses_live_settings_and_existing_output(self):
        config = copy.deepcopy(self.config)
        config['execution']['mode']='live'
        self.settings.write_text(json.dumps(config))
        with self.assertRaises(ValueError):
            self.run_object()
        self.settings.write_text(json.dumps(self.config))
        self.run_object()
        with self.assertRaises(FileExistsError):
            self.run_object()
        self.order_mock.assert_not_called()

    def test_fake_cash_override_and_invalid_cash(self):
        self.assertEqual(paper_settings(self.settings,25000)[1],25000)
        for cash in (0,-1,True,float('nan')):
            with self.subTest(cash=cash), self.assertRaises(ValueError):
                paper_settings(self.settings,cash)

    def test_invalid_checkpoint_symbol_is_rejected(self):
        with self.assertRaises(ValueError):
            FrozenPolicy(self.models['PPO'],'INFY')

    def test_simulated_daily_loss_halts_and_records_report(self):
        config = copy.deepcopy(self.config)
        config['risk_limits']['max_daily_loss_inr']=5
        self.settings.write_text(json.dumps(config))
        run = self.run_object()
        run.model.action = Mock(side_effect=lambda obs:1 if obs[2]==0 else 0)
        frame = self.features.copy()
        frame.iloc[90:,frame.columns.get_indexer(['open','high','low','close'])] = 90
        report = run.replay(frame)
        self.assertEqual(report['status'],'risk_halt')
        self.assertEqual(run.env.position,0)
        self.assertGreater(report['forced_exits'],0)
        self.order_mock.assert_not_called()

    def test_streaming_does_not_replay_startup_history(self):
        run = self.run_object(source='groww_completed_candles')
        run.model.action = Mock(side_effect=lambda obs:1 if obs[2]==0 else 0)
        raw = pd.read_csv(self.csv)
        start = datetime(2026,10,20,10,36,tzinfo=IST)  # 81 completed bars
        times = iter([start,start,start+timedelta(minutes=1),start+timedelta(minutes=2)])
        class Provider:
            def __init__(self): self.calls=0
            def fetch(self,now):
                self.calls+=1
                if self.calls>2: raise KeyboardInterrupt()
                return raw.iloc[:80+self.calls].copy()
        report = run.poll(Provider(),clock=lambda:next(times),sleep=lambda seconds:None)
        self.assertEqual(report['decisions'],1)
        self.assertEqual(len(run.decisions),1)
        self.assertEqual(report['status'],'interrupted')
        self.assertEqual(run.env.position,0)
        self.assertEqual(report['executed_orders'],2)  # one fake buy and exit on Ctrl+C
        self.order_mock.assert_not_called()

    def test_stale_live_data_stops_without_orders(self):
        run = self.run_object(source='groww_completed_candles')
        provider = Mock()
        provider.fetch.return_value=pd.read_csv(self.csv).iloc[:80]
        with self.assertRaises(PaperDataError):
            run.poll(provider,clock=lambda:datetime(2026,10,20,12,tzinfo=IST),sleep=lambda _:None)
        self.assertEqual(json.loads((run.output/'paper_report.json').read_text())['status'],'data_or_policy_error')
        self.order_mock.assert_not_called()

    def test_changed_live_history_fails_closed_and_saves_report(self):
        run=self.run_object(source='groww_completed_candles')
        raw=pd.read_csv(self.csv)
        times=iter([datetime(2026,10,20,10,36,tzinfo=IST),
                    datetime(2026,10,20,10,36,tzinfo=IST),
                    datetime(2026,10,20,10,37,tzinfo=IST)])
        first=raw.iloc[:81].copy()
        second=raw.iloc[:82].copy()
        second.loc[10,'volume']+=1
        provider=Mock()
        provider.fetch.side_effect=[first,second]
        with self.assertRaises(PaperDataError):
            run.poll(provider,clock=lambda:next(times),sleep=lambda _:None)
        self.assertEqual(run.report()['status'],'data_or_policy_error')
        self.assertEqual(run.report()['executed_orders'],0)
        self.order_mock.assert_not_called()

    def test_replay_on_next_day_is_eligible_only_with_actual_executions(self):
        run=self.run_object()
        frame=self.features.copy()
        frame.index+=pd.Timedelta(days=1)
        run.model.action=Mock(return_value=0)
        report=run.replay(frame)
        self.assertFalse(report['overlaps_training'])
        self.assertEqual(report['executed_orders'],0)
        with self.assertRaises(ValueError):
            verify_paper_report(run.output/'paper_report.json',self.models['PPO'],'RELIANCE')
        self.order_mock.assert_not_called()

    def test_live_report_gate_binds_exact_model_and_requires_out_of_sample(self):
        run = self.run_object()
        run.model.action = Mock(side_effect=[1,2]+[0]*len(self.features))
        run.replay(self.features)
        report_path = run.output/'paper_report.json'
        with self.assertRaises(ValueError):
            verify_paper_report(report_path,self.models['PPO'],'RELIANCE')
        model = FrozenPolicy(self.models['PPO'],'RELIANCE')
        outsample = PaperRun(model,self.settings,self.root/'later-session','RELIANCE')
        model.action = Mock(side_effect=[1,2]+[0]*len(self.features))
        frame = self.features.copy()
        frame.index = frame.index + pd.Timedelta(days=1)
        report = outsample.replay(frame)
        self.assertFalse(report['overlaps_training'])
        path = outsample.output/'paper_report.json'
        self.assertEqual(verify_paper_report(path,self.models['PPO'],'RELIANCE')['mode'],'paper')
        with self.assertRaises(ValueError):
            verify_paper_report(path,self.models['DQN'],'RELIANCE')

    def test_manual_live_entry_requires_matching_paper_and_all_explicit_gates(self):
        run=self.run_object()
        run.model.action=Mock(side_effect=[1,2]+[0]*len(self.features))
        frame=self.features.copy()
        frame.index+=pd.Timedelta(days=1)
        run.replay(frame)
        config=copy.deepcopy(self.config)
        config['execution']['mode']='live'
        config['charges']['rates_verified']=True
        self.settings.write_text(json.dumps(config))
        argv=['--settings',str(self.settings),'--model-dir',str(self.models['PPO']),
              '--paper-report',str(run.output/'paper_report.json'),'--symbol','RELIANCE',
              '--instrument-token','2885','--tick-size','0.05',
              '--journal',str(self.root/'fresh-live.jsonl'),'--confirm-real-money']
        adapter=Mock()
        adapter.run_market_session=AsyncMock()
        with patch('groww_env.LiveGrowwExecutionAdapter',return_value=adapter) as factory:
            with patch('growwapi.GrowwAPI.get_access_token') as auth, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(live_main(argv),0)
                factory.assert_called_once()
                adapter.run_market_session.assert_awaited_once_with('2885','RELIANCE')
                auth.assert_not_called()
        self.order_mock.assert_not_called()

    def test_live_cli_requires_confirmation_before_authentication(self):
        argv=['--settings',str(self.settings),'--model-dir',str(self.models['PPO']),
              '--paper-report','missing.json','--symbol','RELIANCE','--instrument-token','2885',
              '--tick-size','0.05','--journal',str(self.root/'live.jsonl')]
        with patch('growwapi.GrowwAPI.get_access_token') as auth:
            with self.assertRaises(SystemExit) as exit:
                live_main(argv)
            self.assertEqual(exit.exception.code,2)
            auth.assert_not_called()
            with self.assertRaises(SystemExit):
                live_main(argv+['--confirm-real-money'])  # paper config still blocked
            auth.assert_not_called()
        self.order_mock.assert_not_called()


class MarketDataTests(unittest.TestCase):
    def test_read_only_source_excludes_forming_candle(self):
        client=Mock()
        client.get_historical_candles.return_value={'candles':[
            ['2026-10-20 10:00:00',100,101,99,100,1000],
            ['2026-10-20 10:01:00',100,101,99,100,1000]]}
        source=ReadOnlyGrowwCandles(client,'RELIANCE')
        frame=source.fetch(datetime(2026,10,20,10,1,30,tzinfo=IST))
        self.assertEqual(len(frame),1)
        client.get_historical_candles.assert_called_once()
        client.place_order.assert_not_called()
        client.get_positions_for_user.assert_not_called()
        self.assertFalse(hasattr(source,'place_order'))

    def test_access_token_uses_only_client_factory(self):
        client=Mock()
        factory=Mock(return_value=client)
        provider=Mock()
        with patch('paper_market_data.load_credentials',return_value=GrowwCredentials(access_token='offline-token')):
            source=ReadOnlyGrowwCandles.connect('unused.json','RELIANCE',
                       client_factory=factory,token_provider=provider)
        factory.assert_called_once_with('offline-token')
        provider.assert_not_called()
        self.assertFalse(hasattr(source,'place_order'))
        client.place_order.assert_not_called()

    def test_api_key_and_secret_exchange_is_supported_without_logging(self):
        factory=Mock(return_value=Mock())
        provider=Mock(return_value='offline-token')
        credentials=GrowwCredentials(api_key='offline-key',api_secret='offline-secret')
        with patch('paper_market_data.load_credentials',return_value=credentials):
            ReadOnlyGrowwCandles.connect('unused.json','RELIANCE',
                   client_factory=factory,token_provider=provider)
        provider.assert_called_once_with(api_key='offline-key',secret='offline-secret')
        factory.return_value.place_order.assert_not_called()
        provider.side_effect=RuntimeError('offline-secret')
        with patch('paper_market_data.load_credentials',return_value=credentials):
            with self.assertRaises(PaperDataError) as result:
                ReadOnlyGrowwCandles.connect('unused.json','RELIANCE',
                   client_factory=factory,token_provider=provider)
        self.assertNotIn('offline-secret',str(result.exception))

    def test_sdk_errors_are_sanitized(self):
        client=Mock()
        client.get_historical_candles.side_effect=RuntimeError('secret-should-never-appear')
        with self.assertRaises(PaperDataError) as result:
            ReadOnlyGrowwCandles(client,'RELIANCE').fetch(datetime(2026,10,20,10,tzinfo=IST))
        self.assertNotIn('secret-should-never-appear',str(result.exception))

    def test_in_memory_indicators_match_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv=Path(tmp)/'data.csv'
            _write_session_csv(csv)
            pd.testing.assert_frame_equal(load_intraday_csv(csv),prepare_intraday_frame(pd.read_csv(csv)))


class LiveCredentialRouteTests(unittest.IsolatedAsyncioTestCase):
    async def check_route(self, values, expected_kwargs):
        from groww_env import LiveGrowwExecutionAdapter
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            config=json.loads(Path(__file__).with_name('settings.json').read_text())
            path=root/'settings.json'
            path.write_text(json.dumps(config))
            provider=Mock(return_value='offline-returned-token')
            client=Mock()
            factory=Mock(return_value=client)
            env={config['groww'][field+'_env']:'' for field in
                 ('api_key','api_secret','totp_secret','access_token')}
            env.update({config['groww'][key+'_env']:value for key,value in values.items()})
            adapter=LiveGrowwExecutionAdapter(root,root/'journal.jsonl',settings_path=path,
                     tick_size='0.05',token_provider=provider,client_factory=factory,
                     clock=lambda:datetime(2026,10,20,10,tzinfo=IST),alert=lambda _:None)
            with patch.dict('os.environ',env), patch('requests.sessions.Session.request',side_effect=AssertionError('HTTP forbidden')):
                await adapter.authenticate()
            if expected_kwargs is None:
                provider.assert_not_called()
            else:
                provider.assert_called_once_with(**expected_kwargs)
            factory.assert_called_once()
            client.place_order.assert_not_called()
            journal=(root/'journal.jsonl').read_text()
            for value in values.values():
                self.assertNotIn(value,journal)
            self.assertNotIn('offline-returned-token',journal)

    async def test_live_api_key_secret_route_is_supported_without_orders(self):
        await self.check_route({'api_key':'offline-key','api_secret':'offline-secret'},
                               {'api_key':'offline-key','secret':'offline-secret'})

    async def test_live_existing_access_token_skips_exchange(self):
        await self.check_route({'access_token':'offline-token'},None)


if __name__=='__main__':
    unittest.main(verbosity=2)
