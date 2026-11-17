"""Independent policy market state: causality, compatibility and accounting gates."""
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
import torch
from data_loader import prepare_intraday_frame, load_intraday_csv
from market_context import (MODE, SCHEMA, MARKET_COLUMNS, MARKET_NAMES, OBSERVATION_NAMES,
                            prepare_market_context, market_matrix, schema_diagnostics)
from multi_session import load_sessions, score_session, SessionEnvironment
from paper_trade import FrozenPolicy, PaperRun
from rl_brain import TradingEnvironment
from train_multisession import train_multisession
from test_multisession import market_days
from groww_env import ExecutionHalted, IST, LiveGrowwExecutionAdapter

ROOT=Path(__file__).resolve().parent


class MarketContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.settings=ROOT/'simulation_market_50k_settings.json'
        self.config=json.loads(self.settings.read_text())
        self.csv=self.root/'sessions.csv'; market_days(8).to_csv(self.csv,index=False)
        self.http=patch('requests.sessions.Session.request',side_effect=AssertionError('No broker/network calls'))
        self.http.start(); self.addCleanup(self.http.stop)
        self.cpu=patch('torch.cuda.is_available',return_value=False)
        self.cpu.start(); self.addCleanup(self.cpu.stop)

    def day(self):
        full=prepare_intraday_frame(market_days(1))
        return prepare_market_context(full).dropna().assign(lstm_score=0)

    def test_zero_forecast_does_not_collapse_flat_policy_state(self):
        day=self.day().iloc[59:]
        env=TradingEnvironment(day,symbol='RELIANCE',settings_path=self.settings)
        first,_=env.reset()
        second,_,_,_,_=env.step(0,end_of_data=False)
        self.assertEqual(env.simulator_version,5)
        self.assertEqual(env.observation_space.shape,(13,))
        self.assertEqual(tuple(env.observation_names),OBSERVATION_NAMES)
        np.testing.assert_array_equal(first[:4],[0,1,0,0])
        np.testing.assert_array_equal(second[:4],[0,1,0,0])
        self.assertFalse(np.array_equal(first[4:],second[4:]))
        self.assertNotEqual(first[4],second[4])  # Actual price return changes, not merely clock.
        self.assertEqual(env.cash,50000)
        self.assertEqual(env.total_charges,0)
        self.assertTrue(env.observation_space.contains(first))

    def test_indicators_and_observations_are_causal_across_future_mutations(self):
        raw=market_days(1)
        before=prepare_market_context(prepare_intraday_frame(raw))
        changed=raw.copy()
        changed.loc[120:,['open','high','low','close']]*=1.1
        changed.loc[120:,'volume']*=10
        after=prepare_market_context(prepare_intraday_frame(changed))
        pd.testing.assert_frame_equal(before.iloc[:120].loc[:,MARKET_COLUMNS],
                                      after.iloc[:120].loc[:,MARKET_COLUMNS])
        prefix=prepare_market_context(prepare_intraday_frame(raw.iloc[:120]))
        pd.testing.assert_frame_equal(before.iloc[:120].loc[:,MARKET_COLUMNS],prefix.loc[:,MARKET_COLUMNS])

    def test_context_survives_forecast_warmup_slicing_without_resetting_history(self):
        sessions=load_sessions(self.csv,self.config)
        first=sessions[0]
        sliced=first.iloc[59:].assign(lstm_score=0)
        env=TradingEnvironment(sliced,symbol='RELIANCE',settings_path=self.settings)
        observation,_=env.reset()
        np.testing.assert_allclose(observation[4:],market_matrix(first.iloc[[59]])[0],rtol=1e-6)
        self.assertTrue(np.isfinite(market_matrix(first)).all())
        # Recomputing after slicing would lose the five/ten-bar context; refuse it.
        missing=sliced.drop(columns=list(MARKET_COLUMNS))
        with self.assertRaisesRegex(ValueError,'Missing policy market context'):
            TradingEnvironment(missing,symbol='RELIANCE',settings_path=self.settings)

    def test_context_resets_each_day_and_unknown_schema_is_rejected(self):
        sessions=load_sessions(self.csv,self.config)
        raw=pd.read_csv(self.csv)
        second=raw[raw.timestamp.str.startswith('2026-10-06')]
        expected=prepare_market_context(prepare_intraday_frame(second)).dropna()
        np.testing.assert_array_equal(market_matrix(sessions[1]),market_matrix(expected))
        invalid=json.loads(self.settings.read_text()); invalid['simulation']['policy_observation']='typo'
        path=self.root/'bad.json'; path.write_text(json.dumps(invalid))
        with self.assertRaisesRegex(ValueError,'Unknown policy observation'):
            TradingEnvironment(self.day(),symbol='RELIANCE',settings_path=path)

    def test_accounting_and_loss_gate_remain_independent_of_market_inputs(self):
        day=self.day().iloc[59:].copy()
        env=TradingEnvironment(day,symbol='RELIANCE',settings_path=self.settings)
        env.reset(); env.step(1,end_of_data=False)
        self.assertGreater(env.quantity,1)
        self.assertLessEqual(env.executions[0]['notional_inr'],250000)
        self.assertGreaterEqual(env.cash,0)
        env.df.loc[2,'low']=float(env.df.loc[2,'open'])*.98
        _,_,done,_,info=env.step(0,end_of_data=False)
        self.assertTrue(done)
        self.assertFalse(env.gatekeeper.trading_enabled)
        self.assertEqual(env.quantity,0)
        self.assertLessEqual(info['daily_pnl'],-1000)

    def test_legacy_checkpoints_cannot_silently_run_thirteen_input_preset(self):
        day=self.day()
        old=TradingEnvironment(day,symbol='RELIANCE',settings_path=ROOT/'simulation_50k_settings.json')
        self.assertEqual(old.observation_space.shape,(4,))
        model=Mock(); model.metadata={'simulator_version':4,'execution_profile':old.execution_profile()}
        run=PaperRun(model,ROOT/'paper_market_50k_settings.json',self.root/'old','RELIANCE')
        with self.assertRaisesRegex(ValueError,'retrain'):
            run.initialize(day)

    def test_training_reload_and_paper_use_same_thirteen_input_vectors(self):
        output=self.root/'model'
        metrics=train_multisession(self.csv,self.settings,output,symbol='RELIANCE',lstm_epochs=1,
                                  rl_timesteps=32,seed=3,allow_weak_forecast=True)
        self.assertEqual(metrics['simulator_version'],5)
        self.assertEqual(metrics['policy_observation_names'],list(OBSERVATION_NAMES))
        self.assertEqual(metrics['policy_input_diagnostics']['observation_dimension'],13)
        self.assertGreater(metrics['policy_input_diagnostics']['varying_market_inputs'],1)
        model=FrozenPolicy(output,'RELIANCE')
        self.assertEqual(model.policy.observation_space.shape,(13,))
        raw=pd.read_csv(self.csv)
        raw=raw[raw.timestamp.str.startswith('2026-10-05')]
        features=prepare_intraday_frame(raw)
        scored=model.score(features)
        env=TradingEnvironment(scored,symbol='RELIANCE',settings_path=self.settings)
        train_observation,_=env.reset()
        run=PaperRun(model,ROOT/'paper_market_50k_settings.json',self.root/'paper','RELIANCE')
        run.initialize(scored)
        np.testing.assert_array_equal(run.observation,train_observation)
        with patch.object(model.policy,'learn',side_effect=AssertionError('Paper never learns')):
            report=run.replay(features)
        self.assertEqual(report['simulator_version'],5)
        self.assertEqual(report['initial_cash_inr'],50000)
        self.assertFalse(report['uses_real_money'])
        self.assertFalse(report['learning_during_session'])
        self.assertEqual(report['policy_observation_names'],list(OBSERVATION_NAMES))
        # A v5 checkpoint cannot hide a mismatched/missing schema behind shape alone.
        metadata=json.loads((output/'run_metrics.json').read_text())
        metadata['policy_observation_names']=[]
        (output/'run_metrics.json').write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError,'observation schema'):
            FrozenPolicy(output,'RELIANCE')


class MarketLiveIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_market_context_presets_cannot_enable_live_authentication(self):
        for name in ('simulation_market_50k_settings.json','paper_market_50k_settings.json'):
            with tempfile.TemporaryDirectory() as tmp:
                token=Mock(side_effect=AssertionError('No authentication'))
                adapter=LiveGrowwExecutionAdapter(tmp,Path(tmp)/'journal',settings_path=ROOT/name,
                    tick_size='.05',token_provider=token,alert=lambda _:None,
                    clock=lambda:datetime(2026,11,16,10,tzinfo=IST))
                with self.assertRaises(ExecutionHalted):
                    await adapter.run_market_session('2885','RELIANCE')
                token.assert_not_called()


if __name__=='__main__': unittest.main()
