"""Causal relative inputs, artifact compatibility and nested date selection."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from data_loader import load_intraday_csv, prepare_intraday_frame
from forecast_features import (RAW, RELATIVE, RELATIVE_COLUMNS, RELATIVE_NAMES,
    prepare_features, feature_matrix, feature_names, feature_set_from_metadata)
from forecast_experiment import compare, evaluate_candidate
from multi_session import load_sessions, score_session
from paper_trade import FrozenPolicy
from quant_agent import QuantLSTMBrain
from train_multisession import train_multisession
from test_multisession import market_days
import json


class ForecastFeatureTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.csv=self.root/'data.csv'; market_days(12).to_csv(self.csv,index=False)
        self.config=json.loads(Path(__file__).with_name('simulation_settings.json').read_text())
        self.settings=self.root/'settings.json'; self.settings.write_text(json.dumps(self.config))
        self.network=patch('requests.sessions.Session.request',side_effect=AssertionError('No HTTP'))
        self.network.start(); self.addCleanup(self.network.stop)
        self.cpu=patch('torch.cuda.is_available',return_value=False)
        self.cpu.start(); self.addCleanup(self.cpu.stop)

    def test_relative_features_are_invariant_to_price_and_volume_units(self):
        raw=market_days(1)
        full=prepare_intraday_frame(raw)
        relative=prepare_features(full,RELATIVE).dropna()
        scaled=raw.copy()
        scaled[['open','high','low','close']]*=3
        scaled['volume']*=10
        other=prepare_features(prepare_intraday_frame(scaled),RELATIVE).dropna()
        np.testing.assert_allclose(feature_matrix(relative,RELATIVE),
                                   feature_matrix(other,RELATIVE),atol=1e-10)
        self.assertEqual(feature_matrix(relative,RELATIVE).shape[1],7)
        np.testing.assert_array_equal(relative.close,full.loc[relative.index].close)

    def test_future_rows_do_not_change_past_features_or_warmup(self):
        full=prepare_intraday_frame(market_days(1))
        prefix=prepare_features(full.iloc[:100],RELATIVE)
        changed=full.copy()
        changed.iloc[100:,changed.columns.get_indexer(['open','high','low','close','vwap'])]*=2
        later=prepare_features(changed,RELATIVE)
        pd.testing.assert_frame_equal(prefix.loc[:,RELATIVE_COLUMNS],
                                      later.loc[prefix.index,RELATIVE_COLUMNS])
        raw_days=load_sessions(self.csv,self.config)
        relative_days=load_sessions(self.csv,self.config,RELATIVE)
        for raw,relative in zip(raw_days,relative_days):
            np.testing.assert_array_equal(raw.index,relative.index)
            self.assertTrue(np.isfinite(feature_matrix(relative,RELATIVE)).all())

    def test_saved_feature_schema_rejects_ambiguous_inputs(self):
        self.assertEqual(feature_set_from_metadata({}),RAW)
        valid={'feature_transform':{'name':RELATIVE},'forecast_features':list(RELATIVE_NAMES)}
        self.assertEqual(feature_set_from_metadata(valid),RELATIVE)
        with self.assertRaises(ValueError): feature_set_from_metadata(dict(valid,forecast_features=list(QuantLSTMBrain.FEATURES)))
        with self.assertRaises(ValueError): feature_set_from_metadata({'feature_transform':{'name':'unknown'}})
        with self.assertRaises(ValueError): feature_names('unknown')
        full=prepare_intraday_frame(market_days(1)).dropna()
        with self.assertRaises(ValueError): feature_matrix(full,RELATIVE)

    def fake_fit(self, fit, validation, epochs, device, feature_set):
        scaler=StandardScaler().fit(np.concatenate([feature_matrix(day,feature_set) for day in fit]))
        model=QuantLSTMBrain()
        with torch.no_grad(): model.linear.weight.zero_(); model.linear.bias.zero_()
        return model,scaler,{'selected_epoch':0,'training_mean_return':0}

    def test_each_fold_scoring_day_is_outside_fitting_and_checkpoint_selection(self):
        sessions=load_sessions(self.csv,self.config)[:8]
        calls=[]
        def fit(fitting,validation,epochs,device,feature_set):
            calls.append(([day.index[0].date() for day in fitting],validation[0].index[0].date()))
            return self.fake_fit(fitting,validation,epochs,device,feature_set)
        with patch('forecast_experiment.fit_forecast',side_effect=fit):
            report=evaluate_candidate(sessions,RAW,1,42,torch.device('cpu'))
        self.assertEqual(len(calls),3)
        for call,fold in zip(calls,report['folds']):
            self.assertLess(max(call[0]),call[1])
            self.assertLess(str(call[1]),fold['scoring_dates'][0])
        self.assertEqual(report['folds_beating_both_baselines'],0)

    def test_later_comparison_cannot_switch_the_preselected_candidate(self):
        raw={'pooled_metrics':{'model_mse':.5,'zero_return_mse':1,'training_mean_mse':1},
             'folds_beating_both_baselines':3}
        relative={'pooled_metrics':{'model_mse':.6,'zero_return_mse':1,'training_mean_mse':1},
                  'folds_beating_both_baselines':3}
        with patch('forecast_experiment.evaluate_candidate',side_effect=[raw,relative]), \
             patch('forecast_experiment.fit_forecast',side_effect=self.fake_fit), \
             patch('forecast_experiment.prediction_metrics',return_value={
                 'model_mse':2,'zero_return_mse':1,'training_mean_mse':1}):
            report=compare(self.csv,self.settings,self.root/'comparison',
                fit_end='2026-10-14',validation_end='2026-10-20',epochs=1)
        self.assertEqual(report['selected_feature_set'],RAW)
        self.assertFalse(report['further_research_supported'])
        self.assertTrue((self.root/'comparison'/'comparison_report.json').exists())

    def test_experiment_refuses_input_with_later_evaluation_or_practice_dates(self):
        with self.assertRaisesRegex(ValueError,'later evaluation/practice'):
            compare(self.csv,self.settings,self.root/'invalid',fit_end='2026-10-14',
                    validation_end='2026-10-16',epochs=1)
        self.assertFalse((self.root/'invalid').exists())

    def test_relative_model_reload_matches_policy_training_units(self):
        output=self.root/'model'
        report=train_multisession(self.csv,self.settings,output,symbol='RELIANCE',
            lstm_epochs=1,rl_timesteps=32,seed=3,allow_weak_forecast=True,feature_set=RELATIVE)
        self.assertEqual(report['feature_transform'],{'name':RELATIVE})
        self.assertEqual(report['forecast_features'],list(RELATIVE_NAMES))
        policy=FrozenPolicy(output,'RELIANCE')
        self.assertEqual(policy.feature_set,RELATIVE)
        sessions=load_sessions(self.csv,self.config,RELATIVE)
        # Rebuild the saved scaler from its artifact, not from evaluation data.
        scaler=StandardScaler()
        with np.load(output/'feature_scaler.npz') as values:
            scaler.mean_=values['mean']; scaler.scale_=values['scale']
        scaler.n_features_in_=7
        model=QuantLSTMBrain()
        model.load_state_dict(torch.load(output/'lstm.pt',weights_only=True))
        expected=score_session(model,sessions[-1],scaler,torch.device('cpu'),RELATIVE)
        raw=pd.read_csv(self.csv)
        raw=raw[raw.timestamp.str.startswith(str(sessions[-1].index[0].date()))]
        actual=policy.score(prepare_intraday_frame(raw))
        np.testing.assert_allclose(actual.lstm_score,expected.lstm_score,atol=1e-8)
        self.assertEqual(report['execution_profile']['initial_cash_inr'],25000)


if __name__=='__main__': unittest.main()
