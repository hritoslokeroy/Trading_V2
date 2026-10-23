"""Independent diagnostic arithmetic; diagnostics never steer model actions."""
import unittest
import pandas as pd
from training_diagnostics import TrainingDiagnostics, forecast_metrics


class DiagnosticsTests(unittest.TestCase):
    def test_training_counters_distinguish_proposals_fills_and_rejections(self):
        callback=TrainingDiagnostics()
        callback.locals={'actions':[1,2,1], 'dones':[False,True,False], 'infos':[
            {'buy_count':1,'equity_change_inr':-10,'reward_penalty':.0001},
            {'forced_exit_count':1,'equity_change_inr':-50,'reward_penalty':.01},
            {'order_rejected':True,'equity_change_inr':0,'reward_penalty':.01}]}
        self.assertTrue(callback._on_step())
        result=callback.summary()
        self.assertEqual(result['action_counts'],{'hold':0,'buy':2,'sell':1})
        self.assertEqual(result['steps'],3)
        self.assertEqual(result['completed_episodes'],1)
        self.assertEqual(result['buy_executions'],1)
        self.assertEqual(result['sell_executions'],0)
        self.assertEqual(result['forced_exits'],1)
        self.assertEqual(result['rejected_entries'],1)
        self.assertEqual(result['summed_equity_changes_inr'],-60)
        self.assertAlmostEqual(result['summed_reward_penalties'],.0201)

    def test_forecast_comparison_excludes_unknown_future_target(self):
        frame=pd.DataFrame({'close':[100,101,101],'lstm_score':[.01,0,999]})
        result=forecast_metrics(frame)
        self.assertEqual(result['samples'],2)
        self.assertAlmostEqual(result['model_mse'],0)
        self.assertAlmostEqual(result['zero_return_mse'],.00005)
        empty=forecast_metrics(frame.iloc[:1])
        self.assertEqual(empty['samples'],0)
        self.assertIsNone(empty['model_mse'])


if __name__=='__main__': unittest.main()
