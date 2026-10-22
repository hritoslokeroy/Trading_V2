"""Check that reported executions reflect fills, using real training and forced proposals."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from main import train_pipeline
from test_training_smoke import _write_session_csv

class TrainingMetrics(unittest.TestCase):
    def run_pipeline(self,max_order):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);root=Path(t.name)
        csv=root/'session.csv';_write_session_csv(csv)
        frame = pd.read_csv(csv)
        frame[['open', 'high', 'low', 'close']] = 100.0
        frame.to_csv(csv, index=False)
        s=root/'settings.json'
        s.write_text(json.dumps({'groww':{},'risk_limits':{'trading_enabled':True,'max_order_value_inr':max_order,'max_position_value_inr':1000,'max_daily_loss_inr':500,'max_open_positions':1,'mandatory_stop_loss_pct':1.0,'hard_cutoff_time':'15:15'},'data':{'timezone':'Asia/Kolkata','supported_symbols':['RELIANCE']}}))
        with patch('stable_baselines3.PPO.predict',return_value=(np.array(1),None)):
            return train_pipeline(csv,s,root/'out',symbol='RELIANCE',algorithm='PPO',lstm_epochs=1,rl_timesteps=64,seed=3)
    def test_rejected_buys_are_not_reported_as_executed(self):
        r=self.run_pipeline(1)['evaluation']
        self.assertEqual(r['voluntary_trades'],0)
        self.assertEqual(r['buy_trades'],0,f'All orders exceed INR 1 cap, yet metrics report {r}')
    def test_end_of_data_liquidation_is_counted(self):
        r=self.run_pipeline(1000)['evaluation']
        self.assertEqual(r['voluntary_trades'],1)
        self.assertEqual(r['forced_exits'],1,f'Long position is liquidated at end of data, but metrics report {r}')

if __name__=='__main__':unittest.main(verbosity=2)
