"""Independent offline checks of economic and risk invariants; no broker access."""
import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from rl_brain import TradingEnvironment
from risk_manager import RiskGatekeeper

class FinancialInvariants(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings=Path(self.tmp.name)/'settings.json'
        self.config={'groww':{},'risk_limits':{'trading_enabled':True,'max_order_value_inr':1000,'max_position_value_inr':1000,'max_daily_loss_inr':500,'max_open_positions':1,'mandatory_stop_loss_pct':1.0,'hard_cutoff_time':'15:15'},'data':{'timezone':'Asia/Kolkata','supported_symbols':['RELIANCE']}}
    def env(self, prices=None, initial_cash=100000, start='2026-10-20 10:00'):
        self.settings.write_text(json.dumps(self.config))
        prices=prices or [100]*6
        frame=pd.DataFrame({'open':prices,'high':prices,'low':prices,'close':prices,'volume':1000,'lstm_score':0.0},index=pd.date_range(start,periods=len(prices),freq='min',tz='Asia/Kolkata'))
        e=TradingEnvironment(frame,symbol='RELIANCE',settings_path=self.settings,initial_cash=initial_cash)
        e.reset();self.addCleanup(e.close)
        return e
    def test_open_position_equity_includes_owned_share(self):
        e=self.env();_,_,_,_,info=e.step(1)
        self.assertAlmostEqual(info['equity'],e.cash+100*e.lot_size,places=6)
    def test_round_trip_pnl_includes_fees(self):
        e=self.env();e.step(1);_,_,_,_,info=e.step(2)
        self.assertAlmostEqual(info['daily_pnl'],e.cash-e.initial_cash,places=6)
    def test_entry_cannot_spend_more_than_available_cash(self):
        e=self.env(initial_cash=50);e.step(1)
        self.assertGreaterEqual(e.cash,0)
        self.assertEqual(e.position,0)
    def test_entry_respects_position_value_limit(self):
        self.config['risk_limits']['max_position_value_inr']=50
        e=self.env();e.step(1)
        self.assertEqual(e.position,0)
    def test_mandatory_stop_closes_falling_position(self):
        e=self.env([100,100,90,90,90,90]);e.step(1);e.step(0)
        self.assertEqual(e.position,0,'Position remains open despite 10% fall and mandatory 1% stop')
    def test_unrealized_loss_triggers_daily_loss_limit(self):
        self.config['risk_limits']['max_daily_loss_inr']=5
        e=self.env([100,100,90,90,90,90]);e.step(1);_,_,terminated,_,_=e.step(0)
        self.assertTrue(terminated,'A >INR 10 open loss does not trigger INR 5 daily-loss limit')
    def test_cutoff_info_matches_post_liquidation_cash(self):
        e=self.env(start='2026-10-20 15:13');e.step(1)
        _,_,terminated,_,info=e.step(0)
        self.assertTrue(terminated);self.assertEqual(e.position,0)
        self.assertAlmostEqual(info['cash'],e.cash,places=6)
    def test_gatekeeper_halt_is_latched(self):
        self.env();g=RiskGatekeeper(self.settings)
        self.assertTrue(g.validate_new_order('RELIANCE',100,0))
        self.assertFalse(g.check_daily_loss(-500))
        self.assertFalse(g.check_daily_loss(0))
        self.assertFalse(g.validate_new_order('RELIANCE',100,0))
    def test_stop_price_calculation(self):
        self.env();g=RiskGatekeeper(self.settings)
        self.assertEqual(float(g.calculate_stop_loss(100,'BUY')),99)
        self.assertEqual(float(g.calculate_stop_loss(100,'SELL')),101)

if __name__=='__main__':unittest.main(verbosity=2)
