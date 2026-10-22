"""Trading invariants around gap fills, clock boundaries and cumulative rewards."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from gymnasium.utils.env_checker import check_env
from rl_brain import TradingEnvironment
from risk_manager import DEFAULT_SETTINGS_PATH


class SimulatorEdgeTests(unittest.TestCase):
    def env(self, *, times=None, prices=None, max_loss=500, cash=100000, fee_bps=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / 'settings.json'
        settings = json.loads(DEFAULT_SETTINGS_PATH.read_text())
        settings['risk_limits'].update(trading_enabled=True, max_daily_loss_inr=max_loss)
        path.write_text(json.dumps(settings))
        prices = prices if prices is not None else [100.] * 6
        times = times if times is not None else pd.date_range('2026-10-20 10:00', periods=len(prices), freq='min', tz='Asia/Kolkata')
        frame = pd.DataFrame({key: prices for key in ['open','high','low','close']}, index=times)
        e = TradingEnvironment(frame, symbol='RELIANCE', settings_path=path, initial_cash=cash,
                               fee_bps=fee_bps, slippage_bps=0, turnover_penalty=0,
                               overtrade_penalty=0, risk_penalty=0)
        self.addCleanup(e.close)
        e.reset()
        return e

    def test_reward_sum_matches_net_cash_pnl(self):
        e = self.env()
        total = 0
        for action in [1,0,2,1,0]:
            _, reward, done, _, info = e.step(action)
            total += reward
        self.assertTrue(done)
        self.assertAlmostEqual(total, (e.cash-e.initial_cash)/e.initial_cash)
        self.assertAlmostEqual(info['daily_pnl'], -info['total_charges'])
        self.assertEqual(info['equity'], info['cash'])

    def test_session_charges_match_the_actual_execution_ledger(self):
        e = self.env()
        for action in (1,2,1,2,0):
            result = e.step(action)
        expected = e.charges.estimate_orders(
            [(fill['price']*fill['quantity'],fill['side']) for fill in e.executions])
        self.assertAlmostEqual(result[4]['total_charges'],expected['total'])
        for key, value in result[4]['charge_breakdown'].items():
            self.assertAlmostEqual(value,expected[key])
        self.assertAlmostEqual(sum(x['charges']['total'] for x in e.executions),expected['total'])
        self.assertAlmostEqual(e.initial_cash-e.cash,expected['total'])
        e.reset()
        self.assertEqual(e.total_charges,0)
        self.assertEqual(e.charge_ledger.summary()['total'],0)

    def test_gap_fills_at_gap_price_instead_of_stale_stop(self):
        e = self.env(prices=[100,100,90,90,90,90])
        e.step(1)
        _, _, _, _, info = e.step(0)
        self.assertEqual(e.position, 0)
        self.assertEqual(e.executions[-1]['price'], 90)
        self.assertEqual(info['forced_exit_count'], 1)
        self.assertAlmostEqual(info['daily_pnl'], -10-info['total_charges'])

    def test_intrabar_stop_does_not_wait_for_close(self):
        e = self.env()
        e.df.loc[2,'low'] = 98
        e.step(1)
        e.step(0)
        self.assertEqual(e.position,0)
        self.assertEqual(e.executions[-1]['price'],99)
        self.assertEqual(e.executions[-1]['reason'],'stop_loss')

    def test_intrabar_loss_limit_precedes_wider_stop(self):
        e = self.env(max_loss=0.5, fee_bps=0)
        e.df.loc[2,'low'] = 99.2
        e.step(1)
        _, _, done, _, info = e.step(0)
        self.assertTrue(done)
        self.assertEqual(e.executions[-1]['price'],99.5)
        self.assertAlmostEqual(info['daily_pnl'],-0.5)

    def test_fees_are_reserved_before_buy(self):
        e = self.env(cash=100)
        _, _, _, _, info = e.step(1)
        self.assertTrue(info['order_rejected'])
        self.assertEqual(e.cash,100)
        self.assertEqual(info['total_charges'],0)

    def test_order_caps_use_slipped_fill_value(self):
        e = self.env()
        e.slippage = 0.01
        e.max_position = 100
        e.step(1)
        self.assertEqual(e.position,0)

    def test_no_entry_can_fill_at_1515(self):
        e = self.env(times=pd.date_range('2026-10-20 15:14',periods=6,freq='min',tz='Asia/Kolkata'))
        _, _, done, _, info = e.step(1)
        self.assertTrue(done)
        self.assertEqual(info['total_executed_trades'],0)
        self.assertEqual(e.cash,e.initial_cash)

    def test_utc_timestamps_obey_kolkata_cutoff(self):
        e = self.env(times=pd.date_range('2026-10-20 09:44',periods=6,freq='min',tz='UTC'))
        self.assertTrue(e.step(1)[2])
        self.assertEqual(e.position,0)

    def test_end_of_data_uses_final_close(self):
        e = self.env()
        e.df.loc[5,['close','high']] = 101
        e.step(1)
        for _ in range(4):
            result=e.step(0)
        self.assertTrue(result[2])
        self.assertEqual(e.executions[-1]['price'],101)
        self.assertEqual(result[4]['forced_exit_count'],1)
        with self.assertRaises(RuntimeError):
            e.step(0)

    def test_input_validation(self):
        for times in [pd.date_range('2026-10-18 10:00',periods=6,freq='min'),
                      pd.date_range('2026-10-20 08:00',periods=6,freq='min'),
                      pd.date_range('2026-10-20 10:00',periods=6,freq='2min')]:
            with self.subTest(times=times), self.assertRaises(ValueError):
                self.env(times=times)
        e=self.env()
        for action in (3, True, -1, 1.5):
            with self.subTest(action=action), self.assertRaises(ValueError):
                e.step(action)

    def test_gym_api_contract(self):
        check_env(self.env(), skip_render_check=True)


if __name__ == '__main__':
    unittest.main()
