"""Session rounding, worked examples and actual statement reconciliation."""
from decimal import Decimal
import unittest
from charges import IntradayCharges, IntradayChargeLedger, reconcile_statement


class ChargeLedgerTests(unittest.TestCase):
    def test_profit_example_with_paise_reporting(self):
        bill = IntradayCharges().estimate_round_trip(100, 200, 215)
        self.assertEqual(bill, {'brokerage': 40.0, 'stt': 5.38, 'stamp_duty': 0.60,
                               'exchange': 1.23, 'sebi': 0.04, 'ipft': 0.04,
                               'gst': 7.44, 'total': 54.73,
                               'gross_pnl': 1500.0, 'net_pnl': 1445.27})

    def test_loss_example_includes_correct_gst(self):
        bill = IntradayCharges().estimate_round_trip(100, 200, 185)
        self.assertEqual(bill['brokerage'], 38.50)
        self.assertEqual(bill['stt'], 4.63)
        self.assertEqual(bill['stamp_duty'], 0.60)
        self.assertEqual(bill['gst'], 7.15)
        self.assertEqual(bill['total'], 52.10)
        self.assertEqual(bill['net_pnl'], -1552.10)

    def test_gst_rounding_happens_after_aggregation(self):
        tariff = IntradayCharges()
        # Individually rounded orders differ by one paise from a session bill.
        separate = tariff.estimate(20000,'BUY')['total'] + tariff.estimate(21500,'SELL')['total']
        self.assertAlmostEqual(separate,54.72)
        self.assertEqual(tariff.estimate_orders([(20000,'BUY'),(21500,'SELL')])['total'],54.73)

    def test_brokerage_cap_is_per_order_not_per_session(self):
        tariff = IntradayCharges()
        bill = tariff.estimate_orders([(20000,'BUY')]*3 + [(21500,'SELL')]*3)
        self.assertEqual(bill['brokerage'],120)

    def test_small_order_brokerage_rounds_per_executed_order(self):
        bill = IntradayCharges().estimate_orders([(Decimal('178.03'),'BUY')]*10)
        self.assertEqual(bill['brokerage'],44.50)

    def test_unknown_rounding_policy_is_rejected(self):
        for setting in ({'aggregation':'order'}, {'rounding':'ceil_rupee'}):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                IntradayCharges.from_settings({'charges':setting})

    def test_read_only_quotes_and_marginal_debits_reconcile(self):
        ledger = IntradayChargeLedger(IntradayCharges())
        zero = ledger.summary()
        quote = ledger.quote(20000,'BUY')
        self.assertEqual(ledger.summary(),zero)
        self.assertEqual(ledger.quote(20000,'BUY'),quote)
        buy = ledger.record(20000,'BUY')
        sell = ledger.record(21500,'SELL')
        for key, amount in ledger.summary().items():
            self.assertEqual(Decimal(str(buy[key]))+Decimal(str(sell[key])),Decimal(str(amount)))
        self.assertEqual(ledger.summary()['total'],54.73)

    def test_sub_paise_heads_are_not_lost_on_many_small_orders(self):
        ledger = IntradayChargeLedger(IntradayCharges())
        for _ in range(100):
            ledger.record(100,'BUY')
        self.assertEqual(ledger.summary()['sebi'],0.01)
        self.assertEqual(ledger.summary()['ipft'],0.01)
        self.assertEqual(ledger.summary()['exchange'],0.30)
        self.assertEqual(ledger.summary()['stamp_duty'],0.30)

    def test_each_session_starts_with_an_empty_ledger(self):
        tariff = IntradayCharges()
        ledger = IntradayChargeLedger(tariff)
        ledger.record(20000,'BUY')
        self.assertEqual(IntradayChargeLedger(tariff).summary()['total'],0)

    def test_invalid_order_does_not_change_ledger(self):
        ledger = IntradayChargeLedger(IntradayCharges())
        ledger.record(100,'BUY')
        before = ledger.summary()
        with self.assertRaises(ValueError):
            ledger.record(-1,'SELL')
        self.assertEqual(ledger.summary(),before)
        for qty in (0,True,1.5):
            with self.subTest(qty=qty), self.assertRaises(ValueError):
                ledger.tariff.estimate_round_trip(qty,200,215)

    def test_actual_equity_statement_reconciles_without_rounding_up(self):
        bill = reconcile_statement('-1220.98',
             {'brokerage':'-4.45','exchange':'-0.06','igst':'-0.81',
              'stt':'-3.68','sebi':'0','stamp_duty':'-0.19','ipft':'0'},'-1230.17')
        self.assertTrue(bill['matches'])
        self.assertEqual(bill['total_charges'],9.19)
        self.assertEqual(bill['difference'],0)

    def test_actual_fo_statement_is_reconciled_separately(self):
        bill = reconcile_statement('-10700',
             {'brokerage':'-20','exchange':'-7.29','igst':'-5.06',
              'stt':'-84.24','sebi':'-0.42','stamp_duty':'0','ipft':'-0.42'},'-10817.43')
        self.assertTrue(bill['matches'])
        self.assertEqual(bill['total_charges'],117.43)
        combined = reconcile_statement('-11920.98',
             {'brokerage':'-24.45','exchange':'-7.35','igst':'-5.87',
              'stt':'-87.92','sebi':'-0.42','stamp_duty':'-0.19','ipft':'-0.42'},'-12047.60')
        self.assertTrue(combined['matches'])
        self.assertEqual(combined['total_charges'],126.62)

    def test_reconciliation_detects_mismatch_and_bad_values(self):
        bill = reconcile_statement('-100',{'brokerage':'-5'},'-104.99')
        self.assertFalse(bill['matches'])
        self.assertEqual(bill['difference'],0.01)
        for value in (float('nan'),True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                reconcile_statement(value,{'brokerage':'-5'},'-105')
        with self.assertRaises(ValueError):
            reconcile_statement('-100',{'brokerage':'5'},'-105')


if __name__=='__main__':
    unittest.main()
