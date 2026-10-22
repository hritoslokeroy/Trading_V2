"""Independent tariff arithmetic checks; no claim of current tariff verification."""
import unittest
from charges import IntradayCharges


class ChargeTests(unittest.TestCase):
    def test_buy_breakdown_for_ten_thousand_rupees(self):
        bill = IntradayCharges().estimate(10000, 'BUY')
        self.assertEqual(bill, {'brokerage': 10.0, 'stt': 0.0, 'stamp_duty': 0.3,
                                'exchange': 0.3, 'sebi': 0.01, 'ipft': 0.01,
                                'gst': 1.86, 'total': 12.48})

    def test_sell_breakdown_for_ten_thousand_rupees(self):
        bill = IntradayCharges().estimate(10000, 'SELL')
        self.assertEqual(bill['stamp_duty'], 0)
        self.assertEqual(bill['stt'], 2.5)
        self.assertEqual(bill['total'], 14.68)

    def test_small_order_brokerage_and_maximum(self):
        fees = IntradayCharges()
        self.assertEqual(fees.estimate(100, 'BUY')['brokerage'], 2.5)
        self.assertEqual(fees.estimate(1000, 'BUY')['brokerage'], 5)
        self.assertEqual(fees.estimate(100000, 'BUY')['brokerage'], 20)

    def test_custom_rate_overrides(self):
        fees = IntradayCharges.from_settings({'charges': {'rates': {'stt_sell_rate': 0.0005}}})
        self.assertEqual(fees.estimate(10000, 'SELL')['stt'], 5)

    def test_bad_rates_fail_instead_of_silently_ignoring(self):
        for rates in ({'stt_sell_rate': float('nan')}, {'gst_rate': -1},
                      {'exchange_rate': True}, {'unknown_rate': 1}):
            with self.subTest(rates=rates), self.assertRaises(ValueError):
                IntradayCharges.from_settings({'charges': {'rates': rates}})

    def test_invalid_orders_rejected(self):
        for value, side in ((0,'BUY'),(-1,'BUY'),(float('inf'),'BUY'),(True,'BUY'),(100,'HOLD')):
            with self.subTest(value=value, side=side), self.assertRaises(ValueError):
                IntradayCharges().estimate(value, side)


if __name__ == '__main__':
    unittest.main()
