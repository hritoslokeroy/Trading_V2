"""Configurable estimates for NSE cash intraday charges, in INR.

Rates are indicative, not a broker contract-note reconciliation.
Components retain Decimal precision until session totals are rounded to paise.
Broker contract-note aggregation can still differ; actual charges take precedence.
No delivery DP charges, income tax, or subscription charges are included.
"""
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from math import isfinite


@dataclass(frozen=True)
class IntradayCharges:
    # Fractions of turnover: 0.00025 means 0.025%.
    brokerage_rate: float = 0.001
    brokerage_min_inr: float = 5.0
    brokerage_max_inr: float = 20.0
    small_order_brokerage_cap_rate: float = 0.025
    stt_sell_rate: float = 0.00025
    stamp_buy_rate: float = 0.00003
    exchange_rate: float = 0.0000297
    sebi_rate: float = 0.000001
    ipft_rate: float = 0.000001
    gst_rate: float = 0.18

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            if name.endswith('_rate') and value >= 1:
                raise ValueError(f"{name} must be a fraction below 1")
        if self.brokerage_min_inr > self.brokerage_max_inr:
            raise ValueError("Minimum brokerage cannot exceed maximum brokerage")

    @classmethod
    def from_settings(cls, settings):
        if not isinstance(settings, dict):
            raise ValueError('settings must be an object')
        config = settings.get('charges', {})
        if not isinstance(config, dict):
            raise ValueError('charges must be an object')
        if config.get('aggregation', 'session') != 'session':
            raise ValueError('Only session charge aggregation is supported')
        if config.get('rounding', 'paise_half_up_per_component') != 'paise_half_up_per_component':
            raise ValueError('Only paise_half_up_per_component rounding is supported')
        rates = config.get('rates', {})
        if not isinstance(rates, dict):
            raise ValueError('charges.rates must be an object')
        unknown = set(rates) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError('Unknown charge rates: ' + ', '.join(sorted(unknown)))
        return cls(**rates)

    def _raw(self, turnover, side):
        side = side.upper() if isinstance(side, str) else ''
        if side not in ('BUY', 'SELL'):
            raise ValueError('side must be BUY or SELL')
        if isinstance(turnover, bool):
            raise ValueError('turnover must be finite and positive')
        try:
            value = Decimal(str(turnover))
        except (InvalidOperation, ValueError):
            raise ValueError("turnover must be finite and positive") from None
        if not value.is_finite() or value <= 0:
            raise ValueError('turnover must be finite and positive')
        r = {k: Decimal(str(v)) for k, v in asdict(self).items()}
        brokerage = min(r['brokerage_max_inr'],
                        max(r['brokerage_min_inr'], value * r['brokerage_rate']),
                        value * r['small_order_brokerage_cap_rate'])
        # Brokerage is an executed-order charge, including its paise rounding.
        brokerage = brokerage.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        exchange = value * r['exchange_rate']
        sebi, ipft = value * r['sebi_rate'], value * r['ipft_rate']
        parts = {'brokerage': brokerage,
                 'stt': value * r['stt_sell_rate'] if side == 'SELL' else Decimal(0),
                 'stamp_duty': value * r['stamp_buy_rate'] if side == 'BUY' else Decimal(0),
                 'exchange': exchange, 'sebi': sebi, 'ipft': ipft,
                 'gst': (brokerage + exchange + sebi + ipft) * r['gst_rate']}
        return parts

    def estimate(self, turnover, side):
        """Estimate one executed order. Brokerage is calculated once per order."""
        return _rounded(self._raw(turnover, side))

    def estimate_orders(self, orders):
        """Aggregate executed orders before rounding each session charge head.

        Each item is (turnover, side) for ONE executed order, not a partial fill.
        Keep different segments and sessions in separate ledgers.
        """
        ledger = IntradayChargeLedger(self)
        for turnover, side in orders:
            ledger.record(turnover, side)
        return ledger.summary()

    def estimate_round_trip(self, quantity, buy_price, sell_price):
        """Two executed orders; calculation only, independent of live position sizing."""
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 1:
            raise ValueError('quantity must be a positive integer')
        buy = _decimal(buy_price, 'buy_price', positive=True) * quantity
        sell = _decimal(sell_price, 'sell_price', positive=True) * quantity
        result = self.estimate_orders(((buy, 'BUY'), (sell, 'SELL')))
        gross = sell - buy
        result['gross_pnl'] = float(gross)
        result['net_pnl'] = float(gross - Decimal(str(result['total'])))
        return result


_COMPONENTS = ('brokerage', 'stt', 'stamp_duty', 'exchange', 'sebi', 'ipft', 'gst')
_PAISE = Decimal('0.01')


def _decimal(value, name, *, positive=False):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a finite number')
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f'{name} must be a finite number') from None
    if not result.is_finite() or (positive and result <= 0):
        raise ValueError(f'{name} must be finite' + (' and positive' if positive else ''))
    return result


def _rounded(parts):
    amounts = {k: v.quantize(_PAISE, rounding=ROUND_HALF_UP) for k, v in parts.items()}
    result = {k: float(v) for k, v in amounts.items()}
    result['total'] = float(sum(amounts.values(), Decimal(0)))
    return result


class IntradayChargeLedger:
    """Session-local charges; quote is read-only, record is for confirmed fills.

    Marginal debits make the sum of execution charges equal the rounded session
    bill. A tiny head can debit zero on one order and one paise on a later order.
    STT/stamp duty keep paise; no unverified whole-rupee ceiling is imposed.
    """
    def __init__(self, tariff):
        self.tariff = tariff
        self._parts = {key: Decimal(0) for key in _COMPONENTS}

    def summary(self):
        return _rounded(self._parts)

    def quote(self, turnover, side):
        raw = self.tariff._raw(turnover, side)
        before = self.summary()
        after = _rounded({k: self._parts[k] + raw[k] for k in _COMPONENTS})
        return {k: float(Decimal(str(after[k])) - Decimal(str(before[k]))) for k in after}

    def record(self, turnover, side):
        raw = self.tariff._raw(turnover, side)
        debit = self.quote(turnover, side)
        for key in _COMPONENTS:
            self._parts[key] += raw[key]
        return debit


def reconcile_statement(obligation, charges, net_amount):
    """Reconcile ACTUAL signed statement amounts for one segment/session.

    Supply deductions as negative amounts, as on the user's statement. IGST is
    one GST amount: do not also add CGST/SGST on an IGST-only statement. This
    checks settlement arithmetic; it does not infer trade P&L or tariff rates.
    """
    before = _decimal(obligation, 'obligation')
    actual_net = _decimal(net_amount, 'net_amount')
    if not isinstance(charges, dict) or not charges:
        raise ValueError('charges must contain the actual statement charge heads')
    parts = {key: _decimal(value, key) for key, value in charges.items()}
    if any(v > 0 for v in parts.values()):
        raise ValueError('Statement deductions must be negative or zero')
    expected = before + sum(parts.values(), Decimal(0))
    return {'source': 'broker_statement', 'total_charges': float(-sum(parts.values(), Decimal(0))),
            'expected_net_amount': float(expected), 'actual_net_amount': float(actual_net),
            'difference': float(actual_net - expected), 'matches': actual_net == expected}


def main(argv=None):
    """Offline charge calculator; never authenticates or submits an order."""
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description='Estimate NSE cash intraday round-trip charges in INR')
    parser.add_argument('--quantity', type=int, required=True)
    parser.add_argument('--buy', required=True, help='buy price per share in INR')
    parser.add_argument('--sell', required=True, help='sell price per share in INR')
    parser.add_argument('--settings', type=Path, default=Path(__file__).with_name('settings.json'))
    args = parser.parse_args(argv)
    try:
        tariff = IntradayCharges.from_settings(json.loads(args.settings.read_text()))
        bill = tariff.estimate_round_trip(args.quantity, args.buy, args.sell)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps({'source': 'estimate', 'aggregation': 'session',
                      'rounding': 'paise_half_up_per_component', 'charges_and_pnl': bill},
                     indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
