"""Configurable estimates for NSE cash intraday charges, in INR.

Rates are indicative, not a broker contract-note reconciliation. In particular,
STT rounding and turnover aggregation can differ from per-order estimates.
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
        config = settings.get('charges', {})
        if not isinstance(config, dict):
            raise ValueError('charges must be an object')
        rates = config.get('rates', {})
        if not isinstance(rates, dict):
            raise ValueError('charges.rates must be an object')
        unknown = set(rates) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError('Unknown charge rates: ' + ', '.join(sorted(unknown)))
        return cls(**rates)

    def estimate(self, turnover, side):
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
        exchange = value * r['exchange_rate']
        sebi, ipft = value * r['sebi_rate'], value * r['ipft_rate']
        parts = {'brokerage': brokerage,
                 'stt': value * r['stt_sell_rate'] if side == 'SELL' else Decimal(0),
                 'stamp_duty': value * r['stamp_buy_rate'] if side == 'BUY' else Decimal(0),
                 'exchange': exchange, 'sebi': sebi, 'ipft': ipft,
                 'gst': (brokerage + exchange + sebi + ipft) * r['gst_rate']}
        result = {k: float(v.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)) for k, v in parts.items()}
        result['total'] = float(sum(Decimal(str(v)) for v in result.values()))
        return result
