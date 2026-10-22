"""Read-only market data capability for paper trading. No order-routing API."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from data_loader import load_credentials
from download_history import candles_to_frame, fetch_history

IST = ZoneInfo('Asia/Kolkata')


class PaperDataError(RuntimeError):
    """Sanitized market-data failure; SDK responses/credentials are not exposed."""


class ReadOnlyGrowwCandles:
    """Expose only completed NSE cash candles, never account or order methods."""
    def __init__(self, client, symbol):
        self.__client = client
        self.symbol = symbol

    @classmethod
    def connect(cls, settings_path, symbol, *, client_factory=None, token_provider=None):
        try:
            from growwapi import GrowwAPI
            from groww_auth import token_from_credentials
            credentials = load_credentials(settings_path, required=())
            token = token_from_credentials(credentials, provider=token_provider)
            client = (client_factory or GrowwAPI)(token)
            return cls(client, symbol)
        except Exception as error:
            from groww_auth import GrowwAuthError
            if isinstance(error, GrowwAuthError):
                raise PaperDataError(str(error)) from None
            raise PaperDataError('Groww data authentication failed. Check the configured credential variables and API access; no secrets were logged.') from None

    def history(self, start, end):
        """Read historical market prices, with the same credential boundary."""
        try:
            return fetch_history(self.__client,self.symbol,start,end)
        except Exception:
            raise PaperDataError('Historical data download failed. Check dates, symbol and API data permissions; no credentials were logged.') from None

    def fetch(self, now: datetime):
        if now.tzinfo is None:
            raise ValueError('Market clock must be timezone aware')
        now = now.astimezone(IST)
        try:
            response = self.__client.get_historical_candles(
                exchange='NSE', segment='CASH', groww_symbol=f'NSE-{self.symbol}',
                start_time=f'{now:%Y-%m-%d} 09:15:00', end_time=f'{now:%Y-%m-%d %H:%M:%S}',
                candle_interval='1minute', timeout=15)
            if not isinstance(response, dict) or not isinstance(response.get('candles'), list):
                raise ValueError('Unknown candle response schema')
            if not response['candles']:
                return pd.DataFrame(columns=['timestamp','open','high','low','close','volume'])
            frame = candles_to_frame(response['candles'])
            stamps = pd.DatetimeIndex(frame.timestamp)
            stamps = stamps.tz_localize(IST) if stamps.tz is None else stamps.tz_convert(IST)
            if stamps.hasnans or stamps.has_duplicates or not stamps.is_monotonic_increasing:
                raise ValueError('Invalid market timestamps')
            frame['timestamp'] = stamps
            frame = frame[(stamps.date == now.date()) & (stamps < pd.Timestamp(now).floor('min'))]
            # Use only regular-session, COMPLETED candles; never the forming minute.
            times = pd.DatetimeIndex(frame.timestamp).strftime('%H:%M')
            frame = frame[(times >= '09:15') & (times <= '15:29')].copy()
            frame['timestamp'] = frame.timestamp.astype(str)
            return frame
        except Exception:
            raise PaperDataError('Groww candle data failed validation or could not be fetched; paper session stopped without broker orders.') from None
