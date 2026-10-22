"""Download 1-minute NSE cash candles from Groww into the CSV format the trainer reads.

Run from the repository root (credentials come from the environment variables named in
config/settings.json; nothing is printed or stored):

    python scripts/download_history.py --symbol RELIANCE --start 2026-07-01 --end 2026-10-21

Output: data/historical/<SYMBOL>_1m.csv with columns timestamp,open,high,low,close,volume.
Read-only: this script never places orders. Groww limits 1-minute requests to 30 days, so the
range is fetched in chunks. Check the result with --verify before training.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

IST = "Asia/Kolkata"
CHUNK_DAYS = 28  # Groww allows 30 days per 1-minute request
OPEN, LAST = "09:15", "15:29"


def candles_to_frame(candles: list) -> pd.DataFrame:
    """Groww candle rows: [timestamp, open, high, low, close, volume, open_interest]."""
    rows = []
    for row in candles:
        stamp = row[0]
        if isinstance(stamp, (int, float)):
            stamp = pd.Timestamp(stamp, unit="s", tz="UTC").tz_convert(IST).tz_localize(None)
        else:
            stamp = pd.Timestamp(str(stamp))
            if stamp.tzinfo is not None:
                stamp = stamp.tz_convert(IST).tz_localize(None)
        rows.append((stamp, *row[1:6]))
    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    for name in ("open", "high", "low", "close", "volume"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep regular-session candles, drop duplicates and invalid rows, sort by time."""
    frame = frame.drop_duplicates("timestamp").sort_values("timestamp")
    frame = frame.dropna()
    stamps = pd.DatetimeIndex(frame["timestamp"])
    frame = frame[(stamps.strftime("%H:%M") >= OPEN) & (stamps.strftime("%H:%M") <= LAST)]
    frame = frame[(frame[["open", "high", "low", "close"]] > 0).all(axis=1) & (frame.volume >= 0)]
    bad = (frame.high < frame[["open", "low", "close"]].max(axis=1)) | (
        frame.low > frame[["open", "high", "close"]].min(axis=1)
    )
    return frame[~bad].reset_index(drop=True)


def fetch_history(client, symbol: str, start: date, end: date, *, pause: float = 0.5) -> pd.DataFrame:
    """Fetch [start, end] inclusive in chunks through an authenticated GrowwAPI client."""
    interval = getattr(client, "CANDLE_INTERVAL_MIN_1", None)
    if interval is None:
        raise RuntimeError("This growwapi version has no CANDLE_INTERVAL_MIN_1; upgrade growwapi")

    parts = []
    cursor = start
    while cursor <= end:
        stop = min(cursor + timedelta(days=CHUNK_DAYS - 1), end)
        response = client.get_historical_candles(
            exchange=client.EXCHANGE_NSE,
            segment=client.SEGMENT_CASH,
            groww_symbol=f"NSE-{symbol}",
            start_time=f"{cursor:%Y-%m-%d} 09:15:00",
            end_time=f"{stop:%Y-%m-%d} 15:30:00",
            candle_interval=interval,
        )
        candles = response.get("candles") or []
        print(f"{cursor} -> {stop}: {len(candles)} candles")
        if candles:
            parts.append(candles_to_frame(candles))
        cursor = stop + timedelta(days=1)
        time.sleep(pause)

    if not parts:
        raise RuntimeError("No candles returned; check the symbol, dates and API access")
    return clean(pd.concat(parts, ignore_index=True))


def verify(frame: pd.DataFrame) -> None:
    stamps = pd.DatetimeIndex(frame["timestamp"])
    days = pd.Series(stamps.date).value_counts().sort_index()
    short = days[days < 300]
    print(f"{len(days)} trading days, {len(frame)} candles, {stamps.min()} -> {stamps.max()}")
    print(f"median candles/day: {int(days.median())} (full session is 375)")
    if len(short):
        print(f"WARNING: {len(short)} days have under 300 candles (gaps or half-days):")
        print(short.to_string())
    print(f"price range: {frame.close.min():.2f} - {frame.close.max():.2f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", required=True, help="NSE trading symbol, e.g. RELIANCE")
    parser.add_argument("--start", type=date.fromisoformat, help="first date, YYYY-MM-DD")
    parser.add_argument("--end", type=date.fromisoformat, help="last date, YYYY-MM-DD")
    parser.add_argument("--settings", type=Path, default=REPOSITORY_ROOT / "config" / "settings.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", action="store_true", help="only inspect an existing output CSV")
    args = parser.parse_args(argv)

    symbol = args.symbol.strip().upper()
    out = args.out or REPOSITORY_ROOT / "data" / "historical" / f"{symbol}_1m.csv"

    if args.verify:
        verify(pd.read_csv(out, parse_dates=["timestamp"]))
        return 0
    if not args.start or not args.end or args.start > args.end:
        parser.error("--start and --end are required, with start <= end")

    import pyotp
    from growwapi import GrowwAPI
    from src.utils.data_loader import load_credentials

    credentials = load_credentials(args.settings, required=("api_key", "totp_secret"))
    totp = pyotp.TOTP(credentials.totp_secret, digits=6).now()
    token = GrowwAPI.get_access_token(api_key=credentials.api_key, totp=totp)
    client = GrowwAPI(token)

    frame = fetch_history(client, symbol, args.start, args.end)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.assign(timestamp=pd.to_datetime(frame["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")).to_csv(
        out, index=False
    )
    print(f"Saved {len(frame)} candles to {out}")
    verify(frame)
    return 0


if __name__ == "__main__":
    sys.exit(main())
