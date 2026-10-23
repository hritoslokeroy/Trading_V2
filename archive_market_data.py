"""Accumulate immutable completed NSE minute-day snapshots; no broker access."""
import argparse
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from data_loader import prepare_intraday_frame
from prepare_training_data import COLUMNS, select_sessions


IDENTITY = ['symbol','interval','source']


def normalize(raw, symbol, source, now):
    raw = raw.copy()
    raw.columns = raw.columns.str.lower()
    if raw.columns.has_duplicates or not set(COLUMNS).issubset(raw.columns):
        raise ValueError('Expected unique timestamp,open,high,low,close,volume columns')
    if 'symbol' in raw and not raw.symbol.eq(symbol).all():
        raise ValueError('Input contains another symbol; never merge different stocks')
    if 'interval' in raw and not raw.interval.eq('1minute').all():
        raise ValueError('Archive accepts one-minute data only')
    stamps = pd.DatetimeIndex(pd.to_datetime(raw.timestamp,format='mixed',errors='raise'))
    stamps = stamps.tz_localize('Asia/Kolkata') if stamps.tz is None else stamps.tz_convert('Asia/Kolkata')
    if stamps.hasnans or stamps.has_duplicates:
        raise ValueError('Missing/duplicate source timestamps')
    if (stamps.second != 0).any() or (stamps.microsecond != 0).any() or (stamps.nanosecond != 0).any():
        raise ValueError('Expected minute-aligned timestamps')
    raw['timestamp'] = stamps
    raw = raw.sort_values('timestamp').set_index('timestamp')
    today = pd.Timestamp(now).tz_convert('Asia/Kolkata').date()
    raw = raw[(raw.index.date < today) & (raw.index.dayofweek < 5)].between_time('09:15','15:29')
    if raw.empty:
        return pd.DataFrame(columns=COLUMNS[1:]+IDENTITY,
                            index=pd.DatetimeIndex([],tz='Asia/Kolkata',name='timestamp'))
    for _, day in raw.groupby(raw.index.date):
        if len(day) > 1 and not (day.index.to_series().diff() == pd.Timedelta(minutes=1)).any():
            raise ValueError('No consecutive one-minute bars; do not import coarser intervals')
    validated = prepare_intraday_frame(raw.reset_index().assign(
        timestamp=lambda frame:frame.timestamp.astype(str)))
    result = validated.loc[:,COLUMNS[1:]].copy()
    result['symbol'], result['interval'] = symbol, '1minute'
    result['source'] = raw['source'] if 'source' in raw else source
    if result.source.isna().any():
        raise ValueError('Archive source metadata is missing')
    return result


def read_archive(path, symbol, now):
    raw = pd.read_csv(path)
    if not set(IDENTITY).issubset(raw.columns):
        raise ValueError('Existing file is not a self-identifying archive; import it with --csv into a new archive path')
    normalized = normalize(raw,symbol,'archive',now)
    if len(normalized) != len(raw):
        raise ValueError('Saved archive has rows outside completed regular sessions; check the clock. No history was removed.')
    return normalized


def status(frame, target=20):
    if frame.empty:
        return {'stored_days':0,'eligible_sessions':0,'rows':0,'target_sessions':target,'ready_for_next_development_run':False}
    # All archive days are already completed; use a clock after its last date
    # to count stored data without referring to today's public download window.
    clock = frame.index.max().normalize()+pd.Timedelta(days=1)
    sessions, decisions = select_sessions(frame.reset_index(),now=clock)
    return {'stored_days':len(set(frame.index.date)),'eligible_sessions':len(sessions),
        'rows':len(frame),'first':str(frame.index.min()),'last':str(frame.index.max()),
        'target_sessions':target,'ready_for_next_development_run':len(sessions)>=target,
        'selection':decisions}


def collect(raw, archive, symbol, *, now=None, source='local_csv', target=20):
    path = Path(archive)
    now = pd.Timestamp.now(tz='Asia/Kolkata') if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError('Collection clock needs a timezone')
    symbol = symbol.strip().upper()
    if not symbol:
        raise ValueError('Symbol must be nonempty')
    incoming = normalize(raw,symbol,source,now)
    existing = read_archive(path,symbol,now) if path.exists() else incoming.iloc[:0].copy()
    overlap = existing.index.intersection(incoming.index)
    revised = 0
    if len(overlap):
        old = existing.loc[overlap,COLUMNS[1:]].to_numpy(dtype=float)
        new = incoming.loc[overlap,COLUMNS[1:]].to_numpy(dtype=float)
        same = np.isclose(old,new,rtol=1e-10,atol=1e-8)
        same[:,-1] = old[:,-1] == new[:,-1]  # Actual reported volume must match.
        revised = int((~same.all(axis=1)).sum())
    # Existing dates are immutable, even if a source later adds/revises candles.
    # Prepared datasets/models therefore remain reproducible. New archive paths
    # can be used to research corrections without silently changing history.
    known_dates = set(existing.index.date)
    additions = incoming[~np.isin(incoming.index.date,list(known_dates))]
    if not len(additions):
        merged = existing.copy()
    elif not len(existing):
        merged = additions.copy()
    else:
        merged = pd.concat([existing,additions]).sort_index()
    if merged.empty:
        raise ValueError('No completed regular-session candles were available')
    if len(additions):
        path.parent.mkdir(parents=True,exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
        os.close(descriptor)
        try:
            merged.reset_index().assign(timestamp=lambda frame:frame.timestamp.astype(str)).to_csv(temporary,index=False)
            os.replace(temporary,path)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
    report = status(merged,target)
    report.update(new_days=len(set(additions.index.date)),new_rows=len(additions),
                  source_revisions_ignored=revised)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbol',default='RELIANCE')
    parser.add_argument('--archive',required=True,type=Path)
    parser.add_argument('--csv',type=Path,help='Import an existing one-stock minute CSV instead of downloading')
    parser.add_argument('--status',action='store_true',help='Inspect saved archive only; no network')
    args = parser.parse_args(argv)
    try:
        symbol = args.symbol.strip().upper()
        now = pd.Timestamp.now(tz='Asia/Kolkata')
        if args.status:
            report = status(read_archive(args.archive,symbol,now))
        else:
            if args.csv:
                raw = pd.read_csv(args.csv)
                source = 'local_csv'
            else:
                try:
                    import yfinance as yf
                except ImportError:
                    raise ValueError('Install optional yfinance with this project Python first') from None
                downloaded = yf.Ticker(symbol+'.NS').history(period='7d',interval='1m',auto_adjust=False,actions=False)
                if downloaded.empty:
                    raise ValueError('Public source returned no candles; saved archive was not changed')
                if downloaded.index.tz is None:
                    raise ValueError('Public source timestamps have no timezone')
                raw = downloaded.rename_axis('timestamp').reset_index()
                source = 'Yahoo Finance via yfinance'
            report = collect(raw,args.archive,symbol,now=now,source=source)
            print(f"Added {report['new_days']} completed days / {report['new_rows']} candles.")
            if report['source_revisions_ignored']:
                print(f"Source revised {report['source_revisions_ignored']} overlapping candles; stored day snapshots were preserved.")
        print(f"Archive: {args.archive}")
        print(f"Stored days: {report['stored_days']}; usable continuous sessions: {report['eligible_sessions']}")
        print(f"Next development data target: {report['target_sessions']} usable sessions (not a profitability guarantee).")
        print('Data target reached.' if report['ready_for_next_development_run'] else 'Keep collecting; do not repeat training on the same small history.')
        print('No Groww credentials, model learning or broker orders were used.')
    except (ValueError,OSError,KeyError) as error:
        parser.exit(2,f'Archive operation stopped: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
