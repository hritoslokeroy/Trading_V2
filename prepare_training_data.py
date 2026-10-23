"""Prepare recent public candles or an existing CSV for multi-session research.

Select a contiguous section per completed day, without inventing missing bars.
Reserve the latest eligible day in a separate practice file. No broker API calls.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from data_loader import prepare_intraday_frame


COLUMNS = ['timestamp','open','high','low','close','volume']


def select_sessions(raw, *, now=None, minimum_bars=150):
    raw = raw.copy()
    raw.columns = raw.columns.str.lower()
    if not set(COLUMNS).issubset(raw.columns):
        raise ValueError('Data must contain timestamp,open,high,low,close,volume')
    stamps = pd.DatetimeIndex(pd.to_datetime(raw.timestamp,format='mixed',errors='raise'))
    if stamps.tz is None:
        stamps = stamps.tz_localize('Asia/Kolkata')
    else:
        stamps = stamps.tz_convert('Asia/Kolkata')
    raw['timestamp'] = stamps
    raw = raw.sort_values('timestamp').set_index('timestamp')
    if raw.index.has_duplicates or raw.index.hasnans:
        raise ValueError('Duplicate or missing timestamps; resolve source data rather than guessing')
    now = pd.Timestamp.now(tz='Asia/Kolkata') if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError('Preparation clock needs a timezone')
    today = now.tz_convert('Asia/Kolkata').date()
    selected, decisions = [], []
    for day, frame in raw.groupby(raw.index.date):
        if day >= today or pd.Timestamp(day).dayofweek >= 5:
            decisions.append({'date':str(day),'selected':False,'reason':'current/future day or weekend'})
            continue
        frame = frame.between_time('09:15','15:29')
        if frame.empty:
            decisions.append({'date':str(day),'selected':False,'reason':'no regular-session candles'})
            continue
        # Invalid source rows are rejected, not silently repaired or filled.
        validated = prepare_intraday_frame(frame.reset_index().assign(
            timestamp=lambda data:data.timestamp.astype(str)))
        groups = validated.index.to_series().diff().ne(pd.Timedelta(minutes=1)).cumsum()
        longest = max((part for _,part in validated.groupby(groups)),key=len)
        if len(longest) < minimum_bars:
            decisions.append({'date':str(day),'selected':False,'reason':'continuous section too short',
                              'longest_bars':len(longest)})
            continue
        # Compute indicators later, after selecting each actual continuous section.
        selected.append(longest.loc[:,COLUMNS[1:]].copy())
        decisions.append({'date':str(day),'selected':True,'source_bars':len(frame),
                          'selected_bars':len(longest),'first':str(longest.index[0]),
                          'last':str(longest.index[-1]),'partial_session':len(longest)!=375})
    return selected, decisions


def prepare(raw, output_dir, *, now=None, source='local_csv', symbol=None):
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError('Choose a new data directory to preserve earlier files')
    sessions, decisions = select_sessions(raw,now=now)
    if len(sessions) < 5:
        raise ValueError('Need five completed eligible sessions: four for chronological model groups and one for practice')
    root.mkdir(parents=True,exist_ok=False)
    training = pd.concat(sessions[:-1])
    practice = sessions[-1]
    for name, frame in [('multi_sessions.csv',training),('practice_multi.csv',practice)]:
        frame.reset_index().assign(timestamp=lambda data:data.timestamp.dt.strftime('%Y-%m-%d %H:%M:%S')).to_csv(root/name,index=False)
    manifest = {'source':source,'symbol':symbol,'symbol_scope':'one configured NSE CASH stock',
        'training_dates':[str(frame.index[0].date()) for frame in sessions[:-1]],
        'practice_date':str(practice.index[0].date()),'selection':decisions,
        'notes':'No gap filling. Partial-session indicators begin at the chosen section. Previously inspected days are development data, not an untouched final test.'}
    (root/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path,help='Existing multi-day one-minute OHLCV file; no download')
    parser.add_argument('--symbol',default='RELIANCE')
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists():
            raise FileExistsError('Choose a new data directory')
        if args.csv:
            raw = pd.read_csv(args.csv)
            source = 'local_csv'
        else:
            try:
                import yfinance as yf
            except ImportError:
                raise ValueError('Install optional downloader: .\\.venv\\Scripts\\python.exe -m pip install yfinance') from None
            data = yf.Ticker(args.symbol+'.NS').history(period='7d',interval='1m',auto_adjust=False,actions=False)
            if data.empty:
                raise ValueError('Public source returned no candles; do not fabricate training data')
            if data.index.tz is None:
                raise ValueError('Public source timestamps have no timezone')
            raw = data.rename_axis('timestamp').reset_index()
            source = 'Yahoo Finance via yfinance; recent one-minute data, availability not guaranteed'
        manifest = prepare(raw,args.output_dir,source=source,symbol=args.symbol)
    except (ValueError,OSError,KeyError) as error:
        parser.exit(2,f'Data preparation stopped: {error}\n')
    print('Training dates:',', '.join(manifest['training_dates']))
    print('Reserved practice date:',manifest['practice_date'])
    print('Saved:',args.output_dir/'multi_sessions.csv')
    print('Saved:',args.output_dir/'practice_multi.csv')
    print('Source/section details:',args.output_dir/'data_manifest.json')
    print('No Groww credentials or orders were used.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
