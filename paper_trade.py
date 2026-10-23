"""Fake-money paper trading: frozen AI models, CSV replay or read-only Groww candles.

No code in this module imports a live execution adapter or calls broker orders.
Paper fills on next candle open are retrospective estimates once that minute
closes. They are not real-time executable fills and never promote to live trading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, time as clock_time
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN, PPO
from data_loader import load_intraday_csv, load_settings, prepare_intraday_frame
from paper_market_data import PaperDataError, ReadOnlyGrowwCandles
from quant_agent import QuantLSTMBrain
from rl_brain import TradingEnvironment

IST = ZoneInfo('Asia/Kolkata')
DEFAULT_PAPER_SETTINGS = Path(__file__).with_name('paper_settings.json')


def artifact_fingerprint(model_dir):
    """Identify the exact frozen checkpoints used for a paper session."""
    digest = hashlib.sha256()
    for name in ('run_metrics.json','lstm.pt','feature_scaler.npz','rl_policy.zip'):
        digest.update(name.encode())
        with (Path(model_dir)/name).open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):
                digest.update(block)
    return digest.hexdigest()


def paper_settings(path, cash=None):
    settings = load_settings(path)
    if settings.get('execution', {}).get('mode') != 'paper':
        raise ValueError('Paper trading requires execution.mode=paper; never use live settings')
    if settings.get('risk_limits', {}).get('trading_enabled') is not True:
        raise ValueError('Enable the simulated risk gate in paper settings')
    if settings.get('data', {}).get('timezone') != 'Asia/Kolkata':
        raise ValueError('Paper market times require Asia/Kolkata')
    config = settings.get('paper', {})
    if not isinstance(config,dict):
        raise ValueError('paper settings must be an object')
    balance = config.get('initial_cash_inr', 100000) if cash is None else cash
    for name, value in [('initial_cash_inr', balance),
                        ('poll_interval_seconds', config.get('poll_interval_seconds', 10)),
                        ('max_candle_delay_seconds', config.get('max_candle_delay_seconds', 180))]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if config.get('poll_interval_seconds', 10) < 5:
        raise ValueError('Poll interval must be at least 5 seconds')
    return settings, float(balance)


class FrozenPolicy:
    """Load the existing scaler and models; never fit or learn on practice data."""
    def __init__(self, model_dir, symbol):
        root = Path(model_dir)
        self.fingerprint = artifact_fingerprint(root)
        self.metadata = json.loads((root/'run_metrics.json').read_text())
        meta = self.metadata
        if (meta.get('simulator_version') not in (2, 4) or meta.get('symbol') != symbol
                or meta.get('features') != list(QuantLSTMBrain.FEATURES)
                or meta.get('target') != 'next_close_fractional_return'
                or meta.get('algorithm') not in ('PPO','DQN')):
            raise ValueError('Model metadata is incompatible; train a matching model first')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lstm = QuantLSTMBrain().to(self.device).eval()
        self.lstm.load_state_dict(torch.load(root/'lstm.pt', weights_only=True, map_location=self.device))
        loader = PPO if meta['algorithm'] == 'PPO' else DQN
        self.policy = loader.load(root/'rl_policy.zip', device=self.device)
        if self.policy.observation_space.shape != (4,) or self.policy.action_space.n != 3:
            raise ValueError('Policy needs four observations and three actions')
        with np.load(root/'feature_scaler.npz', allow_pickle=False) as scaler:
            self.mean, self.scale = scaler['mean'].copy(), scaler['scale'].copy()
        if (self.mean.shape != (7,) or self.scale.shape != (7,)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all()
                or (self.scale <= 0).any()):
            raise ValueError('Invalid trained feature scaler')

    def score(self, features):
        ready = features.dropna(subset=list(QuantLSTMBrain.FEATURES)).copy()
        length = QuantLSTMBrain.SEQUENCE_LENGTH
        if len(ready) < length:
            return ready.iloc[:0].assign(lstm_score=pd.Series(dtype=float))
        values = (ready.loc[:, list(QuantLSTMBrain.FEATURES)].to_numpy(dtype=float) - self.mean) / self.scale
        if not np.isfinite(values).all() or (np.abs(values) > np.finfo(np.float32).max).any():
            raise ValueError('Paper features are nonfinite or overflow')
        if not (np.diff(ready.index.as_unit('ns').asi8) == 60_000_000_000).all():
            raise ValueError('Paper inference needs contiguous one-minute candles')
        scores = []
        with torch.inference_mode():
            for end in range(length-1, len(ready)):
                inputs = torch.as_tensor(values[end-length+1:end+1], dtype=torch.float32,
                                         device=self.device).unsqueeze(0)
                score = float(self.lstm(inputs).item())
                if not math.isfinite(score):
                    raise ValueError('Nonfinite LSTM paper score')
                scores.append(score)
        result = ready.iloc[length-1:].copy()
        result['lstm_score'] = scores
        return result

    def action(self, observation):
        proposal, _ = self.policy.predict(observation, deterministic=True)
        value = np.asarray(proposal).item()
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value not in (0,1,2):
            raise ValueError('Invalid paper policy action')
        return int(value)


class PaperRun:
    """One isolated simulated session; persistence contains prices and fake money only."""
    def __init__(self, model, settings_path, output_dir, symbol, *, cash=None, source='csv_replay'):
        self.settings, self.cash = paper_settings(settings_path, cash)
        self.settings_path = Path(settings_path)
        if symbol not in self.settings.get('data', {}).get('supported_symbols', []):
            raise ValueError('Choose a symbol enabled in paper settings')
        self.symbol, self.model, self.source = symbol, model, source
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=False)
        self.env = None
        self.curve, self.decisions = [], []
        self.rejections, self.max_drawdown = 0, 0.0
        self.peak = self.cash
        self.observation = None
        self.status = 'created'
        self.previous = None
        self.started_at = None
        self.last_info = {}

    def initialize(self, scored, *, start_latest=False):
        environment = TradingEnvironment(scored, symbol=self.symbol, settings_path=self.settings_path,
                                         initial_cash=self.cash)
        profile = environment.execution_profile()
        trained_profile = self.model.metadata.get('execution_profile', {})
        if environment.simulator_version == 4 or self.model.metadata.get('simulator_version') == 4:
            if (self.model.metadata.get('simulator_version') != environment.simulator_version
                    or any(trained_profile.get(key) != profile[key] for key in
                           ('position_sizing','leverage','max_order_value_inr','max_position_value_inr','observation_schema'))):
                raise ValueError('Model sizing/leverage profile differs; retrain with matching simulation settings')
        self.env = environment
        self.observation, _ = self.env.reset()
        if start_latest:
            self.env.current_step = len(self.env.df)-1
            self.observation = self.env._get_obs()
        self.started_at = self.env.df.iloc[self.env.current_step].timestamp
        self._point(self.env._info())

    def _point(self, info):
        self.peak = max(self.peak, info['equity'])
        self.max_drawdown = max(self.max_drawdown, self.peak-info['equity'])
        stamp = self.env.df.iloc[self.env.current_step].timestamp.isoformat()
        self.curve.append({'timestamp':stamp, 'equity':info['equity'], 'cash':info['cash'],
                           'net_pnl':info['daily_pnl'], 'charges':info['total_charges'],
                           'quantity':info.get('quantity',0),
                           'margin_reserved_inr':info.get('margin_reserved_inr',0),
                           'position_value_inr':info.get('position_value_inr',0)})

    def advance(self, *, streaming=False):
        action = self.model.action(self.observation)
        decision_time = self.env.df.iloc[self.env.current_step].timestamp.isoformat()
        self.observation, _, done, _, info = self.env.step(action, end_of_data=not streaming)
        self.last_info = info
        self.decisions.append({'candle_timestamp':decision_time,'action':action,
                               'rejected':info['order_rejected']})
        self.rejections += int(info['order_rejected'])
        self._point(info)
        return done

    def report(self):
        fills = self.env.executions if self.env else []
        info = self.env._info() if self.env else {'cash':self.cash, 'equity':self.cash,
                 'daily_pnl':0, 'total_charges':0, 'charge_breakdown':{}}
        training_end = self.model.metadata.get('boundaries', {}).get('rl_end')
        overlap = None
        if training_end and self.env:
            overlap = self.started_at <= pd.Timestamp(training_end)
        return {'mode':'paper', 'uses_real_money':False, 'learning_during_session':False,
                'model_fingerprint':getattr(self.model,'fingerprint',None),
                'algorithm':self.model.metadata.get('algorithm'),
                'source':self.source, 'status':self.status, 'symbol':self.symbol,
                'simulator_version':self.env.simulator_version if self.env else None,
                'execution_profile':self.env.execution_profile() if self.env else None,
                'position_sizing':self.settings.get('simulation',{}).get('position_sizing','one_share'),
                'quantity':self.env.quantity if self.env else 0,
                'max_executed_quantity':max((fill['quantity'] for fill in fills),default=0),
                'practice_started_at':self.started_at.isoformat() if self.started_at is not None else None,
                'timezone':'Asia/Kolkata', 'initial_cash_inr':self.cash, 'final_cash_inr':info['cash'],
                'equity_inr':info['equity'], 'net_pnl_inr':info['daily_pnl'],
                'estimated_charges_inr':info['total_charges'], 'charge_breakdown':info['charge_breakdown'],
                'max_drawdown_inr':self.max_drawdown, 'decisions':len(self.decisions),
                'action_counts':{name:sum(d['action']==action for d in self.decisions)
                                 for action,name in enumerate(('hold','buy','sell'))},
                'rejected_entries':self.rejections, 'executed_orders':len(fills),
                'forced_exits':sum(int(f['forced']) for f in fills),
                'overlaps_training':None if overlap is None else bool(overlap),
                'fills_are_estimates':True,
                'fill_model':'next candle open, observed after minute completion; last-known-close exits on stop/error',
                'auto_switch_to_live':False}

    def save(self):
        report = self.report()
        # Only serialize local report fields, never SDK responses or credentials.
        (self.output/'paper_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        (self.output/'executions.json').write_text(json.dumps(self.env.executions if self.env else [],indent=2,allow_nan=False)+'\n')
        (self.output/'decisions.json').write_text(json.dumps(self.decisions,indent=2,allow_nan=False)+'\n')
        pd.DataFrame(self.curve).to_csv(self.output/'equity_curve.csv',index=False)
        return report

    def finish(self, status):
        self.status = status
        if self.env:
            self._point(self.env.finish(status))
        return self.save()

    def completed_status(self, default):
        if self.last_info.get('reason') == 'risk_cutoff_breach':
            cutoff = clock_time.fromisoformat(self.settings['risk_limits']['hard_cutoff_time'])
            return ('market_cutoff' if self.env.df.iloc[self.env.current_step].timestamp.time() >= cutoff
                    else 'risk_halt')
        return default

    def replay(self, features):
        try:
            scored = self.model.score(features)
            if len(scored) < 2:
                raise ValueError('Need at least 61 indicator-ready candles for paper replay')
            self.initialize(scored)
            while not self.env.done:
                self.advance()
            return self.finish(self.completed_status('replay_complete'))
        except KeyboardInterrupt:
            return self.finish('interrupted')
        except Exception:
            self.finish('failed')
            raise

    def poll(self, provider, *, clock=lambda:datetime.now(IST), sleep=time.sleep, on_update=None):
        """Follow new completed candles only; startup history warms up, never trades."""
        config = self.settings.get('paper', {})
        interval = config.get('poll_interval_seconds',10)
        max_delay = config.get('max_candle_delay_seconds',180)
        session_date = clock().astimezone(IST).date()
        try:
            while True:
                now = clock()
                if now.tzinfo is None:
                    raise ValueError('Paper clock must be timezone aware')
                now = now.astimezone(IST)
                if now.date() != session_date or now.weekday() >= 5 or now.time() >= clock_time(15,15):
                    return self.finish('market_cutoff')
                if now.time() < clock_time(9,15):
                    sleep(interval)
                    continue
                raw = provider.fetch(now)
                opening = pd.Timestamp(now).normalize()+pd.Timedelta(hours=9,minutes=15)
                if raw.empty:
                    if (pd.Timestamp(now)-opening).total_seconds() > max_delay:
                        raise PaperDataError('No fresh completed market candles')
                    sleep(interval)
                    continue
                frame = prepare_intraday_frame(raw, rsi_period=self.settings['data'].get('rsi_period',14))
                completed = pd.Timestamp(now).floor('min')
                if frame.index.tz is None or frame.index[-1] >= completed or set(frame.index.date) != {session_date}:
                    raise PaperDataError('Wrong-date or incomplete market candles')
                if (pd.Timestamp(now)-(frame.index[-1]+pd.Timedelta(minutes=1))).total_seconds() > max_delay:
                    raise PaperDataError('Market candles are stale')
                if self.previous is not None:
                    common = self.previous.index
                    if not common.isin(frame.index).all() or not frame.loc[common].equals(self.previous):
                        raise PaperDataError('Previously consumed market candles changed')
                if self.previous is not None and frame.index.equals(self.previous.index):
                    sleep(interval)
                    continue
                self.previous = frame.copy()
                # Gaps or missing indicators must never be filled using invented prices.
                if not (np.diff(frame.index.as_unit('ns').asi8) == 60_000_000_000).all():
                    raise PaperDataError('Market candle gap detected')
                scored = self.model.score(frame)
                if len(scored) >= 2:
                    if self.env is None:
                        self.initialize(scored, start_latest=True)
                        self.status = 'paper_running'
                    else:
                        last = self.env.df.iloc[-1].timestamp
                        new = scored.loc[scored.index > last]
                        if len(new) > 1:
                            raise PaperDataError('Missed completed candles; stopped instead of backfilling trades')
                        if len(new):
                            self.env.append_candles(new)
                            if self.advance(streaming=True):
                                return self.finish(self.completed_status('risk_halt'))
                    report = self.save()
                    if on_update:
                        on_update(report)
                elif on_update:
                    on_update({'status':'warming_up','completed_candles':len(frame)})
                sleep(interval)
        except KeyboardInterrupt:
            return self.finish('interrupted')
        except Exception:
            self.finish('data_or_policy_error')
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--csv',type=Path,help='replay a historical session without credentials')
    source.add_argument('--live-data',action='store_true',help='read Groww candles; all money remains simulated')
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--settings',type=Path,default=DEFAULT_PAPER_SETTINGS)
    parser.add_argument('--symbol',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--cash',type=float,help='virtual starting INR; never a real account balance')
    args = parser.parse_args(argv)
    try:
        settings, _ = paper_settings(args.settings,args.cash)
        model = FrozenPolicy(args.model_dir,args.symbol)
        features = (load_intraday_csv(args.csv,rsi_period=settings['data'].get('rsi_period',14))
                    if args.csv else None)
        run = PaperRun(model,args.settings,args.output_dir,args.symbol,cash=args.cash,
                       source='groww_completed_candles' if args.live_data else 'csv_replay')
        if args.csv:
            report = run.replay(features)
        else:
            print('PAPER ONLY: virtual money; Groww market data is read-only. Ctrl+C saves the session.')
            try:
                provider = ReadOnlyGrowwCandles.connect(args.settings,args.symbol)
            except PaperDataError:
                run.finish('data_authentication_failed')
                raise
            def progress(report):
                if report['status'] == 'warming_up':
                    print(f"PAPER warm-up: {report['completed_candles']} completed minute candles")
                else:
                    print(f"PAPER virtual equity INR {report['equity_inr']:.2f}; net P&L {report['net_pnl_inr']:.2f}; orders {report['executed_orders']}")
            report = run.poll(provider,on_update=progress)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(2,f'Paper trading stopped: {exc}\n')
    print(f"PAPER result: {report['status']}; virtual equity INR {report['equity_inr']:.2f}; net P&L INR {report['net_pnl_inr']:.2f}; orders {report['executed_orders']}")
    print(f'Report saved to {args.output_dir}. Real trading was not enabled.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
