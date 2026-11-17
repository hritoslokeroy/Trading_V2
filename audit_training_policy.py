"""Replay a frozen policy on its original RL dates; strictly in-sample diagnosis."""
import argparse
import json
from pathlib import Path
import pandas as pd
from data_loader import prepare_intraday_frame
from paper_trade import FrozenPolicy, PaperRun, paper_settings


def audit(csv_path,model_dir,settings_path,output_dir,*,symbol='RELIANCE'):
    root=Path(output_dir)
    if root.exists():
        raise FileExistsError('Choose a fresh audit output folder; earlier results are preserved')
    settings,cash=paper_settings(settings_path)
    model=FrozenPolicy(model_dir,symbol)
    if model.metadata.get('execution_profile',{}).get('initial_cash_inr') != cash:
        raise ValueError('Audit virtual cash must match the model training starting capital')
    expected=model.metadata.get('session_splits',{}).get('rl')
    if not isinstance(expected,list) or not expected or any(not isinstance(day,str) for day in expected):
        raise ValueError('Model does not identify its original RL training dates')
    raw=pd.read_csv(csv_path)
    if 'timestamp' not in raw:
        raise ValueError('Input needs timestamp,open,high,low,close,volume')
    stamps=pd.DatetimeIndex(pd.to_datetime(raw.timestamp,format='mixed',errors='raise'))
    stamps=stamps.tz_localize('Asia/Kolkata') if stamps.tz is None else stamps.tz_convert('Asia/Kolkata')
    if stamps.hasnans or stamps.has_duplicates or not stamps.is_monotonic_increasing:
        raise ValueError('Input timestamps must be sorted, unique and valid')
    dates=sorted(set(str(day) for day in stamps.date))
    if dates!=sorted(expected):
        raise ValueError('Input must contain exactly the model RL training dates; exclude evaluation/practice dates')
    # Validate all dates before creating the audit output. No original data or
    # checkpoint is overwritten. Indicators reset by day as during training.
    prepared=[]
    for day in dates:
        frame=raw.loc[stamps.strftime('%Y-%m-%d')==day].copy()
        frame['timestamp']=stamps[stamps.strftime('%Y-%m-%d')==day].astype(str)
        features=prepare_intraday_frame(frame,rsi_period=settings['data'].get('rsi_period',14))
        prepared.append((day,features))
    root.mkdir(parents=True,exist_ok=False)
    reports=[]
    for day,features in prepared:
        run=PaperRun(model,settings_path,root/day,symbol)
        report=run.replay(features)
        reports.append(report)
        print(f"{day}: frozen entries {sum(x['side']=='BUY' for x in run.env.executions)}, "
              f"orders {report['executed_orders']}, net INR {report['net_pnl_inr']:.2f}",flush=True)
    result={'mode':'training_data_audit','uses_real_money':False,'learning_during_audit':False,
        'is_out_of_sample':False,'model_fingerprint':model.fingerprint,'symbol':symbol,
        'dates':dates,'capital_policy':'Each day starts at configured virtual cash, matching training episode resets.',
        'initial_cash_per_day_inr':cash,
        'summed_independent_daily_net_pnl_inr':sum(r['net_pnl_inr'] for r in reports),
        'executed_orders':sum(r['executed_orders'] for r in reports),
        'action_counts':{name:sum(r['action_counts'][name] for r in reports) for name in ('hold','buy','sell')},
        'sell_while_flat':sum(r['sell_while_flat'] for r in reports),
        'estimated_charges_inr':sum(r['estimated_charges_inr'] for r in reports),
        'daily_results':[{'date':day,'net_pnl_inr':r['net_pnl_inr'],'executed_orders':r['executed_orders'],
            'action_counts':r['action_counts'],'status':r['status']} for day,r in zip(dates,reports)],
        'interpretation':'Training exploration totals are not the final frozen policy. This replay diagnoses fitting behaviour; it cannot establish predictive skill on new data.'}
    (root/'training_policy_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--settings',type=Path,default=Path(__file__).with_name('paper_market_50k_settings.json'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--symbol',default='RELIANCE')
    args=parser.parse_args(argv)
    try:
        result=audit(args.csv,args.model_dir,args.settings,args.output_dir,symbol=args.symbol)
    except (ValueError,OSError,KeyError) as error:
        parser.exit(2,f'Training-policy audit stopped: {error}\n')
    print('Frozen policy actions:',result['action_counts'])
    print('Saved:',args.output_dir/'training_policy_audit.json')
    print('In-sample diagnosis only. No learning, real-account calls or broker orders.')
    return 0


if __name__=='__main__': raise SystemExit(main())
