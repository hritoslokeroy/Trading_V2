"""Cost-aware 1/5/15-minute research with a simple, frozen-protocol ridge baseline.

No live orders, PPO training or deployable artifacts. Prediction labels are
future market moves; costs and risk controls are applied in chronological replay.
"""
import argparse
from datetime import date
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from data_loader import load_settings
from forecast_features import RELATIVE, feature_matrix
from multi_session import load_sessions
from rl_brain import TradingEnvironment

HORIZONS = (1,5,15)
ALPHA = 1000.0  # Prespecified, fixed for all horizons/folds; no later-date tuning.


def examples(sessions,horizon):
    if horizon not in HORIZONS:
        raise ValueError('Use a prespecified horizon of 1, 5 or 15 minutes')
    x,y,dates = [],[],[]
    for session in sessions:
        values = feature_matrix(session,RELATIVE)
        for end in range(59,len(session)-horizon-1):
            entry,exit_ = end+1,end+horizon+1
            if session.index[entry].strftime('%H:%M') >= '15:15':
                continue
            if session.index[exit_].strftime('%H:%M') > '15:15':
                continue
            x.append(values[end])
            y.append(float(session.open.iloc[exit_]/session.open.iloc[entry]-1))
            dates.append(str(session.index[end].date()))
    if not x:
        raise ValueError('No complete same-day forecast targets before the cutoff')
    return np.asarray(x),np.asarray(y),dates


def fit_model(sessions,horizon):
    x,y,_ = examples(sessions,horizon)
    model = make_pipeline(StandardScaler(),Ridge(alpha=ALPHA))
    model.fit(x,y*10000)  # Train in basis points, return canonical fractional forecasts.
    return model,float(y.mean())


def predict(model,values):
    result = np.asarray(model.predict(values),dtype=float)/10000
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite horizon forecasts')
    return result


def error_metrics(model,sessions,horizon,train_mean):
    x,y,_ = examples(sessions,horizon)
    p = predict(model,x)
    return {'samples':len(y),'model_mse':float(np.mean((p-y)**2)),
            'zero_return_mse':float(np.mean(y**2)),
            'training_mean_mse':float(np.mean((train_mean-y)**2)),
            'mean_prediction_bps':float(p.mean()*10000),
            'mean_actual_move_bps':float(y.mean()*10000)}


def estimated_cost_hurdle(env,price):
    """Known-price flat-round-trip estimate, not future entry/exit information."""
    entry = price*(1+env.slippage)
    quantity = env._size_entry(entry)
    if quantity == 0:
        return float('inf')
    # Use a fresh per-round-trip tariff estimate. Actual chronological replay
    # uses the simulator's session ledger and may differ by rounding/price moves.
    fees = env.charges.estimate_round_trip(quantity,entry,price*(1-env.slippage))['total']
    return (fees+2*price*env.slippage*quantity)/(price*(1-env.slippage)*quantity)


def replay(model,session,horizon,settings_path,symbol,initial_cash):
    env = TradingEnvironment(session,symbol=symbol,settings_path=settings_path,initial_cash=initial_cash)
    env.reset()
    # Preserve the same completed-indicator/60-row observation warmup as the LSTM.
    env.current_step = 59
    values = feature_matrix(session,RELATIVE)
    predictions = predict(model,values[59:])
    entry_step = None
    rejected = signals = eligible = positive = 0
    seen_predictions, hurdles = [],[]
    peak,drawdown = initial_cash,0.0
    while not env.done:
        i = env.current_step
        if i+1 >= len(env.df):
            env.finish('end_of_data')
            break
        action = 0
        if env.position:
            # Filled at entry_step's open. Exit proposal at the preceding close
            # fills entry_step+horizon open, unless risk forces an earlier exit.
            if i >= entry_step+horizon-1:
                action = 2
        else:
            planned_exit = session.index[i]+pd.Timedelta(minutes=horizon+1)
            # Only clock/known-price information decides entry, not future labels
            # or knowledge of which future samples will have positive outcomes.
            if planned_exit.time() <= pd.Timestamp('15:15').time():
                prediction = float(predictions[i-59])
                hurdle = estimated_cost_hurdle(env,float(env.df.close.iloc[i]))
                eligible += 1; positive += int(prediction>0)
                seen_predictions.append(prediction)
                if np.isfinite(hurdle): hurdles.append(hurdle)
                if prediction > hurdle:
                    action = 1
                    signals += 1
        _,_,done,_,info = env.step(action)
        rejected += int(info['order_rejected'])
        peak = max(peak,info['equity']); drawdown = max(drawdown,peak-info['equity'])
        if info['buy_count'] and env.position:
            entry_step = env.current_step
        if not env.position:
            entry_step = None
        if done:
            break
    info = env._info()
    buys = [fill for fill in env.executions if fill['side']=='BUY']
    exits = [fill for fill in env.executions if fill['side']=='SELL']
    return {'date':str(session.index[0].date()),'starting_cash_inr':initial_cash,
            'ending_cash_inr':info['cash'],'net_pnl_inr':info['equity']-initial_cash,
            'estimated_charges_inr':info['total_charges'],'max_drawdown_inr':drawdown,
            'entry_signals':signals,'eligible_flat_decisions':eligible,
            'positive_gross_forecasts':positive,
            'max_forecast_bps':max(seen_predictions)*10000 if seen_predictions else None,
            'min_estimated_cost_hurdle_bps':min(hurdles)*10000 if hurdles else None,
            'rejected_entries':rejected,'round_trips':len(exits),
            'executed_orders':len(env.executions),'forced_exits':sum(int(fill['forced']) for fill in exits),
            'final_quantity':env.quantity,'executions':env.executions,
            'entry_count':len(buys)}


def summarize_days(days,initial_cash):
    return {'net_pnl_inr':sum(day['net_pnl_inr'] for day in days),
            'ending_cash_inr':days[-1]['ending_cash_inr'] if days else initial_cash,
            'round_trips':sum(day['round_trips'] for day in days),
            'positive_days':sum(day['net_pnl_inr']>0 for day in days),
            'estimated_charges_inr':sum(day['estimated_charges_inr'] for day in days),
            'executed_orders':sum(day['executed_orders'] for day in days),
            'days':days,'no_trade_benchmark_pnl_inr':0.0}


def pooled_metrics(folds):
    total = sum(fold['forecast']['samples'] for fold in folds)
    return {key:sum(fold['forecast'][key]*fold['forecast']['samples'] for fold in folds)/total
            for key in ('model_mse','zero_return_mse','training_mean_mse')}


def development_folds(sessions,horizon,settings_path,symbol,initial_cash):
    folds,days = [],[]
    cash = initial_cash
    for index in range(len(sessions)-3,len(sessions)):
        fit = sessions[:index]
        score = sessions[index:index+1]
        print(f'Horizon {horizon}: fit through {fit[-1].index[0].date()}, score {score[0].index[0].date()}',flush=True)
        model,mean = fit_model(fit,horizon)
        metrics = error_metrics(model,score,horizon,mean)
        day = replay(model,score[0],horizon,settings_path,symbol,cash) if cash>0 else None
        if day:
            cash = day['ending_cash_inr']; days.append(day)
        folds.append({'fit_dates':[str(frame.index[0].date()) for frame in fit],
                      'scoring_dates':[str(score[0].index[0].date())], 'forecast':metrics})
    pooled = pooled_metrics(folds)
    wins = sum(fold['forecast']['model_mse'] < min(fold['forecast']['zero_return_mse'],
                fold['forecast']['training_mean_mse']) for fold in folds)
    trading = summarize_days(days,initial_cash)
    # A conservative development screen, not proof of significance/profitability.
    passed = (pooled['model_mse'] < min(pooled['zero_return_mse'],pooled['training_mean_mse'])
              and wins>=2 and trading['net_pnl_inr']>0
              and trading['positive_days']>=2 and trading['round_trips']>=3)
    return {'folds':folds,'pooled_forecast':pooled,'folds_beating_both_baselines':wins,
            'trading':trading,'development_screen_passed':bool(passed)}


def experiment(csv_path,settings_path,output_dir,*,fit_end,validation_end,symbol='RELIANCE'):
    root = Path(output_dir)
    if root.exists(): raise FileExistsError('Choose a fresh research output folder')
    fit_end,validation_end = date.fromisoformat(str(fit_end)),date.fromisoformat(str(validation_end))
    if validation_end<=fit_end: raise ValueError('Comparison end must follow fit end')
    settings = load_settings(settings_path)
    if (settings.get('execution',{}).get('mode')!='simulation'
            or settings.get('risk_limits',{}).get('trading_enabled') is not True
            or settings.get('data',{}).get('timezone')!='Asia/Kolkata'
            or symbol not in settings.get('data',{}).get('supported_symbols',[])):
        raise ValueError('Use enabled IST simulation settings and a configured symbol')
    sessions = load_sessions(csv_path,settings,RELATIVE)
    if any(day.index[0].date()>validation_end for day in sessions):
        raise ValueError('Use forecast_review.csv only; later evaluation/practice dates are forbidden')
    fitting = [day for day in sessions if day.index[0].date()<=fit_end]
    comparison = [day for day in sessions if fit_end<day.index[0].date()<=validation_end]
    if len(fitting)<8 or not comparison: raise ValueError('Need eight fitting dates and a later comparison')
    probe = TradingEnvironment(fitting[0],symbol=symbol,settings_path=settings_path)
    profile = probe.execution_profile(); initial_cash = profile['initial_cash_inr']
    if probe.simulator_version!=4: raise ValueError('Use the fake-money normalized profile')
    report = {'status':'research_complete','uses_real_money':False,'symbol':symbol,
        'execution_profile':profile,'model':'StandardScaler + Ridge, fixed alpha 1000, targets in basis points',
        'feature_set':RELATIVE,'horizons_minutes':list(HORIZONS),
        'target':'Future entry-open to exit-open market return. Enter next minute; planned exit H minutes after entry.',
        'trading_rule':'Enter only if predicted gross return exceeds known-price estimated charges plus slippage; simulate non-overlapping positions with existing risk controls.',
        'selection_rule':'Among 5/15-minute candidates require pooled forecast skill, two fold wins, positive net P&L, two positive days and three completed round trips; choose highest fold net P&L before later comparison.',
        'cost_note':'Configured estimates, session charge ledger and 2 bps slippage per leg. No financing costs, spread/depth or broker-specific leverage eligibility verification.',
        'notes':'Development evidence only. No PPO, deployable model, broker orders or later evaluation/practice data. Repeated/overlapping forecast targets are not independent trades.',
        'fit_dates':[str(day.index[0].date()) for day in fitting],
        'reserved_comparison_dates':[str(day.index[0].date()) for day in comparison]}
    report['candidates'] = {str(h):development_folds(fitting,h,settings_path,symbol,initial_cash) for h in HORIZONS}
    eligible = [h for h in (5,15) if report['candidates'][str(h)]['development_screen_passed']]
    if not eligible:
        report.update(selected_horizon=None,later_comparison=None,further_research_supported=False,
                      decision='No candidate met the prespecified development screen. Later comparison not scored; do not force trades or enable live.')
    else:
        chosen = max(eligible,key=lambda h:report['candidates'][str(h)]['trading']['net_pnl_inr'])
        print(f'Frozen horizon choice: {chosen}; scoring later development comparison.',flush=True)
        model,mean = fit_model(fitting,chosen)
        metrics = error_metrics(model,comparison,chosen,mean)
        cash = initial_cash; days=[]
        for session in comparison:
            if cash<=0: break
            day = replay(model,session,chosen,settings_path,symbol,cash)
            cash=day['ending_cash_inr']; days.append(day)
        trading = summarize_days(days,initial_cash)
        passed = (metrics['model_mse']<min(metrics['zero_return_mse'],metrics['training_mean_mse'])
                  and trading['net_pnl_inr']>0 and trading['round_trips']>=3)
        report.update(selected_horizon=chosen,later_comparison={'forecast':metrics,'trading':trading},
                      further_research_supported=bool(passed),
                      decision='Further separate research is justified, not live activation.' if passed else
                               'Selected horizon did not pass later comparison; do not deploy or force trades.')
    root.mkdir(parents=True,exist_ok=False)
    (root/'horizon_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path,required=True)
    parser.add_argument('--settings',type=Path,default=Path(__file__).with_name('simulation_settings.json'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--fit-end',type=date.fromisoformat,required=True)
    parser.add_argument('--validation-end',type=date.fromisoformat,required=True)
    parser.add_argument('--symbol',default='RELIANCE')
    args=parser.parse_args(argv)
    try:
        report=experiment(args.csv,args.settings,args.output_dir,fit_end=args.fit_end,
                          validation_end=args.validation_end,symbol=args.symbol)
    except (ValueError,OSError,KeyError) as error:
        parser.exit(2,f'Horizon research stopped: {error}\n')
    for horizon,result in report['candidates'].items():
        trading=result['trading']
        print(f"{horizon} minutes: net INR {trading['net_pnl_inr']:.2f}, round trips {trading['round_trips']}, passed {result['development_screen_passed']}")
    print(report['decision']); print('Saved:',args.output_dir/'horizon_report.json')
    print('No policy training or real orders were performed.')
    return 0


if __name__=='__main__': raise SystemExit(main())
