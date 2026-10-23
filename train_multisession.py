"""Chronological multi-session forecast validation and fake-money PPO/DQN training."""
import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from charges import IntradayCharges
from data_loader import load_settings
from multi_session import (TRANSFORM, SessionEnvironment, SessionWindows,
                           load_sessions, score_session, split_sessions)
from quant_agent import QuantLSTMBrain
from rl_brain import RLTradingBrain, TradingEnvironment
from training_diagnostics import TrainingDiagnostics


def prediction_metrics(model, dataset, device, train_mean):
    predicted, actual = [], []
    model.eval()
    with torch.inference_mode():
        for features, targets in DataLoader(dataset, batch_size=32, shuffle=False):
            predicted.extend((model(features.to(device)).cpu().numpy().reshape(-1)/10000).tolist())
            actual.extend((targets.numpy().reshape(-1)/10000).tolist())
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    if not np.isfinite(predicted).all():
        raise ValueError('Forecast validation produced nonfinite values')
    return {'samples':len(actual), 'model_mse':float(np.mean((predicted-actual)**2)),
            'zero_return_mse':float(np.mean(actual**2)),
            'training_mean_mse':float(np.mean((train_mean-actual)**2)),
            'mean_prediction':float(predicted.mean()), 'mean_actual_return':float(actual.mean())}


def fit_forecast(fit, validation, epochs, device):
    scaler = StandardScaler().fit(np.concatenate([
        day.loc[:, list(QuantLSTMBrain.FEATURES)].to_numpy(dtype=float) for day in fit]))
    train = SessionWindows(fit, scaler)
    valid = SessionWindows(validation, scaler)
    train_mean = float(np.mean([train.targets[day][end] for day,end in train.samples]))
    model = QuantLSTMBrain().to(device)
    # Neutral initial forecast; train in basis points instead of tiny fractions.
    torch.nn.init.zeros_(model.linear.weight)
    torch.nn.init.zeros_(model.linear.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    initial = prediction_metrics(model, valid, device, train_mean)
    best_loss, best_epoch = initial['model_mse'], 0
    best = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    history = [{'epoch':0, 'validation_mse':best_loss}]
    stale = 0
    for epoch in range(1, epochs+1):
        model.train()
        for features, targets in DataLoader(train, batch_size=32, shuffle=True):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(features.to(device)), targets.to(device))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite forecast training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
        metrics = prediction_metrics(model, valid, device, train_mean)
        history.append({'epoch':epoch, 'validation_mse':metrics['model_mse']})
        print(f"Forecast epoch {epoch}: validation MSE {metrics['model_mse']:.8g}", flush=True)
        if metrics['model_mse'] < best_loss:
            best_loss, best_epoch, stale = metrics['model_mse'], epoch, 0
            best = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 5:
            break
    model.load_state_dict(best)
    model.eval()
    validation_metrics = prediction_metrics(model, valid, device, train_mean)
    passed = validation_metrics['model_mse'] < min(validation_metrics['zero_return_mse'],
                                                  validation_metrics['training_mean_mse'])
    return model, scaler, {'selected_epoch':best_epoch, 'history':history,
        'training_loss_units':'basis_points_squared',
        'reported_mse_units':'fractional_return_squared',
        'checkpoint_selection':'Minimum validation MSE including neutral epoch zero; patience 5.',
        'training_mean_return':train_mean, 'fit':prediction_metrics(model, train, device, train_mean),
        'validation':validation_metrics, 'validation_passed':bool(passed)}


def evaluate_sessions(policy, sessions, symbol, settings_path, initial_cash):
    cash = initial_cash
    peak, drawdown = cash, 0.0
    daily = []
    actions = dict.fromkeys(('hold','buy','sell'),0)
    orders = buys = sells = forced = rejects = sells_while_flat = 0
    fees = reward_sum = 0.0
    executions = []
    for session in sessions:
        if cash <= 0:
            break
        env = TradingEnvironment(session, symbol=symbol, settings_path=settings_path, initial_cash=cash)
        observation, info = env.reset()
        started = cash
        while True:
            proposal, _ = policy.predict(observation, deterministic=True)
            action = int(np.asarray(proposal).item())
            actions[('hold','buy','sell')[action]] += 1
            observation, reward, done, truncated, info = env.step(action)
            reward_sum += reward
            buys += info['buy_count']; sells += info['sell_count']
            forced += info['forced_exit_count']; rejects += int(info['order_rejected'])
            sells_while_flat += int(info.get('sell_while_flat',False))
            peak = max(peak, info['equity'])
            drawdown = max(drawdown, peak-info['equity'])
            if done or truncated:
                break
        cash = info['cash']
        fees += info['total_charges']
        orders += len(env.executions)
        executions.extend(env.executions)
        daily.append({'date':str(session.index[0].date()), 'starting_cash_inr':started,
            'ending_cash_inr':cash, 'net_pnl_inr':cash-started,
            'charges_inr':info['total_charges'], 'executed_orders':len(env.executions),
            'halted':not env.gatekeeper.trading_enabled, 'exit_reason':info['reason']})
        env.close()
    return {'cash':cash, 'equity':cash,
            'net_pnl_inr':cash-initial_cash, 'total_charges':fees,
            'max_drawdown_inr':drawdown, 'action_counts':actions,
            'buy_trades':buys, 'sell_trades':sells, 'forced_exits':forced,
            'rejected_entries':rejects, 'sell_while_flat':sells_while_flat,
            'total_executed_trades':orders,
            'steps':sum(actions.values()), 'reward':reward_sum,
            'sessions_evaluated':len(daily), 'sessions_available':len(sessions),
            'daily_results':daily, 'executions':executions,
            'capital_policy':'Carry virtual equity between evaluation days; reset daily loss gate each day.'}


def train_multisession(csv_path, settings_path, output_dir, *, symbol,
                       lstm_epochs=30, rl_timesteps=10000, seed=42,
                       algorithm='PPO', allow_weak_forecast=False):
    for name, value in (('lstm_epochs',lstm_epochs),('rl_timesteps',rl_timesteps)):
        if isinstance(value,bool) or not isinstance(value,int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0 <= seed < 2**32:
        raise ValueError('Invalid seed')
    if algorithm not in ('PPO','DQN'):
        raise ValueError('Use PPO or DQN')
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError('Choose a new output folder; previous runs are preserved')
    settings = load_settings(settings_path)
    if (settings.get('execution',{}).get('mode') != 'simulation'
            or settings.get('risk_limits',{}).get('trading_enabled') is not True
            or settings.get('data',{}).get('timezone') != 'Asia/Kolkata'
            or symbol not in settings.get('data',{}).get('supported_symbols',[])):
        raise ValueError('Use enabled simulation settings, IST and a configured symbol')
    groups = split_sessions(load_sessions(csv_path, settings))
    # Validate risk configuration before any model work.
    probe = TradingEnvironment(groups['fit'][0].assign(lstm_score=0),
                               symbol=symbol, settings_path=settings_path)
    if probe.simulator_version != 4:
        raise ValueError('Use the normalized fake-money execution profile')
    profile = probe.execution_profile()
    probe.close()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dates = {name:[str(day.index[0].date()) for day in days] for name,days in groups.items()}
    print('Chronological session groups:', json.dumps(dates), flush=True)
    model, scaler, forecast = fit_forecast(groups['fit'],groups['validation'],lstm_epochs,device)
    output.mkdir(parents=True, exist_ok=False)
    torch.save({key:value.detach().cpu() for key,value in model.state_dict().items()},output/'lstm.pt')
    np.savez(output/'feature_scaler.npz',mean=scaler.mean_,scale=scaler.scale_)
    report = {'training_pipeline':'multi_session_v1', 'simulator_version':4,
        'execution_profile':profile, 'forecast_transform':TRANSFORM.copy(),
        'mode':'simulation','uses_real_money':False, 'symbol':symbol,'algorithm':algorithm,
        'seed':seed,'device':str(device),'timezone':'Asia/Kolkata',
        'target':'next_close_fractional_return','features':list(model.FEATURES),
        'session_splits':dates,'forecast_diagnostics':forecast,
        'allow_weak_forecast':bool(allow_weak_forecast),
        'split_policy':'Fit scaler/model on fit days; checkpoint on validation only; PPO on RL days; evaluate on later days.',
        'boundaries':{'lstm_end':groups['fit'][-1].index[-1].isoformat(),
                      'validation_end':groups['validation'][-1].index[-1].isoformat(),
                      'rl_start':groups['rl'][0].index[0].isoformat(),
                      'rl_end':groups['rl'][-1].index[-1].isoformat(),
                      'evaluation_start':groups['evaluation'][0].index[0].isoformat()}}
    report['status'] = 'forecast_validated' if forecast['validation_passed'] else 'forecast_validation_failed'
    (output/'forecast_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if not forecast['validation_passed'] and not allow_weak_forecast:
        print('Forecast did not beat BOTH validation baselines. Policy training skipped. See forecast_report.json.',flush=True)
        return report
    scored = {name:[score_session(model,day,scaler,device) for day in days]
              for name,days in groups.items() if name in ('rl','evaluation')}
    environment = SessionEnvironment(scored['rl'],symbol,settings_path)
    kwargs = ({'n_steps':32,'batch_size':16,'n_epochs':2} if algorithm == 'PPO' else
              {'learning_starts':min(100,rl_timesteps//4),'buffer_size':max(1000,rl_timesteps),
               'batch_size':32,'train_freq':1,'gradient_steps':1})
    brain = RLTradingBrain(environment,algorithm=algorithm,seed=seed,device=str(device),model_kwargs=kwargs)
    diagnostics = TrainingDiagnostics()
    print('Training policy on RL sessions only...',flush=True)
    brain.model.learn(total_timesteps=rl_timesteps,callback=diagnostics)
    evaluation = evaluate_sessions(brain.model,scored['evaluation'],symbol,settings_path,profile['initial_cash_inr'])
    # Test forecast errors are measured only after all selection/training is complete.
    report['forecast_diagnostics']['evaluation'] = prediction_metrics(model,
        SessionWindows(groups['evaluation'],scaler),device,forecast['training_mean_return'])
    report.update(status='trained',training_diagnostics=diagnostics.summary(),
                  evaluation=evaluation,training_parameters=kwargs,
                  rl_requested_timesteps=rl_timesteps,rl_actual_timesteps=brain.model.num_timesteps,
                  charge_assumptions={'rates':asdict(IntradayCharges.from_settings(settings)),
                                      'status':'Estimate; not a verified broker contract note.'})
    brain.model.save(output/'rl_policy')
    (output/'run_metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    environment.close()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',required=True,type=Path)
    parser.add_argument('--settings',type=Path,default=Path(__file__).with_name('simulation_settings.json'))
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--symbol',default='RELIANCE')
    parser.add_argument('--lstm-epochs',type=int,default=30)
    parser.add_argument('--rl-timesteps',type=int,default=10000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--algorithm',choices=('PPO','DQN'),default='PPO')
    parser.add_argument('--allow-weak-forecast',action='store_true',
        help='Diagnostic override only: train a fake-money policy despite weak validation forecasts')
    args = parser.parse_args(argv)
    try:
        report = train_multisession(args.csv,args.settings,args.output_dir,symbol=args.symbol,
            lstm_epochs=args.lstm_epochs,rl_timesteps=args.rl_timesteps,seed=args.seed,
            algorithm=args.algorithm,allow_weak_forecast=args.allow_weak_forecast)
    except (ValueError,OSError,KeyError) as error:
        parser.exit(2,f'Training stopped: {error}\n')
    if report['status'] != 'trained':
        return 2
    print('Training actions:',report['training_diagnostics']['action_counts'])
    print('Evaluation actions:',report['evaluation']['action_counts'])
    print(f"Evaluation net P&L INR {report['evaluation']['net_pnl_inr']:.2f}; simulated orders {report['evaluation']['total_executed_trades']}")
    print(f'Report saved to {args.output_dir / "run_metrics.json"}. Real trading remains disabled.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
