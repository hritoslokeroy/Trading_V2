"""Compare raw/relative inputs using nested chronological development folds only.

No policy training, broker access or later evaluation/practice data is used.
"""
import argparse
from datetime import date
import json
from pathlib import Path
import random
import numpy as np
import torch
from data_loader import load_settings
from forecast_features import RAW, RELATIVE
from multi_session import load_sessions, SessionWindows
from train_multisession import fit_forecast, prediction_metrics


def evaluate_candidate(groups, feature_set, epochs, seed, device):
    """Outer day never selects the model epoch, scaler or fitting-data mean."""
    reports = []
    for fold, outer_index in enumerate(range(len(groups)-3,len(groups))):
        random.seed(seed+fold); np.random.seed(seed+fold); torch.manual_seed(seed+fold)
        fit = groups[:outer_index-1]
        inner_validation = groups[outer_index-1:outer_index]
        outer = groups[outer_index:outer_index+1]
        print(f'{feature_set} fold {fold+1}: score {outer[0].index[0].date()}; '
              f'checkpoint {inner_validation[0].index[0].date()}',flush=True)
        model,scaler,selection = fit_forecast(fit,inner_validation,epochs,device,feature_set)
        metrics = prediction_metrics(model,SessionWindows(outer,scaler,feature_set),
                                     device,selection['training_mean_return'])
        reports.append({'fit_dates':[str(day.index[0].date()) for day in fit],
            'checkpoint_dates':[str(day.index[0].date()) for day in inner_validation],
            'scoring_dates':[str(day.index[0].date()) for day in outer],
            'selected_epoch':selection['selected_epoch'],'metrics':metrics})
    total = sum(fold['metrics']['samples'] for fold in reports)
    aggregate = {key:sum(fold['metrics'][key]*fold['metrics']['samples'] for fold in reports)/total
                 for key in ('model_mse','zero_return_mse','training_mean_mse')}
    wins = sum(fold['metrics']['model_mse'] < min(fold['metrics']['zero_return_mse'],
               fold['metrics']['training_mean_mse']) for fold in reports)
    return {'folds':reports,'pooled_metrics':aggregate,'samples':total,
            'folds_beating_both_baselines':wins,'fold_count':len(reports)}


def compare(csv_path, settings_path, output_dir, *, fit_end, validation_end,
            epochs=15, seed=42):
    if isinstance(epochs,bool) or not isinstance(epochs,int) or epochs<1:
        raise ValueError('epochs must be a positive integer')
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0 <= seed < 2**32-3:
        raise ValueError('Invalid seed')
    fit_end,validation_end = date.fromisoformat(str(fit_end)),date.fromisoformat(str(validation_end))
    if validation_end <= fit_end:
        raise ValueError('Validation end must follow fitting end')
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError('Use a fresh experiment output directory')
    settings = load_settings(settings_path)
    if (settings.get('execution',{}).get('mode') != 'simulation'
            or settings.get('data',{}).get('timezone') != 'Asia/Kolkata'):
        raise ValueError('Use IST simulation settings')
    sessions = {name:load_sessions(csv_path,settings,name) for name in (RAW,RELATIVE)}
    if any(day.index[0].date() > validation_end for day in sessions[RAW]):
        raise ValueError('Input includes later evaluation/practice dates. Use forecast_review.csv only.')
    fitting = {name:[day for day in days if day.index[0].date() <= fit_end]
               for name,days in sessions.items()}
    validation = {name:[day for day in days if fit_end < day.index[0].date() <= validation_end]
                  for name,days in sessions.items()}
    if len(fitting[RAW]) < 8 or not validation[RAW]:
        raise ValueError('Need at least eight fit sessions and one later comparison session')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    report = {'status':'development_experiment','uses_real_money':False,'device':str(device),
        'seed':seed,'max_epochs':epochs,
        'fit_dates':[str(day.index[0].date()) for day in fitting[RAW]],
        'comparison_dates':[str(day.index[0].date()) for day in validation[RAW]],
        'selection_rule':'Choose lower pooled fold MSE, then freeze that choice before the later comparison. Require positive skill against both baselines in aggregate and at least two of three folds.',
        'notes':'These previously inspected dates are development data. No policy or later evaluation/practice data is used. No trade reward or risk limits change.'}
    report['candidates'] = {name:evaluate_candidate(fitting[name],name,epochs,seed,device)
                            for name in (RAW,RELATIVE)}
    chosen = min((RAW,RELATIVE),key=lambda name:report['candidates'][name]['pooled_metrics']['model_mse'])
    # Candidate choice uses only nested folds INSIDE the original fitting dates.
    # The later comparison cannot change that choice or its epoch selection.
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model,scaler,selected = fit_forecast(fitting[chosen][:-1],fitting[chosen][-1:],epochs,device,chosen)
    metrics = prediction_metrics(model,SessionWindows(validation[chosen],scaler,chosen),
                                 device,selected['training_mean_return'])
    pooled = report['candidates'][chosen]['pooled_metrics']
    fold_gate = (pooled['model_mse'] < min(pooled['zero_return_mse'],pooled['training_mean_mse'])
                 and report['candidates'][chosen]['folds_beating_both_baselines']>=2)
    comparison_gate = metrics['model_mse'] < min(metrics['zero_return_mse'],metrics['training_mean_mse'])
    report.update(selected_feature_set=chosen,final_checkpoint_dates=[str(fitting[chosen][-1].index[0].date())],
        selected_epoch=selected['selected_epoch'],later_comparison=metrics,
        further_research_supported=bool(fold_gate and comparison_gate),
        decision='Consider a separately validated training experiment; this is not live readiness.' if fold_gate and comparison_gate
                 else 'No reliable advantage established. Do not bypass validation or force trades.')
    root.mkdir(parents=True,exist_ok=False)
    (root/'comparison_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path,required=True)
    parser.add_argument('--settings',type=Path,default=Path(__file__).with_name('simulation_settings.json'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--fit-end',type=date.fromisoformat,required=True)
    parser.add_argument('--validation-end',type=date.fromisoformat,required=True)
    parser.add_argument('--lstm-epochs',type=int,default=15)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args(argv)
    try:
        report=compare(args.csv,args.settings,args.output_dir,fit_end=args.fit_end,
                       validation_end=args.validation_end,epochs=args.lstm_epochs,seed=args.seed)
    except (ValueError,OSError,KeyError) as error:
        parser.exit(2,f'Experiment stopped: {error}\n')
    for name,candidate in report['candidates'].items():
        print(name,candidate['pooled_metrics'],'fold wins:',candidate['folds_beating_both_baselines'])
    print(report['decision'])
    print('Saved:',args.output_dir/'comparison_report.json')
    print('No policy training or broker orders were performed.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
