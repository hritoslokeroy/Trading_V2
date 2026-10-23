"""Manual real-money entry point. Paper mode never imports or calls this module."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path


def verify_paper_report(report_path, model_dir, symbol):
    from paper_trade import artifact_fingerprint
    report = json.loads(Path(report_path).read_text())
    if not isinstance(report,dict):
        raise ValueError('Paper report must be an object')
    if (report.get('mode') != 'paper' or report.get('uses_real_money') is not False
            or report.get('auto_switch_to_live') is not False or report.get('symbol') != symbol
            or report.get('algorithm') != 'PPO'
            or report.get('model_fingerprint') != artifact_fingerprint(model_dir)
            or report.get('status') not in ('replay_complete','market_cutoff','interrupted')):
        raise ValueError('A completed paper report for this exact PPO model and symbol is required')
    if (isinstance(report.get('decisions'),bool) or not isinstance(report.get('decisions'),int)
            or report['decisions'] < 1 or isinstance(report.get('executed_orders'),bool)
            or not isinstance(report.get('executed_orders'),int) or report['executed_orders'] < 1):
        raise ValueError('Paper practice must include decisions and simulated executions')
    if (report.get('simulator_version', 2) != 2
            or report.get('position_sizing', 'one_share') != 'one_share'):
        raise ValueError('Leveraged variable-share paper models cannot activate the one-share live adapter')
    if report.get('overlaps_training') is not False:
        raise ValueError('Use a paper session after the model training period before live activation')
    # A local report is an operator review aid, not independent proof of profitability.
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--paper-report',type=Path,required=True)
    parser.add_argument('--symbol',required=True)
    parser.add_argument('--instrument-token',required=True)
    parser.add_argument('--tick-size',required=True)
    parser.add_argument('--journal',type=Path,required=True)
    parser.add_argument('--confirm-real-money',action='store_true')
    args = parser.parse_args(argv)
    if not args.confirm_real_money:
        parser.error('Real orders require --confirm-real-money; use paper_trade.py for practice')
    try:
        from data_loader import load_settings
        settings = load_settings(args.settings)
        if (settings.get('execution',{}).get('mode') != 'live'
                or settings.get('risk_limits',{}).get('trading_enabled') is not True
                or settings.get('charges',{}).get('rates_verified') is not True):
            raise ValueError('Live mode, trading enabled and reviewed charge rates must all be explicit')
        verify_paper_report(args.paper_report,args.model_dir,args.symbol)
        from groww_env import ExecutionHalted, LiveGrowwExecutionAdapter
        adapter = LiveGrowwExecutionAdapter(args.model_dir,args.journal,
                 settings_path=args.settings,tick_size=args.tick_size)
        print('REAL MONEY MODE: broker orders are enabled for this explicitly requested session.')
        asyncio.run(adapter.run_market_session(args.instrument_token,args.symbol))
    except KeyboardInterrupt:
        parser.exit(130,'Live session interrupted. Verify broker orders and positions; cancellation does not guarantee liquidation.\n')
    except (OSError,ValueError,KeyError,RuntimeError):
        parser.exit(2,'Live startup/session failed. No automatic retry. Check configuration and reconcile any broker exposure before restart.\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
