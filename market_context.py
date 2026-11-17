"""Versioned causal market observations for policies independent of LSTM scores."""
import numpy as np
from forecast_features import RELATIVE, prepare_features

MODE = 'market_context_v1'
SCHEMA = 'market_context_account_fractions_v1'
MARKET_NAMES = ('return_1_pct','return_5_pct','volatility_10_pct','candle_body_pct',
                'candle_range_pct','vwap_distance_pct','log_relative_volume',
                'centered_rsi','session_fraction')
MARKET_COLUMNS = tuple('_policy_'+name for name in MARKET_NAMES)
OBSERVATION_NAMES = ('forecast_return_pct','cash_fraction','position_count','open_pnl_fraction')+MARKET_NAMES


def enabled(settings):
    simulation = settings.get('simulation',{})
    if not isinstance(simulation,dict):
        raise ValueError('simulation settings must be an object')
    mode = simulation.get('policy_observation','legacy')
    if mode not in ('legacy',MODE):
        raise ValueError('Unknown policy observation format')
    return mode == MODE


def prepare_market_context(frame):
    """Call on the full, single-day OHLCV/indicator frame before any slicing."""
    result = prepare_features(frame,RELATIVE)
    if len(set(result.index.date)) != 1:
        raise ValueError('Prepare policy market context separately for each day')
    values = [result['_forecast_return_pct'],result.close.pct_change(5,fill_method=None)*100,
              result.close.pct_change(fill_method=None).rolling(10,min_periods=10).std(ddof=0)*100,
              result['_forecast_body_pct'],result['_forecast_range_pct'],
              result['_forecast_vwap_distance_pct'],result['_forecast_log_relative_volume'],
              result['_forecast_centered_rsi'],result['_forecast_session_fraction']]
    for column,value in zip(MARKET_COLUMNS,values):
        result[column] = value
    return result


def market_matrix(frame):
    if not set(MARKET_COLUMNS).issubset(frame.columns):
        raise ValueError('Missing policy market context; prepare full-day indicators before slicing')
    values = frame.loc[:,list(MARKET_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Policy market context is nonfinite; check warmup and volume')
    return values


def schema_diagnostics(sessions):
    values = np.concatenate([market_matrix(frame) for frame in sessions])
    return {'observation_dimension':len(OBSERVATION_NAMES),
        'independent_market_inputs':len(MARKET_NAMES),
        'varying_market_inputs':int((np.ptp(values,axis=0)>0).sum()),
        'forecast_identically_zero':all(np.equal(frame.lstm_score.to_numpy(),0).all() for frame in sessions),
        'market_ranges':{name:{'min':float(values[:,i].min()),'max':float(values[:,i].max()),
                               'std':float(values[:,i].std())} for i,name in enumerate(MARKET_NAMES)},
        'scope':'RL fitting sessions only. Varying observations do not establish a profitable strategy.'}
