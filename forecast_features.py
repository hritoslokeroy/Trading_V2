"""Versioned causal forecast inputs; raw execution prices stay untouched."""
import numpy as np
from quant_agent import QuantLSTMBrain

RAW = 'raw_v1'
RELATIVE = 'relative_v1'
RELATIVE_NAMES = ('return_pct','body_pct','range_pct','vwap_distance_pct',
                  'log_relative_volume','centered_rsi','session_fraction')
RELATIVE_COLUMNS = tuple('_forecast_'+name for name in RELATIVE_NAMES)


def feature_names(feature_set):
    if feature_set == RAW:
        return QuantLSTMBrain.FEATURES
    if feature_set == RELATIVE:
        return RELATIVE_NAMES
    raise ValueError('Unknown forecast feature set')


def prepare_features(frame, feature_set=RAW):
    feature_names(feature_set)
    result = frame.copy()
    if feature_set == RAW:
        return result
    # Always call on a full day before removing indicator warmup. Do not
    # substitute future values or invent candles to complete an input window.
    previous = result.close.shift(1)
    median_volume = result.volume.rolling(20,min_periods=1).median().replace(0,np.nan)
    values = [result.close/previous*100-100,
              result.close/result.open*100-100,
              (result.high-result.low)/previous*100,
              result.close/result.vwap*100-100,
              np.log1p(result.volume/median_volume),
              (result.rsi-50)/50,
              (result.index.hour*60+result.index.minute-555)/375]
    for column,value in zip(RELATIVE_COLUMNS,values):
        result[column] = value
    return result


def feature_matrix(frame, feature_set=RAW):
    feature_names(feature_set)
    columns = QuantLSTMBrain.FEATURES if feature_set == RAW else RELATIVE_COLUMNS
    if not set(columns).issubset(frame.columns):
        raise ValueError('Prepare forecast features on the complete session before removing warmup')
    values = frame.loc[:,list(columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Forecast inputs contain nonfinite values; check volume and warmup')
    return values


def feature_set_from_metadata(metadata):
    transform = metadata.get('feature_transform')
    if transform is None:
        return RAW
    if transform not in ({'name':RAW},{'name':RELATIVE}):
        raise ValueError('Unsupported forecast input transform')
    name = transform['name']
    if metadata.get('forecast_features') != list(feature_names(name)):
        raise ValueError('Forecast feature names do not match the saved input transform')
    return name
