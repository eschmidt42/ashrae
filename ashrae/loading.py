# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/loading.ipynb (unless otherwise specified).

__all__ = ['DATA_PATH', 'N_TRAIN', 'N_TEST', 'get_csvs', 'CSV_NAMES_MAP', 'get_meter_data', 'get_nan_stats',
           'show_nans', 'test_meter_train_and_test_set', 'get_building_data', 'test_building', 'get_weather_data',
           'test_weather', 'load_all']

# Cell
import pandas as pd
import os
import numpy as np
import typing
from loguru import logger

from fastcore.all import *
from fastai.tabular.all import *

# Cell
DATA_PATH = Path("../data")
N_TRAIN = 10_000 # number of samples to load for the train set
N_TEST = 10_000 # number of samples to load for the test set

# Cell
CSV_NAMES_MAP = {'building_metadata.csv':'building',
                 'test.csv':'test',
                 'train.csv':'train',
                 'weather_test.csv':'weather_test',
                 'weather_train.csv':'weather_train',
                 'ashrae-energy-prediction-publicleaderboard.csv': 'public-leaderboard'}

@typed
def get_csvs(data_path:Path=DATA_PATH, csv_names_map:dict={}) -> dict:
    csv_names = CSV_NAMES_MAP if len(csv_names_map) == 0 else csv_names_map
    csvs = (data_path.ls()
            .filter(lambda x: x.name.endswith('.csv'))
            .map_dict(lambda x: csv_names.get(x.name, None)))
    logger.info(f'Collected csv paths: {csvs}')
    return {v: k for k,v in csvs.items() if v is not None}

# Cell
@typed
def get_meter_data(path:Path, nrows:int=-1) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['timestamp'])
    if nrows > 0: df = df.sample(nrows)
    logger.info(f'Loading meter data: {path}')
    return df_shrink(df, int2uint=True)

# Cell
@typed
def get_nan_stats(df:pd.DataFrame, col:str) -> pd.Series:
    n = df[col].isna().sum()
    return pd.Series({'# NaNs': n,
                      'col': col,
                      'NaNs (%)': 100 * n / len(df)})

# Cell
@typed
def show_nans(df:pd.DataFrame) -> pd.DataFrame:
    nans = []
    for col in df.columns:
        nans.append(get_nan_stats(df, col))
    return (pd.concat(nans, axis=1).T
            .assign(**{
                '# NaNs': lambda x: x['# NaNs'].astype(int),
                'NaNs (%)': lambda x: x['NaNs (%)'].astype(float)})
            .sort_values('# NaNs', ascending=False)
            .set_index('col'))

# Cell
@typed
def test_meter_train_and_test_set(df_train:pd.DataFrame, df_test:pd.DataFrame):
    assert len(df_train) == (20216100 if N_TRAIN == -1 else N_TRAIN)
    assert len(df_test) == (41697600 if N_TEST == -1 else N_TEST)
    assert set(df_train['meter'].unique()) == set(df_test['meter'].unique())
    if N_TRAIN > 20216100 and N_TEST > 41697600:
        assert set(df_train['building_id'].unique()) == set(df_test['building_id'].unique())
    train_nans = show_nans(df_train)
    assert np.allclose(train_nans['# NaNs'].values, 0)
    test_nans = show_nans(df_test)
    assert np.allclose(test_nans['# NaNs'].values, 0)
    logger.info('Passed basic meter info tests')

# Cell
@typed
def get_building_data(path:Path=DATA_PATH/'building_metadata.csv') -> pd.DataFrame:
    # TODO: year_built and floor_count actually are discrete values but contain nans
    # test if 'Int' dtype would work or if it breaks the things downstream
    logger.info(f'Loading building data: {path}')
    df_building = pd.read_csv(path)
    return df_shrink(df_building, int2uint=True)

# Cell
@typed
def test_building(df_building:pd.DataFrame, df_core:pd.DataFrame):
    assert df_building['building_id'].nunique() == len(df_building)
    if N_TRAIN == -1: assert set(df_core['building_id'].unique()) == set(df_building['building_id'].unique())
    building_nans = show_nans(df_building)
    assert np.allclose(building_nans['# NaNs'].values, [1094, 774, 0, 0, 0, 0])
    logger.info('Passed basic building info test')

# Cell
@typed
def get_weather_data(path:Path=DATA_PATH/'weather_train.csv') -> pd.DataFrame:
    # TODO: cloud_coverage, wind_direction could be Int
    logger.info(f'Loading weather data: {path}')
    df_weather = pd.read_csv(path, parse_dates=['timestamp'])
    return df_shrink(df_weather, int2uint=True)

# Cell
@typed
def test_weather(df_weather:pd.DataFrame, df_building:pd.DataFrame):
    assert set(df_weather['site_id'].unique()) == set(df_building['site_id'].unique())
    weather_nans = show_nans(df_weather)
    assert weather_nans.loc['site_id', '# NaNs'] == 0
    assert weather_nans.loc['timestamp', '# NaNs'] == 0
    logger.info('Passed basic weather tests')

# Cell
@typed
def load_all(data_path:Path=DATA_PATH) -> dict:
    'Locates csvs, loads them and performs basic sanity checks'
    # locating csvs
    csvs = get_csvs(data_path)

    # loading data
    ashrae_data = {}

    ## loading meter readings
    ashrae_data['meter_train'] = get_meter_data(csvs['train'], nrows=N_TRAIN)
    ashrae_data['meter_test'] = get_meter_data(csvs['test'], nrows=N_TEST)
    test_meter_train_and_test_set(ashrae_data['meter_train'], ashrae_data['meter_test'])

    # loading building info
    ashrae_data['building'] = get_building_data(csvs['building'])
    test_building(ashrae_data['building'], ashrae_data['meter_train'])

    # loading weather data
    ashrae_data['weather_train'] = get_weather_data(csvs['weather_train'])
    test_weather(ashrae_data['weather_train'], ashrae_data['building'])
    ashrae_data['weather_test'] = get_weather_data(csvs['weather_test'])
    test_weather(ashrae_data['weather_test'], ashrae_data['building'])

    return ashrae_data