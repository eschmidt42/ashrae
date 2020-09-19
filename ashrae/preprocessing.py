# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02_preprocessing.ipynb (unless otherwise specified).

__all__ = ['radical_merging']

# Cell
import pandas as pd
from pathlib import Path
import os
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import typing
import pickle

from sklearn import linear_model, tree, model_selection, ensemble

from fastai.tabular.all import *

# Cell
def radical_merging(df:pd.DataFrame, building:pd.DataFrame,
                    weather:pd.DataFrame, n_sample:int=None,
                    training:bool=True):

    tmp = df.copy(deep=True)

    bid_col = 'building_id'
    sid_col = 'site_id'
    time_col = 'timestamp'
    target_col = 'meter_reading'

    categorical = ['meter', 'primary_use', 'cloud_coverage', bid_col, sid_col]
    continuous = ['square_feet', 'year_built', 'floor_count',
                  'air_temperature', 'dew_temperature',
                  'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction',
                  'wind_speed']

    x_cols = [bid_col, 'meter', target_col, time_col] if training \
            else [bid_col, 'meter', time_col]
    X = tmp.loc[:,x_cols].copy()

    X = pd.merge(X, building, on=bid_col, how='left')
    X = pd.merge(X, weather, on=[sid_col, time_col], how='left')

    #return_cols =  categorical + continuous + [target_col,]  # time_col

    #X = X.loc[:,return_cols]
    if n_sample is not None:
        X = X.sample(n_sample)

    if training:
        X[target_col] = np.log(X[target_col] + 1)

    X = add_datepart(X, time_col)
    categorical.extend(['timestampMonth', 'timestampWeek', 'timestampDay',
                        'timestampDayofweek', 'timestampDayofyear', 'timestampIs_month_end',
                        'timestampIs_month_start', 'timestampIs_quarter_end',
                        'timestampIs_quarter_start', 'timestampIs_year_end',
                        'timestampIs_year_start'])

    continuous.extend(['timestampYear', 'timestampElapsed'])

    X = X.loc[:, [col for col in X.columns.values if col not in [time_col]]]

    missing_cont = [col for col in continuous if col not in X.columns]
    missing_cat = [col for col in categorical if col not in X.columns]
    assert len(missing_cat) == 0, f'{missing_cat} not in X!'
    assert len(missing_cont) == 0, f'{missing_cont} not in X!'

    X.loc[:,continuous] = X.loc[:,continuous].astype(float)
    X.loc[:,categorical] = X.loc[:,categorical].astype('category')

    return X, continuous, categorical