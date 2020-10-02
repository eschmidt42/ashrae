# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01_inspection.ipynb (unless otherwise specified).

__all__ = ['get_csvs', 'CSV_NAMES', 'get_core_Xy', 'show_nans', 'get_building_X', 'get_weather_X']

# Cell
import pandas as pd
from pathlib import Path
from fastcore.utils import *
import os
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import typing

from fastai.tabular.all import *

import matplotlib.pyplot as plt

# Cell
CSV_NAMES = ['building', 'sample_submission', 'test', 'train', 'weather_test', 'weather_train']

def get_csvs(data_path:Path, csv_names:typing.List[str]=None) -> typing.Dict[str, Path]:
    csvs = sorted([v for v in data_path.ls() if v.name.endswith('.csv')])
    csv_names = CSV_NAMES if csv_names is None else csv_names
    return {_name: [_csv for _csv in csvs if _csv.name.startswith(_name)] for _name in csv_names}

# Cell
def get_core_Xy(path:Path, nrows:int=None) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['timestamp'], nrows=nrows)
    return df_shrink(df, int2uint=True)

# Cell
def show_nans(df:pd.DataFrame):
    nans = []
    for col in df.columns:
        nans.append({
            'nans count': df[col].isna().sum(),
            'col':col,
            'nans %': df[col].isna().sum() / len(df) * 100,
        })
    return pd.DataFrame(nans).sort_values('nans count', ascending=False)

# Cell
def get_building_X(path:Path):
    # TODO: year_built and floor_count actually are discrete values but contain nans
    # test if 'Int' dtype would work or if it breaks the things downstream
    df_building = pd.read_csv(path)
    return df_shrink(df_building, int2uint=True)

# Cell
def get_weather_X(path:Path):
    # TODO: cloud_coverage, wind_direction could be Int
    df = pd.read_csv(path, parse_dates=['timestamp'])
    return df