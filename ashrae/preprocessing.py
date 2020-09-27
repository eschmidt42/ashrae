# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02_preprocessing.ipynb (unless otherwise specified).

__all__ = ['Processor', 'DEP_VAR', 'TIME_COL', 'test_var_names', 'store_var_names', 'load_var_names', 'store_df',
           'load_df', 'get_tabular_object', 'train_predict', 'hist_plot_preds', 'BoldlyWrongTimeseries']

# Cell
import pandas as pd
from pathlib import Path
import os
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import typing
import pickle
import ipywidgets as widgets

from sklearn import linear_model, tree, model_selection, ensemble
from ashrae import inspection
from fastai.tabular.all import *

import tqdm

from sklearn import linear_model, tree, model_selection, ensemble

# Cell
DEP_VAR = 'meter_reading'
TIME_COL = 'timestamp'

class Processor:

    dep_var_stats:dict = None

    def __call__(self, df_core:pd.DataFrame, df_building:pd.DataFrame=None,
                 df_weather:pd.DataFrame=None, dep_var:str=None, time_col:str=None,
                 add_time_features:bool=False, add_dep_var_stats:bool=False) -> pd.DataFrame:

        # TODO:
        # - add daily features: temperature delta per site_id, total rain fall, ...
        # - add global stats: mean, median and so on of dep_var by building_id or type
        # - add consumption of the days around the day of interest


        # sanity check presence of df_building if df_weather is given
        if df_weather is not None:
            assert df_building is not None, 'To join the weather info in `df_weather` you need to pass `df_building`.'

        self.dep_var = DEP_VAR if dep_var is None else dep_var
        self.time_col = TIME_COL if time_col is None else time_col

        self.conts, self.cats = [], []

        # sanity check if `df` is a test set (dep_var is missing)
        self.is_train = self.dep_var in df_core.columns

        # core pieces of dependent and independent variables
        self.dep_var_new = f'{self.dep_var}_log1p'
        if self.is_train:
            df_core[self.dep_var_new] = np.log(df_core[self.dep_var].values + 1)
        self.cats += ['building_id', 'meter']

        # adding basic statistics as features
        if add_dep_var_stats:
            df_core = self.add_dep_var_stats(df_core)

        # adding building information
        if df_building is not None:
            df_core = self.add_building_features(df_core, df_building)

        # adding weather information
        if df_weather is not None:
            df_core = self.add_weather_features(df_core, df_weather)

        # add timestamp related fields
        if add_time_features:
            df_core = self.add_time_features(df_core)

        df_core, var_names = self.cleanup(df_core)
        return df_core, var_names


    def add_dep_var_stats(self, df_core:pd.DataFrame):
        assert self.is_train or self.dep_var_stats is not None
        if self.is_train:
            self.dep_var_stats = dict()
        funs = {
            'median': lambda x: torch.median(tensor(x)).item(),
            'mean': lambda x: torch.mean(tensor(x)).item(),
            '5%': lambda x: np.percentile(x, 5),
            '95%': lambda x: np.percentile(x, 95),
        }
        for name, fun in funs.items():
            name = f'{self.dep_var}_{name}'
            self.conts.append(name)

            if self.is_train:
                value = fun(df_core[self.dep_var].values)
                df_core[name] = value
                self.dep_var_stats[name] = value
            else:
                df_core[name] = self.dep_var_stats[name]
        return df_core

    def add_time_features(self, df_core:pd.DataFrame):
        self.cats.extend(['timestampMonth', 'timestampDay', 'timestampWeek', 'timestampDayofweek',
                     'timestampDayofyear', 'timestampIs_month_end', 'timestampIs_month_start',
                     'timestampIs_quarter_start', 'timestampIs_quarter_end',
                     'timestampIs_year_start', 'timestampIs_year_end',])
        return add_datepart(df_core, self.time_col)

    def add_building_features(self, df_core:pd.DataFrame, df_building:pd.DataFrame):
        n = len(df_core)
        df_core = pd.merge(df_core, df_building, on='building_id', how='left')
        assert n == len(df_core)

        self.cats.extend(['site_id', 'primary_use'])
        self.conts.extend(['square_feet', 'year_built', 'floor_count'])
        return df_core

    def add_weather_features(self, df_core:pd.DataFrame, df_weather:pd.DataFrame):
        n = len(df_core)
        df_core = pd.merge(df_core, df_weather, on=['site_id', 'timestamp'], how='left')
        assert n == len(df_core)

        self.cats.extend(['cloud_coverage', 'wind_direction'])
        self.conts.extend(['air_temperature', 'dew_temperature', 'precip_depth_1_hr',
                      'sea_level_pressure', 'wind_speed'])
        return df_core

    def cleanup(self, df_core:pd.DataFrame):
        # converting cats to category type
        for col in self.cats:
            df_core[col] = df_core[col].astype('category')

        # removing features
        to_remove_cols = [self.dep_var, 'timestampYear', self.time_col]
        df_core = df_core.drop(columns=[c for c in df_core.columns if c in to_remove_cols])
        df_core = df_shrink(df_core, int2uint=True)

        var_names = {'conts': self.conts, 'cats': self.cats, 'dep_var': self.dep_var_new}
        if not self.is_train:
            df_core.set_index('row_id', inplace=True)
        missing_cols = [col for col in df_core.columns.values if col not in self.cats + self.conts + [self.dep_var_new]
                        and col != 'timestampElapsed']
        assert len(missing_cols) == 0, f'Missed to assign columns: {missing_cols} to `conts` or `cats`'
        return df_core, var_names

# Cell
def test_var_names(var_names:dict):
    assert isinstance(var_names, dict)
    assert 'conts' in var_names and 'cats' in var_names and 'dep_var' in var_names
    assert isinstance(var_names['conts'], list)
    assert isinstance(var_names['cats'], list)
    assert isinstance(var_names['dep_var'], str)

# Cell
def store_var_names(data_path:Path, var_names:dict):
    fname = data_path/'var_names.pckl'
    print(f'Storing var names at: {fname}')
    with open(fname, 'wb') as f:
        pickle.dump(var_names, f)

# Cell
def load_var_names(fname:Path):
    print(f'Reading var names at: {fname}')
    with open(fname, 'rb') as f:
        var_names = pickle.load(f)
    return var_names

# Cell
def store_df(path:Path, df:pd.DataFrame): df.to_parquet(path)

# Cell
def load_df(path:Path): return pd.read_parquet(path)

# Cell
def get_tabular_object(df:pd.DataFrame, var_names:dict,
                       splits=None):
    procs = [Categorify, FillMissing, Normalize]
    return TabularPandas(df.copy(), procs,
                         var_names['cats'], var_names['conts'],
                         y_names=var_names['dep_var'],
                         splits=splits)
    return to


def train_predict(df:pd.DataFrame, var_names:dict,
                  model, params:dict=None, n_rep:int=3,
                  n_samples_train:int=10000,
                  n_samples_test:int=10000,
                  test_size:float=.2):

    y_col = var_names['dep_var']
    score_vals = []
    params = {} if params is None else params

    to = get_tabular_object(df, var_names)

    for i in tqdm.tqdm(range(n_rep), total=n_rep, desc='Repetition'):

        m = model(**params)

        mask = to.xs.index.isin(
            np.random.choice(to.xs.index.values, size=int(test_size*len(to.xs)), replace=False)
        )

        _X = to.xs.loc[~mask, :].iloc[:n_samples_train]
        _y = to.ys.loc[~mask, y_col].iloc[:n_samples_train]
        m.fit(_X.values, _y.values)

        _X = to.xs.loc[mask, :].iloc[:n_samples_test]
        _y = to.ys.loc[mask, y_col].iloc[:n_samples_test]
        pred = m.predict(_X.values)
        s = torch.sqrt(F.mse_loss(tensor(pred), tensor(_y.values))).item()
        score_vals.append({'iter': i, 'rmse loss': s})

    return pd.DataFrame(score_vals)

# Cell
def hist_plot_preds(y0:np.ndarray, y1:np.ndarray,
                    label0:str='y0', label1:str='y1'):
    res = pd.concat(
        (
            pd.DataFrame({
                'y': y0,
                'set': [label0] * len(y0)
            }),
            pd.DataFrame({
                'y':y1,
                'set': [label1] * len(y1)
            })
        ),
        ignore_index=True
    )

    return px.histogram(res, x='y', color='set', marginal='box',
                        barmode='overlay', histnorm='probability density')

# Cell
class BoldlyWrongTimeseries:
    def __init__(self, xs, y_true, y_pred, t:pd.DataFrame=None):
        if t is None:
            self.df = xs.loc[:,['meter', 'building_id', 'timestampElapsed']].copy()
        else:
            self.df = xs.loc[:,['meter', 'building_id']].join(t.reset_index().drop_duplicates().set_index('index'))
        self.df['y_true'] = y_true
        self.df['y_pred'] = y_pred
        self.compute_misses(y_true)

    def compute_misses(self, y_true):
        fun = lambda x: np.mean(x**2)
        self.miss = (self.df.assign(difference=lambda x: x['y_pred']-x['y_true'])
                     .groupby(['meter', 'building_id'])
                     .agg(loss=pd.NamedAgg(column='difference', aggfunc=fun))
                     .sort_values('loss'))

# Cell
@patch
def plot_boldly_wrong(self:BoldlyWrongTimeseries,
                      nth_last:int=None,
                      meter:int=None, bid:int=None):

    assert (meter is not None and bid is not None) or (nth_last is not None)

    if nth_last is not None:
        ix = self.miss.iloc[[nth_last],:]
        meter = ix.index[0][0]
        bid = ix.index[0][1]
        loss = ix["loss"].values[0]
    else:
        ix = self.miss.xs((meter, bid))
        loss = ix.values[0]

    df_plot = self.df.loc[(self.df['meter']==meter) & (self.df['building_id']==bid)]
    df_plot = pd.concat((
        df_plot[['timestampElapsed', 'y_true']].rename(columns={'y_true':'y'}).assign(label='true'),
        df_plot[['timestampElapsed', 'y_pred']].rename(columns={'y_pred':'y'}).assign(label='pred'))
    )
    return df_plot.plot(kind='scatter', x='timestampElapsed',
                        y='y', color='label', opacity=.4,
                        title=f'pos {nth_last}: meter = {meter}, building_id = {bid}<br>loss = {loss:.3f}')


BoldlyWrongTimeseries.plot_boldly_wrong = plot_boldly_wrong

# Cell
@patch
def init_widgets(self:BoldlyWrongTimeseries):
    self.int_txt_loss = widgets.IntText(min=-len(self.miss), max=len(self.miss),
                                        description='Position', value=-1)
    self.int_txt_meter = widgets.IntText(min=self.df['meter'].min(), max=self.df['meter'].max(),
                                         description='Meter')
    self.int_txt_bid = widgets.IntText(min=self.df['building_id'].min(), max=self.df['building_id'].max(),
                                       description='building id')
    self.run_btn = widgets.Button(description='plot')
    self.switch_btn = widgets.Checkbox(description='Loss-based', value=True)
    self.run_btn.on_click(self.click_boldly_wrong)
    self.out_wdg = widgets.Output()

@patch
def run_boldly(self:BoldlyWrongTimeseries):
    if not hasattr(self, 'switch_btn'):
        self.init_widgets()
    return widgets.VBox([self.switch_btn, self.int_txt_loss,
                         self.int_txt_meter, self.int_txt_bid,
                         self.run_btn, self.out_wdg])

@patch
def click_boldly_wrong(self:BoldlyWrongTimeseries, change):
    self.out_wdg.clear_output()
    nth_last = None if self.switch_btn.value == False else self.int_txt_loss.value
    meter = None if self.switch_btn.value == True else self.int_txt_meter.value
    bid = None if self.switch_btn.value == True else self.int_txt_bid.value
    with self.out_wdg:
        print(f'nth_last {nth_last} meter {meter} bid {bid}')
        try:
            self.plot_boldly_wrong(nth_last=nth_last, meter=meter, bid=bid).show()
        except:
            raise ValueError(f'nth_last {nth_last} meter {meter} bid {bid} not a valid combination! Likely due to missing meter/bid combination')