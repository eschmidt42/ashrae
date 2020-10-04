# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02_preprocessing.ipynb (unless otherwise specified).

__all__ = ['split_dataset', 'Processor', 'DEP_VAR', 'TIME_COL', 'test_var_names', 'store_var_names', 'load_var_names',
           'store_df', 'load_df', 'get_tabular_object', 'train_predict', 'SPLIT_PARAMS', 'hist_plot_preds',
           'BoldlyWrongTimeseries']

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
from sklearn.preprocessing import OneHotEncoder

import itertools

# Cell
def split_dataset(X:pd.DataFrame, split_kind:str='random',
                  train_frac:float=8, t_train:pd.DataFrame=None):

    def random_split():
        n_train = int(len(X)*train_frac)
        train_bool = X.index.isin(np.random.choice(X.index.values, size=n_train, replace=False))
        return train_bool

    def time_split():
        assert 'timestamp' in X.columns
        time_col = 'timestamp'
        ts = X[time_col].sort_values(ascending=True)
        ix = int(len(X)*train_frac)
        threshold_t = ts.iloc[ix:].values[0]
        return X[time_col] < threshold_t

    def time_split_day():
        time_col = 'timestampDayofyear'

        if time_col not in X.columns:
            t = X['timestamp'].dt.dayofyear
        else:
            t = X[time_col]

        days = (t.value_counts()
                .rename('count')
                .sample(frac=1)
                .to_frame()
                .cumsum()
                .pipe(lambda x: x.loc[x['count'] <= (train_frac * len(t))]))

        num_train_days = len(days)
        mask = t.isin(days.index.values)

        assert mask.sum() > 0
        return mask

    def fix_time_split():
        assert t_train is not None
        time_col = 'timestamp'
        assert time_col in X.columns

        mask = X[time_col].isin(t_train[time_col])
        assert mask.sum() > 0
        return mask

    split_funs = {
        'random': random_split,
        'time': time_split,
        'fix_time': fix_time_split,
        'time_split_day': time_split_day,
    }

    assert split_kind in split_funs
    train_bool = split_funs[split_kind]()

    train_idx = np.where(train_bool)[0]
    valid_idx = np.where(~train_bool)[0]

    return (list(train_idx), list(valid_idx))

# Cell
DEP_VAR = 'meter_reading'
TIME_COL = 'timestamp'

class Processor:

    dep_var_stats:dict = None

    def __call__(self, df_core:pd.DataFrame, df_building:pd.DataFrame=None,
                 df_weather:pd.DataFrame=None, dep_var:str=None, time_col:str=None,
                 add_time_features:bool=False, add_dep_var_stats:bool=False,
                 remove_leading_zeros:bool=False, remove_trailing_zeros:bool=False,
                 remove_empty_weeks_before_first_full_week:bool=True, add_onehot:bool=False,
                 t_train:pd.DataFrame=None) -> pd.DataFrame:

        # TODO:
        # - add daily features: temperature delta per site_id, total rain fall, ...
        # - add global stats: mean, median and so on of dep_var by building_id or type
        # - add consumption of the days around the day of interest


        # sanity check presence of df_building if df_weather is given
        if df_weather is not None:
            assert df_building is not None, 'To join the weather info in `df_weather` you need to pass `df_building`.'

        self.dep_var = DEP_VAR if dep_var is None else dep_var
        self.time_col = TIME_COL if time_col is None else time_col

        self.conts, self.cats, self.cats_order = [], [], {}

        # sanity check if `df` is a test set (dep_var is missing)
        self.is_train = self.dep_var in df_core.columns

        # core pieces of dependent and independent variables
        dep_var_new = f'{self.dep_var}_log1p'
        if self.is_train:
            df_core[dep_var_new] = np.log(df_core[self.dep_var].values + 1)
        self.dep_var = dep_var_new
        self.cats += ['building_id', 'meter']

        # add timestamp related fields
        if add_time_features:
            df_core = self.add_time_features(df_core)

        # removing leading zeros
        if remove_leading_zeros and self.is_train:
            df_core = self.remove_leading_zeros(df_core, t_train=t_train)

        # removing trailing zeros
        if remove_trailing_zeros and self.is_train:
            df_core = self.remove_trailing_zeros(df_core, t_train=t_train)

        # removing weeks before the first full week
        if remove_empty_weeks_before_first_full_week and self.is_train:
            df_core = self.remove_empty_weeks_before_first_full_week(df_core, t_train=t_train)

        # adding basic statistics as features
        if add_dep_var_stats:
            df_core = self.add_dep_var_stats(df_core, grp_cols=['meter', 'building_id', 'timestampHour'])

        # adding onehot encoded columns
        if add_onehot:
            df_core = self.add_onehot_encoded(df_core, onehot_cols=['meter'])

        # adding building information
        if df_building is not None:
            df_core = self.add_building_features(df_core, df_building)

        # adding weather information
        if df_weather is not None:
            df_core = self.add_weather_features(df_core, df_weather)


        df_core, var_names = self.cleanup(df_core)
        return df_core, var_names

    def add_onehot_encoded(self, df_core:pd.DataFrame, onehot_cols:typing.List[str]=['meter']):

        t_col = 'timestampHour'
        do_add_t = t_col in onehot_cols and t_col not in df_core.columns.values
        if do_add_t:
            df_core[t_col] = df_core['timestamp'].dt.hour

        if self.is_train:
            onehot = OneHotEncoder()
            df_core['id'] = [str(v) for v in zip(*[df_core[v] for v in onehot_cols])]
            onehot.fit(df_core.loc[:, ['id']])
            self.onehot_tfm = onehot
            self.onehot_cols = onehot_cols

        names = [f'{"-".join(onehot_cols)}_{v}' for v in self.onehot_tfm.categories_[0]]

        self.cats.extend(names)

        df_onehot = pd.DataFrame(self.onehot_tfm.transform(df_core.loc[:, ['id']]).toarray(),
                                 columns=names, index=df_core.index, dtype=bool)

        to_drop = ['id']
        if do_add_t:
            to_drop.append(t_col)
        df_core.drop(columns=to_drop, inplace=True)
        return pd.concat((df_core, df_onehot), axis=1)


    def add_dep_var_stats(self, df_core:pd.DataFrame, grp_cols:typing.List[str]=None):
        assert self.is_train or self.dep_var_stats is not None
        if self.is_train:
            self.dep_var_stats = dict()
        funs = {
            'median': lambda x: torch.median(tensor(x)).item(),
            'mean': lambda x: torch.mean(tensor(x)).item(),
            '5%': lambda x: np.percentile(x, 5),
            '95%': lambda x: np.percentile(x, 95),
        }

        # computing stats for self.dep_var on the coarsest possible level
        for name, fun in funs.items():
            name = f'{self.dep_var}_{name}'
            self.conts.append(name)

            if self.is_train:
                value = fun(df_core[self.dep_var].values)
                df_core[name] = value
                self.dep_var_stats[name] = value
            else:
                df_core[name] = self.dep_var_stats[name]

        # adding stats of self.dep_var on a more granular level
        if grp_cols is not None:
            t_col = 'timestampHour'
            do_add_t = t_col in grp_cols and t_col not in df_core.columns.values
            if do_add_t:
                df_core[t_col] = df_core['timestamp'].dt.hour

            assert all([c in df_core.columns.values for c in grp_cols])

            for fun_name, fun in funs.items():
                name = f'{self.dep_var}_{"-".join(grp_cols)}_{fun_name}'
                self.conts.append(name)

                if self.is_train:

                    self.dep_var_stats[name] = (df_core.groupby(grp_cols)[self.dep_var]
                                                .agg(fun)
                                                .rename(name))
                df_core = df_core.join(self.dep_var_stats[name], on=grp_cols)

            df_core.drop(columns=[t_col], inplace=True)
        return df_core

    def add_time_features(self, df_core:pd.DataFrame):
        self.cats.extend(['timestampMonth', 'timestampDay', 'timestampWeek', 'timestampDayofweek',
                          'timestampDayofyear', 'timestampIs_month_end', 'timestampIs_month_start',
                          'timestampIs_quarter_start', 'timestampIs_quarter_end',
                          'timestampIs_year_start', 'timestampIs_year_end', 'timestampHour'])

        df_core = add_datepart(df_core, self.time_col, drop=False)

        df_core['timestampHour'] = df_core[self.time_col].dt.hour

        self.cats_order.update({
            c: sorted(df_core[c].unique()) for c in ['timestampMonth', 'timestampDay',
                                                     'timestampWeek', 'timestampDayofweek',
                                                     'timestampDayofyear']
        })
        return df_core

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

        self.cats.extend(['cloud_coverage'])
        self.cats_order['cloud_coverage'] = sorted([v for v in df_core['cloud_coverage'].unique() if np.isfinite(v)])
        self.conts.extend(['wind_direction', 'air_temperature', 'dew_temperature', 'precip_depth_1_hr',
                      'sea_level_pressure', 'wind_speed'])
        return df_core

    def remove_leading_zeros(self, df_core:pd.DataFrame, t_train:pd.DataFrame=None):
        'there are time series which start with many 0 values in the dep_var. this method removes those values'

        n = len(df_core)
        assert self.dep_var in df_core.columns
        assert 'timestamp' in df_core.columns
        assert df_core[self.dep_var].min() == 0

        # finding the first timestamps after 0s
        mins = (df_core[df_core[self.dep_var] > 0].groupby(["building_id","meter"])
                .timestamp.min().rename("first_timestamp"))
        df_core = df_core.join(mins,on=["building_id","meter"])

        mask = df_core['first_timestamp'] <= df_core['timestamp']

        if t_train is not None:
            t_mask = df_core['timestamp'].isin(t_train['timestamp'])
            mask = (mask & t_mask) | ~t_mask

        df_core = df_core.loc[mask,:]
        df_core.drop(columns=['first_timestamp'], inplace=True)
        assert len(df_core) < n
        print(f'Removed {(1-len(df_core)/n)*100:.4f} % of rows')

        return df_core


    def remove_trailing_zeros(self, df_core:pd.DataFrame, t_train:pd.DataFrame=None):
        'there are time series which end with many 0 values in the dep_var. this method removes those values'

        n = len(df_core)
        assert self.dep_var in df_core.columns
        assert 'timestamp' in df_core.columns
        assert df_core[self.dep_var].min() == 0

        # finding the first timestamps after 0s
        maxs = (df_core[df_core[self.dep_var] > 0].groupby(["building_id","meter"])
                .timestamp.max().rename("last_timestamp"))
        df_core = df_core.join(maxs,on=["building_id","meter"])

        mask = df_core['last_timestamp'] >= df_core['timestamp']

        if t_train is not None:
            t_mask = df_core['timestamp'].isin(t_train['timestamp'])
            mask = (mask & t_mask) | ~t_mask

        df_core = df_core.loc[mask,:]
        df_core.drop(columns=['last_timestamp'], inplace=True)
        assert len(df_core) < n
        print(f'Removed {(1-len(df_core)/n)*100:.4f} % of rows')

        return df_core

    def remove_empty_weeks_before_first_full_week(self, df_core:pd.DataFrame,
                                                  t_train:pd.DataFrame):
        'there are some timeseries with weeks in the beginning which are basically empty'
        # TODO: something is likely buggy, losing combinations of building_id and meter

        n = len(df_core)
        n_comb = len(df_core.loc[:,['building_id', 'meter']].drop_duplicates())

        def get_combs(df):
            return set([tuple([row['building_id'], row['meter']])
                                for _, row in (df.loc[:,['building_id', 'meter']]
                                               .drop_duplicates()
                                               .iterrows())])

        combs = get_combs(df_core)

        df_core['timestampWeek'] = df_core[self.time_col].dt.isocalendar().week

        counts = (df_core[df_core[self.dep_var] > 0]
                  .groupby(["building_id","meter","timestampWeek"])
                  .timestamp.count()
                  .rename("num_weekly_measurements").reset_index())

        expected_num = 24*7 # hours per week with a measurement
        expected_num *= 0.5

        first_full_week = (counts[counts['num_weekly_measurements'] > expected_num]
                           .groupby(["building_id","meter"])
                           .timestampWeek.min()
                           .rename("first_full_week")
                           .to_frame())

        df_core = df_core.join(first_full_week, on=["building_id","meter"])

        mask = df_core['timestampWeek'] >= df_core['first_full_week']

        na_series = df_core.loc[df_core['first_full_week'].isna(), ['building_id', 'meter']].drop_duplicates()

        print('number of na combinations', len(na_series))
        display('na bids & meters', len(na_series), na_series)

        mask = mask & df_core['first_full_week'].notna() # some time series have in each week less than the required number of observations

        if t_train is not None:
            t_mask = df_core['timestamp'].isin(t_train['timestamp'])
            mask = (mask & t_mask) | ~t_mask

        df_core = df_core.loc[mask,:]

        df_core.drop(columns=['timestampWeek', 'first_full_week'], inplace=True)

        na_combs = get_combs(df_core)
        miss_combs = [v for v in na_combs if v not in combs]
        print('combs diff', miss_combs, len(miss_combs))

        assert len(df_core) < n
        assert len(na_series) == 38
        print(f'Removed {(1-len(df_core)/n)*100:.4f} % of rows')
        print(f'{len(na_series)} of building_id/meter combinations count as empty = {100 * len(na_series) / n_comb:.4f} % of all combinations')
        return df_core


    def cleanup(self, df_core:pd.DataFrame):
        # converting cats to category type
        for col in self.cats:
            if df_core[col].dtype == bool: continue
            df_core[col] = df_core[col].astype('category')
            if col in self.cats_order:
                df_core[col].cat.set_categories(self.cats_order[col],
                                                ordered=True, inplace=True)

        # removing features
        to_remove_cols = ['meter_reading', 'timestampYear'] # , self.time_col
        df_core = df_core.drop(columns=[c for c in df_core.columns if c in to_remove_cols])

        # shrinking the data frame
        df_core = df_shrink(df_core, int2uint=True)

        var_names = {'conts': self.conts, 'cats': self.cats, 'dep_var': self.dep_var}
        if not self.is_train:
            df_core.set_index('row_id', inplace=True)
        missing_cols = [col for col in df_core.columns.values if col not in self.cats + self.conts + [self.dep_var]
                        and col not in ['timestampElapsed', self.time_col, 'meter_reading']]
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
                       splits=None, procs:list=[Categorify, FillMissing, Normalize]):
    return TabularPandas(df.copy(), procs,
                         var_names['cats'], var_names['conts'],
                         y_names=var_names['dep_var'],
                         splits=splits)

SPLIT_PARAMS = dict(
    train_frac = .8,
    split_kind = 'time_split_day',
)


def train_predict(df:pd.DataFrame, var_names:dict,
                  model, params:dict=None, n_rep:int=3,
                  n_samples_train:int=10000,
                  n_samples_valid:int=10000,
                  procs:list=[Categorify, FillMissing, Normalize],
                  split_params:dict=None):

    split_params = SPLIT_PARAMS if split_params is None else split_params
    y_col = var_names['dep_var']
    score_vals = []
    params = {} if params is None else params

    to = get_tabular_object(df, var_names, procs=procs)

    for i in tqdm.tqdm(range(n_rep), total=n_rep, desc='Repetition'):

        m = model(**params)
        splits = split_dataset(df, **split_params)

        mask = to.xs.index.isin(splits[0])

        _X = to.xs.loc[~mask, :].iloc[:n_samples_train]
        _y = to.ys.loc[~mask, y_col].iloc[:n_samples_train]
        m.fit(_X.values, _y.values)

        _X = to.xs.loc[mask, :].iloc[:n_samples_valid]
        _y = to.ys.loc[mask, y_col].iloc[:n_samples_valid]
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
    def __init__(self, xs, y_true, y_pred, info:pd.DataFrame=None):
        if info is None:
            self.df = xs.loc[:,['meter', 'building_id', 'timestamp']].copy()
        else:
            assert all([v in info.columns.values for v in ['meter', 'building_id', 'timestamp']])
            self.df = xs.join(info)

        for col in ['meter', 'building_id']:
            self.df[col].cat.set_categories(sorted(self.df[col].unique()),
                                            ordered=True, inplace=True)

        self.df['y_true'] = y_true
        self.df['y_pred'] = y_pred
        self.compute_misses()

    def compute_misses(self):
        fun = lambda x: np.sqrt(np.mean(x**2))
        self.miss = (self.df.assign(difference=lambda x: x['y_pred']-x['y_true'])
                     .groupby(['building_id', 'meter'])
                     .agg(loss=pd.NamedAgg(column='difference', aggfunc=fun))
                     .dropna()
                     .sort_values('loss'))

# Cell
@patch
def plot_boldly_wrong(self:BoldlyWrongTimeseries,
                      nth_last:int=None,
                      meter:int=None, bid:int=None):

    assert (meter is not None and bid is not None) or (nth_last is not None)

    if nth_last is not None:
        ix = self.miss.iloc[[nth_last],:]
        meter = ix.index[0][1]
        bid = ix.index[0][0]
        loss = ix["loss"].values[0]
    else:
        ix = self.miss.xs((bid,meter))
        loss = ix.values[0]


    df_plot = self.df.loc[(self.df['meter']==int(meter)) & (self.df['building_id']==int(bid))]
    df_plot = pd.concat((
        df_plot[['timestamp', 'y_true']].rename(columns={'y_true':'y'}).assign(label='true'),
        df_plot[['timestamp', 'y_pred']].rename(columns={'y_pred':'y'}).assign(label='pred')),
        ignore_index=True
    )
    return df_plot.plot(kind='scatter', x='timestamp',
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