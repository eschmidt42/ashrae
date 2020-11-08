# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/feature_testing.ipynb (unless otherwise specified).

__all__ = ['get_tabular_object', 'train_predict', 'SPLIT_PARAMS', 'hist_plot_preds', 'BoldlyWrongTimeseries']

# Cell
from loguru import logger
from fastai.tabular.all import *
from ashrae import loading, preprocessing, inspection
from sklearn import linear_model, tree, model_selection, ensemble
import tqdm
import plotly.express as px

import pandas as pd
import ipywidgets as widgets

# Cell
def get_tabular_object(df:pd.DataFrame, var_names:dict,
                       splits=None, procs:list=None):
    if procs is None: procs = []
    return TabularPandas(df.copy(), procs,
                         var_names['cats'], var_names['conts'],
                         y_names=var_names['dep_var'],
                         splits=splits)

SPLIT_PARAMS = dict(
    train_frac = .8,
    split_kind = 'time_split_day',
)


def train_predict(df:pd.DataFrame, var_names:dict,
                  model, model_params:dict=None, n_rep:int=3,
                  n_samples_train:int=10_000,
                  n_samples_valid:int=10_000,
                  procs:list=[Categorify, FillMissing, Normalize],
                  split_params:dict=None):

    split_params = SPLIT_PARAMS if split_params is None else split_params
    y_col = var_names['dep_var']
    score_vals = []
    model_params = {} if model_params is None else model_params

    to = get_tabular_object(df, var_names, procs=procs)

    for i in tqdm.tqdm(range(n_rep), total=n_rep, desc='Repetition'):
        m = model(**model_params)
        splits = preprocessing.split_dataset(df, **split_params)

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
            self.df[col] = self.df[col].astype('category')
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