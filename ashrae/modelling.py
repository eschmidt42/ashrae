# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/03_modelling.ipynb (unless otherwise specified).

__all__ = ['evaluate_torch', 'split_dataset', 'BoldlyWrongTimeseries', 'plot_boldly_wrong', 'init_widgets',
           'run_boldly', 'click_boldly_wrong', 'Swish', 'Sine']

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

import ipywidgets as widgets

# Cell
def evaluate_torch(y_true:torch.Tensor, y_pred:torch.Tensor): return torch.sqrt(F.mse_loss(y_true, y_pred))

# Cell
def split_dataset(X:pd.DataFrame, split_kind:str='random',
                  train_frac:float=8):

    def random_split():
        n_train = int(len(X)*train_frac)
        train_bool = X.index.isin(np.random.choice(X.index.values, size=n_train, replace=False))
        return train_bool

    def time_split():
        time_col = 'timestampElapsed'
        ts = X[time_col].sort_values(ascending=True)
        ix = int(len(X)*train_frac)
        threshold_t = ts.iloc[ix:].values[0]
        return X[time_col] < threshold_t

    split_funs = {
        'random': random_split,
        'time': time_split,
    }

    assert split_kind in split_funs
    train_bool = split_funs[split_kind]()

    train_idx = np.where(train_bool)[0]
    valid_idx = np.where(~train_bool)[0]

    return (list(train_idx), list(valid_idx))

# Cell
class BoldlyWrongTimeseries:
    def __init__(self, xs, y_true, y_pred, t:pd.DataFrame=None):
        if t is None:
            self.df = xs.loc[:,['meter', 'building_id', 'timestampElapsed']].copy()
        else:
            self.df = xs.loc[:,['meter', 'building_id']].join(t)
#         self.get_predictions(model, xs)
        self.df['y_true'] = y_true
        self.df['y_pred'] = y_pred
        self.compute_misses(y_true)

#     def get_predictions(self, model, xs):
#         self.df['y_pred'] = m.predict(xs)

    def compute_misses(self, y_true):
        fun = lambda x: np.mean(x**2)
        self.miss = (self.df.assign(difference=lambda x: x['y_pred']-x['y_true'])
                     .groupby(['meter', 'building_id'])
                     .agg(loss=pd.NamedAgg(column='difference', aggfunc=fun))
                     .sort_values('loss'))

# Cell
def plot_boldly_wrong(self, nth_last:int=None,
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
def init_widgets(self):
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

def run_boldly(self):
    if not hasattr(self, 'switch_btn'):
        self.init_widgets()
    return widgets.VBox([self.switch_btn, self.int_txt_loss,
                         self.int_txt_meter, self.int_txt_bid,
                         self.run_btn, self.out_wdg])

def click_boldly_wrong(self, change):
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

BoldlyWrongTimeseries.init_widgets = init_widgets
BoldlyWrongTimeseries.click_boldly_wrong = click_boldly_wrong
BoldlyWrongTimeseries.run_boldly = run_boldly

# Cell
class Swish(nn.ReLU):
    def forward(self, input:Tensor) -> Tensor:
        if self.inplace:
            res = input.clone()
            torch.sigmoid_(res)
            input *= res
            return input
        else:
            return torch.sigmoid(input) * input

class Sine(nn.ReLU):
    def forward(self, input:Tensor) -> Tensor:
        if self.inplace:
            return torch.sin_(input)
        else:
            return torch.sin(input)