# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/03_modelling.ipynb (unless otherwise specified).

__all__ = ['evaluate_torch', 'cnr', 'split_dataset', 'Swish', 'Sine', 'pick_random']

# Cell
from ashrae import preprocessing

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

import lightgbm as lgb

import ipywidgets as widgets

# Cell
def evaluate_torch(y_true:torch.Tensor, y_pred:torch.Tensor): return torch.sqrt(F.mse_loss(y_true, y_pred))

# Cell
cnr = lambda x: x.clone().numpy().ravel() # clone numpy ravel

# Cell
def split_dataset(X:pd.DataFrame, split_kind:str='random',
                  train_frac:float=8, t_train:pd.DataFrame=None):

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

    def time_split_lina():
        time_col = 'timestampDayofyear'
        assert time_col in X.columns
        days = X[time_col].unique()
        day_values = random.choice(days, size=int(len(days) * trainfrac))
        mask = X[time_col].isin(day_values)
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
    }

    assert split_kind in split_funs
    train_bool = split_funs[split_kind]()

    train_idx = np.where(train_bool)[0]
    valid_idx = np.where(~train_bool)[0]

    return (list(train_idx), list(valid_idx))

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

# Cell
pick_random = lambda x: np.random.choice(x, size=5000, replace=False)