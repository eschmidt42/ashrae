# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/03_modelling.ipynb (unless otherwise specified).

__all__ = ['evaluate_torch', 'split_dataset', 'Swish', 'Sine']

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