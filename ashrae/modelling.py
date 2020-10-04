# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/03_modelling.ipynb (unless otherwise specified).

__all__ = ['evaluate_torch', 'cnr', 'Swish', 'Sine', 'pick_random']

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