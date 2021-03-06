# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/all_meters_one_model.ipynb (unless otherwise specified).

__all__ = ['evaluate_torch', 'cnr', 'EmbeddingFeatures', 'pick_random', 'pretty_dictionary']

# Cell
from ashrae import loading, preprocessing, feature_testing

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

cnr = lambda x: x.clone().numpy().ravel() # clone numpy ravel

# Cell
class EmbeddingFeatures:
    def __init__(self, to:TabularPandas, learn:Learner):
        self.df_embs = {col: self.get_embedding_features_df(col, to, learn) for col in to.cat_names}

    def get_embedding_features_df(self, col:str, to:TabularPandas, learn:Learner): # , to:TabularPandas, learn:Learner
        ix = to.cat_names.index(col)
        w_emb = learn.model.embeds[ix].weight.cpu().detach().clone().numpy()
        df_emb = (pd.DataFrame(w_emb)
                  .add_prefix(f'{col}_embedding_'))
        df_emb.index.rename(col, inplace=True)
        return df_emb

# Cell
@patch
def replace_cat_features_with_embeddings(self:EmbeddingFeatures, X:pd.DataFrame):
    for col, df_emb in self.df_embs.items():
        X = X.join(df_emb, on=col, how='left').drop(columns=col)
    return X

# Cell
@patch
def embedding_assignment_func(self:EmbeddingFeatures, stuff:tuple):
    k, grp = stuff
    grp.drop(columns=['group'], inplace=True)
    grp = self.replace_cat_features_with_embeddings(grp)
    return pd.Series(m.predict(grp.values), index=grp.index)

@patch
def predict_with_embeddings(self:EmbeddingFeatures, X:pd.DataFrame, m,
                            num_rows:int=2_000_000, num_workers:int=1):
    tmp = X.copy()
    tmp['group'] = np.floor(tmp.index.values / num_rows).astype(int)
    n = int(np.ceil(len(tmp)/num_rows))
    y_test_pred = []

    if num_workers > 1:
        pool = Pool(processes=num_workers)
        for grp in tqdm.tqdm(pool.imap(self.embedding_assignment_func, tmp.groupby('group')), total=n):
            y_test_pred.append(grp)
        pool.join()
        pool.close()
    else:
        for stuff in tqdm.tqdm(tmp.groupby('group'), total=n):
            y_test_pred.append(self.embedding_assignment_func(stuff))

    y_test_pred = pd.concat(y_test_pred)
    display(y_test_pred.head(), y_test_pred.tail())
    return y_test_pred.sort_index()

# Cell
def pick_random(x,s:int=50): return np.random.choice(x.ravel(), size=s, replace=False)

# Cell
def pretty_dictionary(d:dict): return ', '.join(f'{k} = {v}' for k,v in d.items())