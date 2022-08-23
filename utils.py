from contextlib import contextmanager
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import pickle
import joblib
import pathlib
from scipy import optimize
from scipy import special

class FocalLoss:

    def __init__(self, gamma, alpha=None):
        self.alpha = alpha
        self.gamma = gamma

    def at(self, y):
        if self.alpha is None:
            return np.ones_like(y)
        return np.where(y, self.alpha, 1 - self.alpha)

    def pt(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.where(y, p, 1 - p)

    def __call__(self, y_true, y_pred):
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        return -at * (1 - pt) ** self.gamma * np.log(pt)

    def grad(self, y_true, y_pred):
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma
        return at * y * (1 - pt) ** g * (g * pt * np.log(pt) + pt - 1)

    def hess(self, y_true, y_pred):
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma

        u = at * y * (1 - pt) ** g
        du = -at * y * g * (1 - pt) ** (g - 1)
        v = g * pt * np.log(pt) + pt - 1
        dv = g * np.log(pt) + g + 1

        return (du * v + u * dv) * y * (pt * (1 - pt))

    def init_score(self, y_true):
        res = optimize.minimize_scalar(
            lambda p: self(y_true, p).sum(),
            bounds=(0, 1),
            method='bounded'
        )
        p = res.x
        log_odds = np.log(p / (1 - p))
        return log_odds

    def lgb_obj(self, preds, train_data):
        y = train_data.get_label()
        p = special.expit(preds)
        return self.grad(y, p), self.hess(y, p)

    def lgb_eval(self, preds, train_data):
        y = train_data.get_label()
        p = special.expit(preds)
        is_higher_better = False
        return 'focal_loss', self(y, p).mean(), is_higher_better

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


# Display/plot feature importance
def display_importances(feature_importance_df_, score):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title(f'LightGBM Features (avg over folds): {score}')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')

def amex_metric_mod(y_true, y_pred):

    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:,0]==0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:,0]==0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] *  weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1]/gini[0] + top_four)

def amex_metric_mod_lgbm(y_pred: np.ndarray, data: lgb.Dataset):

    y_true = data.get_label()
    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:,0]==0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:,0]==0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] *  weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)

    return 'amex_metric', 0.5 * (gini[1]/gini[0]+ top_four), True

def cv_check(df, features):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.1,
        'max_depth': 6,
        'num_leaves': 63,
        'max_bin': 255,
        'min_child_weight': 8,
        'reg_lambda': 70,  # L2 regularization term on weights.
        'colsample_bytree': 0.6,
        'subsample': 0.9,
        'nthread': 8,
        'bagging_freq': 1,
        'verbose': -1,
        'seed': 42
    }

    X = df[features]
    y = df['target'].values

    dtrain = lgb.Dataset(X, y)

    results = lgb.cv(params, dtrain, 9999, nfold=5, feval=amex_metric_mod_lgbm, callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)], seed=42)

    return results

def floorify(x, lo):
    """example: x in [0, 0.01] -> x := 0"""
    return lo if x <= lo+0.01 and x >= lo else x

def floorify_zeros(x):
    """look around values [0,0.01] and determine if in proximity it's categorical. If yes - floorify"""
    has_zeros = len([t for t in x if t>=0 and t<=0.01])>0
    no_proximity = len([t for t in x if t<0 and t>=-0.01])==0 and len([t for t in x if t>0.01 and t<=0.02])==0
    if not no_proximity:
        return x
    if not has_zeros:
        return x
    x = [floorify(t, 0.0) for t in x]
    return x

def floorify_ones(x):
    """look around values [1,1.01] and determine if in proximity it's categorical. If yes - floorify"""
    has_ones = len([t for t in x if t>=1 and t<=1.01])>0
    no_proximity = len([t for t in x if t<1 and t>=0.99])==0 and len([t for t in x if t>1.01 and t<=1.02])==0
    if not no_proximity:
        return x
    if not has_ones:
        return x
    x = [floorify(t, 1.0) for t in x]
    return x

def convert_na(x):
    """nan -> -1 if positive values"""
    if np.nanmin(x)>=0:
        return [-1 if np.isnan(t) else t for t in x]

def convert_to_int(x):
    """float -> int8 if possible"""
    q = convert_na(x)
    if set(np.unique(q)).union({-1,0,1}) == {-1,0,1}:
        return [np.int8(t) for t in q]
    return x

def floorify_ones_and_zeros(t):
    """do everything"""
    t = floorify_zeros(t)
    t = floorify_ones(t)
    t = convert_to_int(t)
    return t

def floorify_frac(x, interval=1):
    """convert to int if float appears ordinal"""
    xt = (np.floor(x/interval+1e-6)).fillna(-1)
    if np.max(xt)<=127:
        return xt.astype(np.int8)
    return xt.astype(np.int16)