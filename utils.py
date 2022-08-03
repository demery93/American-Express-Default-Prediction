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

def to_feature(df, path):
    if df.columns.duplicated().sum() > 0:
        raise Exception(f'duplicated!: {df.columns[df.columns.duplicated()]}')
    df.reset_index(inplace=True, drop=True)
    df.columns = [c.replace('/', '-').replace(' ', '-') for c in df.columns]
    for c in df.columns:
        df[[c]].to_feather(f'{path}_{c}.f')
    return

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

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

class SaveModelCallback:
    def __init__(self,
                 models_folder: pathlib.Path,
                 fold_id: int,
                 min_score_to_save: float,
                 every_k: int,
                 order: int = 0):
        self.min_score_to_save: float = min_score_to_save
        self.every_k: int = every_k
        self.current_score = min_score_to_save
        self.order: int = order
        self.models_folder: pathlib.Path = models_folder
        self.fold_id: int = fold_id

    def __call__(self, env):
        iteration = env.iteration
        score = env.evaluation_result_list[3][2]
        if iteration % self.every_k == 0:
            print(f'iteration {iteration}, score={score:.05f}')
            if score > self.current_score:
                self.current_score = score
                for fname in self.models_folder.glob(f'fold_id_{self.fold_id}*'):
                    fname.unlink()
                print(f'High Score: iteration {iteration}, score={score:.05f}')
                joblib.dump(env.model, self.models_folder / f'fold_id_{self.fold_id}_{score:.05f}.pkl')


def save_model(models_folder: pathlib.Path, fold_id: int, min_score_to_save: float = 0.78, every_k: int = 50):
    return SaveModelCallback(models_folder=models_folder, fold_id=fold_id, min_score_to_save=min_score_to_save, every_k=every_k)