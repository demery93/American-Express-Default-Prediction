import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import gc
import pickle

from sklearn.model_selection import StratifiedKFold, GroupKFold
from utils import amex_metric_mod, amex_metric_mod_lgbm
from config import config

train = pd.read_feather("train_features/train.f")
test = pd.read_feather("test_features/test.f")
features = [c for c in train.columns]
labels = pd.read_csv("input/train_labels.csv")
y = labels['target'].values

train = train.replace(np.inf, 0)
test = test.replace(np.inf, 0)
train = train.replace(-np.inf, 0)
test = test.replace(-np.inf, 0)

params = {
    'objective': 'binary:logistic',
    'tree_method': 'gpu_hist',
    'max_depth': 7,
    'subsample': 0.88,
    'colsample_bytree': 0.5,
    'gamma': 1.5,
    'min_child_weight': 8,
    'lambda': 70,
    'eta': 0.03,
}

kfold = StratifiedKFold(n_splits=config.NFOLDS, random_state=42, shuffle=True)
y = labels.target.values
oof = np.zeros(len(train))
preds = np.zeros(len(test))
# Iterate through each fold

def xgb_amex(y_pred, y_true):
    return 'amex', amex_metric_np(y_pred,y_true.get_label())

def amex_metric_np(preds: np.ndarray, target: np.ndarray) -> float:
    indices = np.argsort(preds)[::-1]
    preds, target = preds[indices], target[indices]

    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_mask = cum_norm_weight <= 0.04
    d = np.sum(target[four_pct_mask]) / np.sum(target)

    weighted_target = target * weight
    lorentz = (weighted_target / weighted_target.sum()).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    n_pos = np.sum(target)
    n_neg = target.shape[0] - n_pos
    gini_max = 10 * n_neg * (n_pos + 20 * n_neg - 19) / (n_pos + 20 * n_neg)

    g = gini / gini_max
    return 0.5 * (g + d)


for fold, (trn_ind, val_ind) in enumerate(kfold.split(train[features], y)):
    print(f'Training fold {fold + 1}')
    x_train, x_val = train[features].iloc[trn_ind], train[features].iloc[val_ind]
    y_train, y_val = y[trn_ind], y[val_ind]

    dtrain = xgb.DMatrix(data=x_train, label=y_train)
    dvalid = xgb.DMatrix(data=x_val, label=y_val)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    bst = xgb.train(params, dtrain=dtrain,
                    num_boost_round=2600, evals=watchlist,
                    early_stopping_rounds=500, feval=xgb_amex, maximize=True,
                    verbose_eval=100)

    oof[val_ind] = bst.predict(dvalid, iteration_range=(0,bst.best_ntree_limit))
    print(f"Fold {fold+1} score: {amex_metric_mod(y_val, oof[val_ind])}")
    preds += bst.predict(xgb.DMatrix(test[features]), iteration_range=(0,bst.best_ntree_limit)) / config.NFOLDS
    with open(f"trained_models/xgb_{fold}.pkl", 'wb') as file:
        pickle.dump(bst, file)

print(f"CV Score: {amex_metric_mod(y, oof)}")
sub = pd.read_csv("input/sample_submission.csv")
sub['prediction'] = preds
sub.to_csv("output/submission_xgb.csv", index=False, header=True)

oof_df = pd.read_csv("input/train_labels.csv")
oof_df['prediction'] = oof
oof_df[['customer_ID','prediction']].to_csv("output/validation_xgb.csv", index=False, header=True)
# CV Score: 0.7966598373335434