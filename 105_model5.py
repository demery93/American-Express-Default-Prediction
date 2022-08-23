import pandas as pd
import numpy as np
import lightgbm as lgb

import gc
import pickle
import sys

from sklearn.model_selection import StratifiedKFold
from utils import amex_metric_mod, amex_metric_mod_lgbm
from tqdm import tqdm
from config import config

train = pd.read_feather("train_features/train.f")
test = pd.read_feather("test_features/test.f")
labels = pd.read_csv("input/train_labels.csv")
sub = pd.read_csv("input/sample_submission.csv")
features = np.load(f"feature_selection/top_2500_features.npy")

y = labels['target'].values
train = train[features]
test = test[features]
gc.collect()

'''
[100]	training's amex_metric: 0.751771	valid_1's amex_metric: 0.753409
[200]	training's amex_metric: 0.76113	valid_1's amex_metric: 0.761548
[300]	training's amex_metric: 0.768304	valid_1's amex_metric: 0.768474
[400]	training's amex_metric: 0.773516	valid_1's amex_metric: 0.774123
[500]	training's amex_metric: 0.779303	valid_1's amex_metric: 0.778707
[600]	training's amex_metric: 0.784227	valid_1's amex_metric: 0.781059
[700]	training's amex_metric: 0.787444	valid_1's amex_metric: 0.784697
[800]	training's amex_metric: 0.789961	valid_1's amex_metric: 0.786534
[900]	training's amex_metric: 0.79219	valid_1's amex_metric: 0.78868
[1000]	training's amex_metric: 0.794307	valid_1's amex_metric: 0.790418
'''

def weighted_logloss(preds, data):
    #numerically stable sigmoid:
    y_true = data.get_label()
    preds_rank = pd.Series(preds).rank(pct=True)
    preds = 1. / (1. + np.exp(-preds))
    weights = np.where(preds_rank >= 0.96, 2, 1)

    grad = -(y_true - preds)
    hess = preds * (1.0 - preds)

    grad = grad*weights
    hess = hess*weights
    return grad, hess

params = {'objective': 'binary',
          'metric': 'custom',
          'learning_rate': 0.01,
          'max_depth': 5,
          'num_leaves': 2 ** 5 - 1,
          'max_bin': 255,
          'min_child_weight': 200,
          'colsample_bytree': 0.4,
          'subsample': 0.9,
          'nthread': 8,
          'bagging_freq': 1,
          'verbose': -1,
          'seed': 42}

kfold = StratifiedKFold(n_splits=config.NFOLDS, random_state=42, shuffle=True)
oof = np.zeros(len(train))
preds = np.zeros(len(test))
print(f"Training on {len(features)} features")
# Iterate through each fold
scores = []
for fold, (trn_ind, val_ind) in enumerate(kfold.split(train[features], y)):
    print(f'Training fold {fold + 1}')
    x_train, x_val = train[features].iloc[trn_ind], train[features].iloc[val_ind]
    y_train, y_val = y[trn_ind], y[val_ind]

    train_dataset = lgb.Dataset(x_train, y_train)
    val_dataset = lgb.Dataset(x_val, y_val)
    model = lgb.train(params=params,
                      train_set=train_dataset,
                      valid_sets=[train_dataset, val_dataset],
                      num_boost_round=10000,
                      callbacks=[lgb.log_evaluation(100), lgb.early_stopping(500)],
                      feval=amex_metric_mod_lgbm,
                      fobj=weighted_logloss)

    with open(f"trained_models/model5_{fold}.pkl", 'wb') as file:
        pickle.dump(model, file)

    val_pred = model.predict(x_val)
    test_pred = model.predict(test[features])

    val_pred = pd.Series(val_pred).rank(pct=True)
    test_pred = pd.Series(test_pred).rank(pct=True)

    oof[val_ind] = val_pred
    preds += test_pred / config.NFOLDS

    score = amex_metric_mod(y_val, oof[val_ind])
    scores.append(score)
    print(f"Fold {fold + 1} score: {score}")

print(f"Average CV Score {np.mean(scores)}")
print(f"Full CV Score: {amex_metric_mod(y, oof)}")
#Average CV Score 0.7982275708417814
#Full CV Score: 0.7978578429144019


sub['prediction'] = preds
sub.to_csv(f"output/submission_model5.csv", index=False, header=True)

oof_df = pd.read_csv("input/train_labels.csv")
oof_df['prediction'] = oof
oof_df[['customer_ID','prediction']].to_csv(f"output/validation_model5.csv", index=False, header=True)