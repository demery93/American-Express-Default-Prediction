'''
Model 1 - LightGBM Model with DART Booster
Features Used - 2,500
CV Score:
'''
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

params = {'objective': 'binary',
          'metric': "amex_metric",
          'boosting': 'dart',
          'seed': 42,
          'num_leaves': 100,
          'learning_rate': 0.01,
          'feature_fraction': 0.20,
          'bagging_freq': 10,
          'bagging_fraction': 0.50,
          'n_jobs': -1,
          'lambda_l2': 2,
          'min_data_in_leaf': 40
          }

kfold = StratifiedKFold(n_splits=config.NFOLDS, random_state=42, shuffle=True)
oof = np.zeros(len(train))
preds = np.zeros(len(test))
# Iterate through each fold
scores = []
print(f"Training on {len(features)} features")
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
                      callbacks=[lgb.log_evaluation(100)],
                      feval=amex_metric_mod_lgbm)

    with open(f"trained_models/model1_{fold}.pkl", 'wb') as file:
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
#Average CV Score 0.7982157767542152
#Full CV Score: 0.7983087782473084

sub['prediction'] = preds
sub.to_csv(f"output/submission_model1.csv", index=False, header=True)

oof_df = pd.read_csv("input/train_labels.csv")
oof_df['prediction'] = oof
oof_df[['customer_ID','prediction']].to_csv(f"output/validation_model1.csv", index=False, header=True)