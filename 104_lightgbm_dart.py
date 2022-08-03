import pandas as pd
import numpy as np
import lightgbm as lgb

import gc
import pickle
import joblib
import os

from sklearn.model_selection import StratifiedKFold
from utils import amex_metric_mod, amex_metric_mod_lgbm, SaveModelCallback
from tqdm import tqdm

train = pd.read_feather("train_features/train.f")
test = pd.read_feather("test_features/test.f")
labels = pd.read_csv("input/train_labels.csv")
sub = pd.read_csv("input/sample_submission.csv")

features = [c for c in train.columns]

y = labels['target'].values

NFOLD = 5
params = {
    'objective': 'binary',
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

kfold = StratifiedKFold(n_splits=NFOLD, random_state=42, shuffle=True)
oof = np.zeros(len(train))
preds = np.zeros(len(test))

# Iterate through each fold
for fold, (trn_ind, val_ind) in enumerate(kfold.split(train[features], y)):
    print(f'Training fold {fold + 1}')
    x_train, x_val = train[features].iloc[trn_ind], train[features].iloc[val_ind]
    y_train, y_val = y[trn_ind], y[val_ind]

    train_dataset = lgb.Dataset(x_train, y_train)
    val_dataset = lgb.Dataset(x_val, y_val)
    model = lgb.train(params=params,
                      train_set=train_dataset,
                      valid_sets=[train_dataset, val_dataset],
                      num_boost_round=8000,
                      callbacks=[lgb.log_evaluation(100)],
                      feval=amex_metric_mod_lgbm)

    with open(f"trained_models/lgb_dart_{fold}.pkl", 'wb') as file:
        pickle.dump(model, file)
    oof[val_ind] = model.predict(x_val)
    preds += model.predict(test[features]) / NFOLD

    print(f"Fold {fold+1} Score: {amex_metric_mod(y[val_ind], oof[val_ind])}")


print(f"CV Score: {amex_metric_mod(y, oof)}")
# CV Score: 0.7981753843222155
sub = pd.read_csv("input/sample_submission.csv")
sub['prediction'] = preds
sub.to_csv("output/submission_lgb_dart.csv", index=False, header=True)

oof_df = pd.read_csv("input/train_labels.csv")
oof_df['prediction'] = oof
oof_df[['customer_ID','prediction']].to_csv("output/validation_lgb_dart.csv", index=False, header=True)