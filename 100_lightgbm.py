import pandas as pd
import numpy as np
import lightgbm as lgb

import gc
import pickle

from sklearn.model_selection import StratifiedKFold
from utils import amex_metric_mod, amex_metric_mod_lgbm
from tqdm import tqdm
from config import config

train = pd.read_feather("train_features/train.f")
test = pd.read_feather("test_features/test.f")
labels = pd.read_csv("input/train_labels.csv")
sub = pd.read_csv("input/sample_submission.csv")

features = [c for c in train.columns]

y = labels['target'].values
print(f"Training on {len(features)} features")

NFOLD = 5
params = {
    'objective': 'binary',
    'metric': 'custom',
    'learning_rate': 0.02,
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

kfold = StratifiedKFold(n_splits=NFOLD, random_state=42, shuffle=True)
oof = np.zeros(len(train))
preds = np.zeros(len(test))
# Iterate through each fold
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
                      callbacks=[lgb.log_evaluation(100), lgb.early_stopping(500)],
                      feval=amex_metric_mod_lgbm)

    with open(f"trained_models/lgb_{fold}.pkl", 'wb') as file:
        pickle.dump(model, file)
    oof[val_ind] = model.predict(x_val)
    preds += model.predict(test[features]) / NFOLD
    print(f"Fold {fold + 1} score: {amex_metric_mod(y_val, oof[val_ind])}")


print(f"CV Score: {amex_metric_mod(y, oof)}")
#0.7959406252159921
# CV Score: 0.7966693069328343

sub['prediction'] = preds
sub.to_csv("output/submission_lgb.csv", index=False, header=True)

oof_df = pd.read_csv("input/train_labels.csv")
oof_df['prediction'] = oof
oof_df[['customer_ID','prediction']].to_csv("output/validation_lgb.csv", index=False, header=True)