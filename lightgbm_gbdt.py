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

#nfeats = sys.argv[1]
nfeats = 2500
print(nfeats)

train = pd.read_feather("train_features/train.f")
test = pd.read_feather("test_features/test.f")

labels = pd.read_csv("input/train_labels.csv")
sub = pd.read_csv("input/sample_submission.csv")

features = np.load(f"feature_selection/top_{nfeats}_features.npy")

y = labels['target'].values
print(f"Training on {len(features)} features")
train = train[features]
test = test[features]
gc.collect()
NFOLD = config.NFOLDS
params = config.lightgbm_gbdt_params[str(nfeats)]
kfold = StratifiedKFold(n_splits=NFOLD, random_state=42, shuffle=True)
oof = np.zeros(len(train))
preds = np.zeros(len(test))
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
                      feval=amex_metric_mod_lgbm)

    with open(f"trained_models/lgb_{fold}_{nfeats}.pkl", 'wb') as file:
        pickle.dump(model, file)
    oof[val_ind] = model.predict(x_val)
    preds += model.predict(test[features]) / NFOLD
    score = amex_metric_mod(y_val, oof[val_ind])
    scores.append(score)
    print(f"Fold {fold + 1} score: {score}")


print(f"Average CV Score {np.mean(scores)}")
print(f"Full CV Score: {amex_metric_mod(y, oof)}")
#Average CV Score 0.7982275708417814
#Full CV Score: 0.7978578429144019

sub['prediction'] = preds
sub.to_csv(f"output/submission_lgb_{nfeats}.csv", index=False, header=True)

oof_df = pd.read_csv("input/train_labels.csv")
oof_df['prediction'] = oof
oof_df[['customer_ID','prediction']].to_csv(f"output/validation_lgb_{nfeats}.csv", index=False, header=True)