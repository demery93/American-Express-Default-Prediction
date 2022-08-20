'''
Model 3 - LightGBM with Focal Loss
Features Used - 1,500
CV Score: 0.7976
# Rounds: 6000
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
from utils import FocalLoss

fl = FocalLoss(alpha=.3, gamma=2)

#nfeats = sys.argv[1]
nfeats = '1500'

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

preds = np.zeros((len(test), config.NBAG))
# Iterate through each fold
scores = []
print(f"Training on {len(features)} features")
for i in range(config.NBAG):
    params = {'objective': 'binary',
              'metric': 'custom',
              'learning_rate': 0.01,
              'max_depth': 8,
              'num_leaves': 2 ** 8 - 1,
              'max_bin': 255,
              'min_child_weight': 10,
              'reg_lambda': 60,  # L2 regularization term on weights.
              'colsample_bytree': 0.5,
              'subsample': 0.9,
              'nthread': 8,
              'bagging_freq': 1,
              'verbose': -1,
              'seed': 42+i}
    print(f'Training model {i + 1}')

    train_dataset = lgb.Dataset(train[features], y)
    model = lgb.train(params=params,
                      train_set=train_dataset,
                      num_boost_round=6000,
                      fobj=fl.lgb_obj)

    with open(f"trained_models/lgb_focal_{i}.pkl", 'wb') as file:
        pickle.dump(model, file)

    preds[:,i] = model.predict(test[features])


pd.DataFrame(preds, columns=[f'model2_pred{i}' for i in range(config.NBAG)]).to_csv("output/model2_submission_expanded.csv", index=False, header=True)

sub['prediction'] = np.mean(preds, axis=1)
sub.to_csv(f"output/model2_submission.csv", index=False, header=True)