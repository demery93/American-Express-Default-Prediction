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
nfeats = '2500'

train = pd.read_feather("train_features/train.f")
labels = pd.read_csv("input/train_labels.csv")
features = np.load(f"feature_selection/top_{nfeats}_features.npy")

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


y = labels['target'].values
print(f"Validating on {len(features)} features")
train = train[features]
gc.collect()

dtrain = lgb.Dataset(train, y)

results = lgb.cv(params, dtrain, 10000, nfold=config.NFOLDS, feval=amex_metric_mod_lgbm,
                 callbacks=[lgb.early_stopping(500), lgb.log_evaluation(100)], seed=42, return_cvbooster=True)



print(f"Rounds: {len(results['amex_metric-mean'])}, CV: {results['amex_metric-mean'][-1]} + {results['amex_metric-stdv'][-1]}")
#Rounds: 9999, CV: 0.7982327096375794 + 0.0038007911677668195
