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
nfeats = '2500'

train = pd.read_feather("train_features/train.f")
labels = pd.read_csv("input/train_labels.csv")
features = np.load(f"feature_selection/top_{nfeats}_features.npy")

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
          'seed': 42}

#0.796163 - 1301 rounds

y = labels['target'].values
print(f"Validating on {len(features)} features")
train = train[features]
gc.collect()

dtrain = lgb.Dataset(train, y)

results = lgb.cv(params, dtrain, 9999, nfold=config.NFOLDS, feval=amex_metric_mod_lgbm, fobj=fl.lgb_obj,
                 callbacks=[lgb.early_stopping(500), lgb.log_evaluation(100)], seed=42)

results.keys()
print(f"Rounds: {len(results['amex_metric-mean'])}, CV: {results['amex_metric-mean'][-1]} + {results['amex_metric-stdv'][-1]}")
#Rounds: 5661, CV: 0.7976211067941789 + 0.003353433966261368
