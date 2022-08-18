import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import StratifiedKFold

import gc
import pickle

import time
import lightgbm as lgb
print('LGB Version',lgb.__version__)

from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm


def amex_metric(y_true, y_pred):

    labels     = np.transpose(np.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = np.where(labels[:,0]==0, 20, 1)
    cut_vals   = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 20, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1]/gini[0] + top_four)

train = pd.read_feather("train_features/train.f")
labels = pd.read_csv("input/train_labels.csv")

importances = []


train['target'] = labels.target.values
train_idx, valid_idx = train_test_split(train.index, test_size=0.2, shuffle=True, random_state=42)
print('#' * 25)
print('### Train Records', len(train_idx))
print('### Validation Records', len(valid_idx))
print('#' * 25)

features = [c for c in train.columns if c not in ['target','customer_ID']]

x_train, x_val = train.loc[train_idx, features], train.loc[valid_idx, features]
y_train, y_val = train.loc[train_idx, 'target'].values, train.loc[valid_idx, 'target']

dtrain = lgb.Dataset(x_train, y_train)
dvalid = lgb.Dataset(x_val, y_val)

lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.05,
        'max_depth': 5,
        'num_leaves': 2 ** 5 - 1,
        'max_bin': 255,
        'min_child_weight': 24,
        'reg_lambda': 80,  # L2 regularization term on weights.
        'colsample_bytree': 0.3,
        'subsample': 0.9,
        'nthread': 8,
        'bagging_freq': 1,
        'verbose': -1,
        'seed': 42
}

# TRAIN MODEL FOLD K
model = lgb.train(params=lgb_params,
                  train_set=dtrain,
                  valid_sets=[dtrain, dvalid],
                  num_boost_round=10000,
                  callbacks=[lgb.log_evaluation(100), lgb.early_stopping(100)])

# INFER OOF FOLD K
oof_preds = model.predict(train.loc[valid_idx, features])
acc = amex_metric(y_val, oof_preds)
print('Kaggle Metric =', acc, '\n')
#0.7979752781408805

results = []
print(' Computing Permutation feature importance...')

# COMPUTE BASELINE (NO SHUFFLE)
oof_preds = model.predict(x_val)
baseline_acc = amex_metric(y_val, oof_preds)

for k in tqdm(range(len(features))):
    # SHUFFLE FEATURE K
    save_col = x_val.iloc[:, k].copy()
    x_val.iloc[:, k] = np.random.permutation(x_val.iloc[:, k])

    # COMPUTE OOF MAE WITH FEATURE K SHUFFLED
    oof_preds = model.predict(x_val)
    acc = amex_metric(y_val.values, oof_preds)
    results.append({'feature': features[k], 'metric': acc})
    x_val.iloc[:, k] = save_col

print()
results = pd.DataFrame(results)
results = results.sort_values('metric', ascending=False)
results['metric_difference'] = baseline_acc - results['metric']

keepcols = results.loc[results['metric_difference'] >= KEEP_THRESH, 'feature'].values.tolist()
dropcols = results.loc[results['metric_difference'] < KEEP_THRESH, 'feature'].values.tolist()

features = features + keepcols

print('#' * 25)
print('### Feature Group', group)
print(f'### Keeping {len(keepcols)} Features from Feature Group {group}')
print(f'### Dropping {len(dropcols)} Features from Feature Group {group}')
print('#' * 25)

del dtrain, Xy_train, dd, X_valid, y_valid, dvalid, model, keepcols, dropcols, results

_ = gc.collect()