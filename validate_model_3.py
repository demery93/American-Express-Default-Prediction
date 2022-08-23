import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

import gc
import pickle

from sklearn.model_selection import StratifiedKFold
from utils import amex_metric_mod, amex_metric_mod_lgbm
from tqdm import tqdm
from config import config


nfeats = 1500
print(nfeats)

train = pd.read_feather("train_features/train.f")
test = pd.read_feather("test_features/test.f")

labels = pd.read_csv("input/train_labels.csv")
sub = pd.read_csv("input/sample_submission.csv")

features = np.load(f"feature_selection/top_{nfeats}_features.npy")

cat_features = [c for c in features if c in config.cat_features]
y = labels['target'].values

NFOLD = 5

kfold = StratifiedKFold(n_splits=NFOLD, random_state=42, shuffle=True)
oof = np.zeros(len(train))
preds = np.zeros(len(test))
# Iterate through each fold
scores = []
for fold, (trn_ind, val_ind) in enumerate(kfold.split(train[features], y)):
    print(f"******* Fold {fold+1} ******* ")
    x_train, x_val = train[features].iloc[trn_ind], train[features].iloc[val_ind]
    y_train, y_val = y[trn_ind], y[val_ind]

    clf = CatBoostClassifier(iterations=10000, random_state=22, task_type = 'GPU', max_depth=8, min_data_in_leaf=8)
    clf.fit(x_train, y_train, eval_set=[(x_val, y_val)], cat_features=cat_features,  verbose=100)

    val_pred = clf.predict_proba(x_val)[:, 1]
    test_pred = clf.predict_proba(test[features])[:, 1]

    val_pred = pd.Series(val_pred).rank(pct=True)
    test_pred = pd.Series(test_pred).rank(pct=True)

    oof[val_ind] = val_pred
    preds += test_pred / NFOLD
    with open(f"trained_models/model3_{fold}.pkl", 'wb') as file:
        pickle.dump(clf, file)

    score = amex_metric_mod(y_val, oof[val_ind])
    scores.append(score)
    print(f"Fold {fold + 1} score: {score}")

print(f"Average CV Score {np.mean(scores)}")
print(f"CV Score: {amex_metric_mod(y, oof)}")

sub = pd.read_csv("input/sample_submission.csv")
sub['prediction'] = preds
sub.to_csv(f"output/submission_catboost_{nfeats}.csv", index=False, header=True)

oof_df = pd.read_csv("input/train_labels.csv")
oof_df['prediction'] = oof
oof_df[['customer_ID','prediction']].to_csv(f"output/validation_catboost_{nfeats}.csv", index=False, header=True)
#CV Score: 0.7971233830844968