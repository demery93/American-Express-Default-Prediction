import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from utils import amex_metric_mod
from sklearn.linear_model import LogisticRegression

oof1 = pd.read_csv("output/validation_xgb.csv")
oof2 = pd.read_csv("output/validation_lgb.csv")
oof3 = pd.read_csv("output/validation_catboost.csv")
oof4 = pd.read_csv("output/validation_mlp.csv")
oof5 = pd.read_csv("output/validation_lgb_dart.csv")
oof6 = pd.read_csv("output/validation_lgb_goss.csv")
labels = pd.read_csv("input/train_labels.csv")

sub1 = pd.read_csv("output/submission_xgb.csv")
sub2 = pd.read_csv("output/submission_lgb.csv")
sub3 = pd.read_csv("output/submission_catboost.csv")
sub4 = pd.read_csv("output/submission_mlp.csv")
sub5 = pd.read_csv("output/submission_lgb_dart.csv")
sub6 = pd.read_csv("output/submission_lgb_goss.csv")


train = np.concatenate([oof1[['prediction']].values,oof2[['prediction']].values,oof3[['prediction']].values,oof4[['prediction']].values,
                        oof5[['prediction']].values, oof6[['prediction']].values], axis=1)
test = np.concatenate([sub1[['prediction']].values,sub2[['prediction']].values,sub3[['prediction']].values,sub4[['prediction']].values,
                       sub5[['prediction']].values, sub6[['prediction']].values], axis=1)
y = labels.target.values

NFOLD = 5
kfold = StratifiedKFold(n_splits=NFOLD, random_state=42, shuffle=True)
oof = np.zeros(len(train))
preds = np.zeros(len(test))
# Iterate through each fold
for fold, (trn_ind, val_ind) in enumerate(kfold.split(train, y)):
    print(f'Training fold {fold + 1}')
    x_train, x_val = train[trn_ind], train[val_ind]
    y_train, y_val = y[trn_ind], y[val_ind]

    clf = LogisticRegression(fit_intercept=True, C=1)
    clf.fit(x_train, y_train)
    oof[val_ind] = clf.predict_proba(x_val)[:,1]
    print(f"Fold {fold+1} score: {amex_metric_mod(y_val, oof[val_ind])}")
    preds += clf.predict_proba(test)[:,1] / NFOLD

print(amex_metric_mod(y, oof))
# 0.7996924368172628
sub = sub2.copy()
sub['prediction'] = preds
sub.to_csv("output/submission_ridge.csv", index=False, header=True)
