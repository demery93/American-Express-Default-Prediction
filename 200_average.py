import pandas as pd
import os
from utils import amex_metric_mod


oof1 = pd.read_csv("output/validation_xgb.csv")
oof2 = pd.read_csv("output/validation_lgb.csv")
oof3 = pd.read_csv("output/validation_catboost.csv")
oof4 = pd.read_csv("output/validation_mlp.csv")
oof5 = pd.read_csv("output/validation_lgb_dart.csv")

labels = pd.read_csv("input/train_labels.csv")

print(f"CV Score XGB Model {amex_metric_mod(labels['target'].values, oof1['prediction'].values)}")
print(f"CV Score LightGBM Model {amex_metric_mod(labels['target'].values, oof2['prediction'].values)}")
print(f"CV Score Catboost Model {amex_metric_mod(labels['target'].values, oof3['prediction'].values)}")
print(f"CV Score MLP Model {amex_metric_mod(labels['target'].values, oof4['prediction'].values)}")
print(f"CV Score DART Model {amex_metric_mod(labels['target'].values, oof5['prediction'].values)}")

oof = (oof1.prediction.values + oof2.prediction.values + oof3.prediction.values + oof4.prediction.values + oof5.prediction.values)/5
oof_rank = (oof1.prediction.rank(pct=True).values + oof2.prediction.rank(pct=True).values + oof3.prediction.rank(pct=True).values + oof4.prediction.rank(pct=True).values + oof5.prediction.rank(pct=True).values)/5
print(f"CV Score Ensemble Model {amex_metric_mod(labels['target'].values, oof)}")
print(f"CV Score Ensemble Model {amex_metric_mod(labels['target'].values, oof_rank)}")
#0.7995


sub1 = pd.read_csv("output/submission_xgb.csv")
sub2 = pd.read_csv("output/submission_lgb.csv")
sub3 = pd.read_csv("output/submission_catboost.csv")
sub4 = pd.read_csv("output/submission_mlp.csv")
sub5 = pd.read_csv("output/submission_lgb_dart.csv")

sub = sub2.copy()
sub['prediction'] = (sub1.prediction.values + sub2.prediction.values + sub3.prediction.values + sub4.prediction.values + sub5.prediction.values)/5
sub['prediction'] = (sub1.prediction.rank(pct=True).values + sub2.prediction.rank(pct=True).values + sub3.prediction.rank(pct=True).values
                     + sub4.prediction.rank(pct=True).values + sub5.prediction.rank(pct=True).values)/5
sub.to_csv("output/submission_ensemble.csv", index=False, header=True)