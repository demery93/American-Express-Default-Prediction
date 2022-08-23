import pandas as pd
import numpy as np
import os
from utils import amex_metric_mod


labels = pd.read_csv("input/train_labels.csv")
sub = pd.read_csv("input/sample_submission.csv")

oof_rank = np.zeros(len(labels))
pred_rank = np.zeros(len(sub))
oof1 = pd.read_csv("output/validation_model1.csv")
oof2 = pd.read_csv("output/validation_model2.csv")
oof3 = pd.read_csv("output/validation_model3.csv")

for df in [oof1, oof2, oof3]:
    oof_rank += df['prediction'].rank(pct=True)/3

print(f"Rank Score Ensemble Model {amex_metric_mod(labels['target'].values, oof_rank)}")
#0.7990

sub1 = pd.read_csv("output/submission_model1.csv")
sub2 = pd.read_csv("output/submission_model2.csv")
sub3 = pd.read_csv("output/submission_model3.csv")

for df in [sub1, sub2, sub3]:
    pred_rank += df['prediction'].rank(pct=True)/3

sub['prediction'] = pred_rank
sub.to_csv("output/submission_ensemble.csv", index=False)

public_sub = pd.read_csv("C:\\Users\\emery\\Downloads\\public.csv")
sub['prediction'] = 0.4*sub.prediction.rank(pct=True) + 0.6*public_sub.prediction.rank(pct=True)
sub.to_csv("output/submission_ensemble_public.csv", index=False)