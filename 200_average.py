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

oof_rank = 0.6*oof1.prediction.rank(pct=True) + 0.15*oof2.prediction.rank(pct=True) + 0.25*oof3.prediction.rank(pct=True)

print(f"Rank Score Ensemble Model {amex_metric_mod(labels['target'].values, oof_rank)}")
#0.7996

sub1 = pd.read_csv("output/submission_model1.csv")
sub2 = pd.read_csv("output/submission_model2.csv")
sub3 = pd.read_csv("output/submission_model3.csv")

pred_rank = 0.6*sub1.prediction.rank(pct=True) + 0.15*sub2.prediction.rank(pct=True) + 0.25*sub3.prediction.rank(pct=True)

sub['prediction'] = pred_rank
sub.to_csv("output/submission_ensemble.csv", index=False)

public_sub = pd.read_csv("C:\\Users\\emery\\Downloads\\public.csv")
sub['prediction'] = 0.5*sub.prediction.rank(pct=True) + 0.5*public_sub.prediction.rank(pct=True)
sub.to_csv("output/submission_ensemble_public.csv", index=False)

public_sub = pd.read_csv("C:\\Users\\emery\\Downloads\\keras-cnn_sub.csv")
sub['prediction'] = 0.95*sub.prediction.rank(pct=True) + 0.05*public_sub.prediction.rank(pct=True)
sub.to_csv("output/submission_ensemble_public2.csv", index=False)