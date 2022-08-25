import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool, EShapCalcType, EFeaturesSelectionAlgorithm

import gc
from config import config

feature_groups = [2500, 1500, 500]
train = pd.read_feather("train_features/train.f")
#cols = [c.replace('-','_minus_') for c in train.columns]
#train.columns = cols
#train.to_feather("train_features/train.f")
labels = pd.read_csv("input/train_labels.csv")
features = [c for c in train.columns if c not in ['customer_ID','target','ind','S_2']]
print(f"Beginning features selection process with {len(features)} features")
for n_features in feature_groups:
    train = pd.read_feather("train_features/train.f")
    train = train[features]
    print(f"Selecting {n_features} features from initial {train.shape[1]}")
    cat_features = [c for c in features if c in config.cat_features]
    CATBOOST_PARAMS = dict(iterations=5000,
                           learning_rate=0.067666,
                           metric_period=100,
                           task_type='GPU',
                           od_type='Iter',
                           od_wait=50,
                           random_seed=config.seed,
                           allow_writing_files=False)

    from sklearn.model_selection import train_test_split
    y = labels.target.values
    tr_x, val_x, tr_y, val_y = train_test_split(train[features], y, test_size=config.TEST_SIZE_SPLIT, random_state=config.seed)

    del train, y
    _ = gc.collect()

    train_pool = Pool(tr_x, tr_y, cat_features=cat_features)
    val_pool = Pool(val_x, val_y, cat_features=cat_features)

    del tr_x, tr_y, val_x, val_y
    _ = gc.collect()

    clf = CatBoostClassifier(**CATBOOST_PARAMS)

    summary = clf.select_features(
        train_pool,
        eval_set=val_pool,
        features_for_select=features,
        num_features_to_select=n_features,
        steps=config.STEPS_TO_SELECT,
        algorithm=EFeaturesSelectionAlgorithm.RecursiveByLossFunctionChange,
        shap_calc_type=EShapCalcType.Regular,
        train_final_model=False,
        verbose=True,
        plot=False
    )

    cat_features = [c for c in cat_features if c not in summary['eliminated_features_names']]
    del train_pool, val_pool, clf
    _ = gc.collect()
    features = summary['selected_features_names']
    np.save(f"feature_selection/top_{n_features}_features.npy", features)