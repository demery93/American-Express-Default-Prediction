import pandas as pd
import numpy as np
from config import config
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool, EShapCalcType, EFeaturesSelectionAlgorithm

import gc

def generate_features(df, features, feature_type='last', recent=0, prefix=""):
    aggs = {}
    for c in features:
        aggs[c] = feature_type

    if(recent>0):
        df_agg = df[df.ind >= recent].groupby('customer_ID').agg(aggs)

    else:
        df_agg = df.groupby('customer_ID').agg(aggs)

    cols = [f"{c}_{prefix}{feature_type}" for c in features]
    df_agg.columns = cols
    return df_agg


def generate_difference_ratio_features(df, features, numerator='last', denominator='first', type='difference'):
    num_aggs, den_aggs = {}, {}
    for c in features:
        num_aggs[c] = numerator
        if(denominator == 'lag'):
            den_aggs[c] = 'last'
        else:
            den_aggs[c] = denominator

    df_agg_numerator = df.groupby('customer_ID', dropna=False).agg(num_aggs)
    if(denominator == 'lag'):
        df_agg_denominator = df[df.ind < 13].groupby('customer_ID', dropna=False).agg(den_aggs)
    else:
        df_agg_denominator = df.groupby('customer_ID', dropna=False).agg(den_aggs)
    cols = [f"{c}_{numerator}_{denominator}_{type}" for c in features]
    df_agg_numerator.columns = cols
    df_agg_denominator.columns = cols
    if(type=='difference'):
        df_agg = df_agg_numerator - df_agg_denominator
    elif(type=='ratio'):
        df_agg = df_agg_numerator / df_agg_denominator
    else:
        return None
    return df_agg

def feature_selection(df, y, features, n, cat_features=None):
    CATBOOST_PARAMS = dict(iterations=5000,
                           learning_rate=0.067666,
                           metric_period=100,
                           task_type='GPU',
                           od_type='Iter',
                           od_wait=20,
                           random_seed=config.seed,
                           allow_writing_files=False)

    tr_x, val_x, tr_y, val_y = train_test_split(df[features], y, test_size=config.TEST_SIZE_SPLIT, random_state=config.seed)

    train_pool = Pool(tr_x, tr_y, cat_features=cat_features)
    val_pool = Pool(val_x, val_y, cat_features=cat_features)

    del tr_x, tr_y, val_x, val_y
    _ = gc.collect()

    clf = CatBoostClassifier(**CATBOOST_PARAMS)

    summary = clf.select_features(
        train_pool,
        eval_set=val_pool,
        features_for_select=features,
        num_features_to_select=n,
        steps=config.STEPS_TO_SELECT,
        algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
        shap_calc_type=EShapCalcType.Regular,
        train_final_model=False,
        verbose=True,
        plot=False
    )

    del train_pool, val_pool, clf
    _ = gc.collect()
    features = summary['selected_features_names']
    return features