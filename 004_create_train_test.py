import pandas as pd
import numpy as np

from feature_generator import generate_features, generate_difference_ratio_features, feature_selection
from config import config
from tqdm import tqdm

import gc

def generate_train_test(df, cols, categoricals, numerical):

    df_diff = df.copy()
    df_diff['ind'] = df_diff['ind'] + 1
    df_diff = df[['customer_ID','ind']].merge(df_diff, on=['customer_ID','ind'], how='left')
    df_diff[numerical] = df[numerical] - df_diff[numerical]

    last_features = generate_features(df, cols, feature_type='last', prefix="")
    mean_features = generate_features(df, numerical, feature_type='mean', prefix="")
    min_features = generate_features(df, numerical, feature_type='min', prefix="")
    max_features = generate_features(df, numerical, feature_type='max', prefix="")
    std_features = generate_features(df, numerical, feature_type='std', prefix="")
    count_features = generate_features(df, categoricals, feature_type='count', prefix="")
    unique_features = generate_features(df, categoricals, feature_type='nunique', prefix="")
    first_features = generate_features(df, cols, feature_type='first', prefix="")
    recent_mean_features = generate_features(df, numerical, feature_type='mean', prefix="recent", recent=10)
    recent_min_features = generate_features(df, numerical, feature_type='min', prefix="recent", recent=10)
    recent_max_features = generate_features(df, numerical, feature_type='max', prefix="recent", recent=10)
    recent_std_features = generate_features(df, numerical, feature_type='std', prefix="recent", recent=10)
    first_last_diff_features = generate_difference_ratio_features(df, numerical, numerator='last', denominator='first', type='difference')
    lag_last_diff_features = generate_difference_ratio_features(df, numerical, numerator='last', denominator='lag', type='difference')
    mean_last_diff_features = generate_difference_ratio_features(df, numerical, numerator='last', denominator='mean', type='difference')
    first_last_ratio_features = generate_difference_ratio_features(df, numerical, numerator='last', denominator='first', type='ratio')
    lag_last_ratio_features = generate_difference_ratio_features(df, numerical, numerator='last', denominator='lag', type='ratio')
    mean_last_ratio_features = generate_difference_ratio_features(df, numerical, numerator='last', denominator='mean', type='ratio')
    min_diff_features = generate_features(df_diff, numerical, feature_type='min', prefix="diff")
    max_diff_features = generate_features(df_diff, numerical, feature_type='max', prefix="diff")
    max_last_features = generate_difference_ratio_features(df, numerical, numerator='last', denominator='max', type='difference')
    min_last_features = generate_difference_ratio_features(df, numerical, numerator='last', denominator='min', type='difference')

    ########################
    ## Self Rank Features ##
    ########################
    max_df = df.groupby('customer_ID')[numerical].max()
    max_df = df[['customer_ID']].merge(max_df, on='customer_ID')
    rank_df = (df[numerical] == max_df[numerical]).astype(int)
    del max_df
    _ = gc.collect()
    for c in tqdm(numerical):
        rank_df[c] = df['ind'].values * rank_df[c]

    rank_df['customer_ID'] = df['customer_ID'].values
    argmax_features = generate_features(rank_df, numerical, feature_type='max', prefix="argmax")

    del rank_df
    _ = gc.collect()

    min_df = df.groupby('customer_ID')[numerical].min()
    min_df = df[['customer_ID']].merge(min_df, on='customer_ID')
    rank_df = (df[numerical] == min_df[numerical]).astype(int)
    del min_df
    _ = gc.collect()
    for c in tqdm(numerical):
        rank_df[c] = df['ind'].values * rank_df[c]

    rank_df['customer_ID'] = df['customer_ID'].values
    argmin_features = generate_features(rank_df, numerical, feature_type='max', prefix="argmin")

    del rank_df
    _ = gc.collect()
    argmin_features = argmin_features.astype(int)
    argmax_features = argmax_features.astype(int)

    rank_df = df.copy()
    for c in tqdm(numerical):
        rank_df[c] = rank_df[c].rank(pct=True)

    last_rank_features = generate_features(rank_df, numerical, feature_type='last', prefix="rank")
    mean_rank_features = generate_features(rank_df, numerical, feature_type='mean', prefix="rank")
    max_rank_features = generate_features(rank_df, numerical, feature_type='max', prefix="rank")
    min_rank_features = generate_features(rank_df, numerical, feature_type='min', prefix="rank")

    df = pd.concat([last_features, mean_features, min_features, max_features, std_features, first_features, count_features,
                    unique_features, recent_mean_features, recent_min_features, recent_max_features, recent_std_features,
                    first_last_diff_features, first_last_ratio_features, lag_last_diff_features, lag_last_ratio_features,
                    mean_last_diff_features, mean_last_ratio_features, min_diff_features, max_diff_features,
                    max_last_features, min_last_features, argmax_features, argmin_features, last_rank_features,
                    mean_rank_features, max_rank_features, min_rank_features], axis=1)

    del last_features, mean_features, min_features, max_features, std_features, first_features, count_features,\
        unique_features, recent_mean_features, recent_min_features, recent_max_features, recent_std_features,\
        first_last_diff_features, first_last_ratio_features, lag_last_diff_features, lag_last_ratio_features,\
        mean_last_diff_features, mean_last_ratio_features, min_diff_features, max_diff_features, \
        max_last_features, min_last_features, argmax_features, argmin_features, last_rank_features,\
        mean_rank_features, max_rank_features, min_rank_features

    gc.collect()
    print(f"Created {df.shape[1]} raw features")
    features = np.load("feature_selection/features_10.npy")
    df = df[features]
    cat_features = [c for c in features if c in config.cat_features]
    df[features] = df[features].astype(np.float32)
    print(f"Selected {df.shape[1]} raw features")
    for c in cat_features:
        df[c] = df[c].fillna(-1).astype(int)

    return df.reset_index(drop=True)

def main():
    ###########
    ## Train ##
    ###########
    print("Preparing Train Data")
    df = pd.read_feather("train_processed/train_with_index.f")
    df = df.replace(-1, np.nan)
    labels = pd.read_csv("input/train_labels.csv")
    train_missing_value_feature = pd.read_feather("train_features/train_missing_value_feature.f")
    train_missing_statement_feature = pd.read_feather("train_features/train_missing_statement_feature.f")
    train_missing_value_feature = labels[['customer_ID']].merge(train_missing_value_feature, on='customer_ID',how='left')
    train_missing_statement_feature = labels[['customer_ID']].merge(train_missing_statement_feature, on='customer_ID', how='left')
    cols = [c for c in df.columns if c not in ['customer_ID', 'S_2', 'ind']]
    categoricals = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
    numerical = [c for c in cols if c not in categoricals]

    df = generate_train_test(df, cols, categoricals, numerical)
    df['missing_value_feature'] = train_missing_value_feature['missing_value_feature'].values
    df['missing_statement_feature'] = train_missing_statement_feature['missing_statement_feature'].values
    df.to_feather("train_features/train.f")

    del df, train_missing_value_feature, train_missing_statement_feature
    gc.collect()
    print("Train Data Complete")
    ##########
    ## Test ##
    ##########
    print("Preparing Test Data")
    df = pd.read_feather("test_processed/test_with_index.f")
    sub = pd.read_csv("input/sample_submission.csv")
    df = df.replace(-1, np.nan)

    test_missing_value_feature = pd.read_feather("test_features/test_missing_value_feature.f")
    test_missing_statement_feature = pd.read_feather("test_features/test_missing_statement_feature.f")
    test_missing_value_feature = sub[['customer_ID']].merge(test_missing_value_feature, on='customer_ID', how='left')
    test_missing_statement_feature = sub[['customer_ID']].merge(test_missing_statement_feature, on='customer_ID',how='left')
    df = generate_train_test(df, cols, categoricals, numerical)
    df['missing_value_feature'] = test_missing_value_feature['missing_value_feature'].values
    df['missing_statement_feature'] = test_missing_statement_feature['missing_statement_feature'].values
    df.to_feather("test_features/test.f")
    del df, test_missing_value_feature, test_missing_statement_feature
    gc.collect()
    print("Test Data Complete")

if __name__ == "__main__":
    main()

