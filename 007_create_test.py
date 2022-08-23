import pandas as pd
import numpy as np

from feature_generator import generate_features, generate_difference_ratio_features, feature_selection
from config import config
from tqdm import tqdm

import gc

def select(df, features):
    cols = [c for c in df.columns if c in features]
    return df[cols]


def generate_train_test(df, cols, categoricals, numerical):
    feats = np.load("feature_selection/top_2500_features.npy")

    for c in numerical:
        df[c] = ((df[c] // 0.01) / 100).astype(np.float32)

    df_diff = df.copy()
    df_diff['ind'] = df_diff['ind'] + 1
    df_diff = df[['customer_ID','ind']].merge(df_diff, on=['customer_ID','ind'], how='left')
    df_diff[numerical] = df[numerical] - df_diff[numerical]

    last_features = select(generate_features(df, cols, feature_type='last', prefix=""), feats).astype('float32')
    mean_features = select(generate_features(df, numerical, feature_type='mean', prefix=""), feats).astype('float32')
    min_features = select(generate_features(df, numerical, feature_type='min', prefix=""), feats).astype('float32')
    max_features = select(generate_features(df, numerical, feature_type='max', prefix=""), feats).astype('float32')
    std_features = select(generate_features(df, numerical, feature_type='std', prefix=""), feats).astype('float32')
    count_features = select(generate_features(df, categoricals, feature_type='count', prefix=""), feats).astype('float32')
    unique_features = select(generate_features(df, categoricals, feature_type='nunique', prefix=""), feats).astype('float32')
    first_features = select(generate_features(df, cols, feature_type='first', prefix=""), feats).astype('float32')
    print("Created Base Features")

    recent_mean_features = select(generate_features(df, numerical, feature_type='mean', prefix="recent", recent=10), feats).astype('float32')
    recent_min_features = select(generate_features(df, numerical, feature_type='min', prefix="recent", recent=10), feats).astype('float32')
    recent_max_features = select(generate_features(df, numerical, feature_type='max', prefix="recent", recent=10), feats).astype('float32')
    recent_std_features = select(generate_features(df, numerical, feature_type='std', prefix="recent", recent=10), feats).astype('float32')
    print("Created Recent Features")

    df_null = df.copy()
    df_null[cols] = df_null[cols].isnull().astype(np.int8)
    last_null_features = select(generate_features(df_null, cols, feature_type='last', prefix="null"), feats).astype(np.int8)
    sum_null_features = select(generate_features(df_null, cols, feature_type='sum', prefix="null"), feats).astype(np.int8)
    min_null_features = select(generate_features(df_null, cols, feature_type='min', prefix="null"), feats).astype(np.int8)
    max_null_features = select(generate_features(df_null, cols, feature_type='max', prefix="null"), feats).astype(np.int8)
    first_null_features = select(generate_features(df_null, cols, feature_type='first', prefix="null"), feats).astype(np.int8)
    print("Created Null Features")

    first_last_diff_features = select(generate_difference_ratio_features(df, numerical, numerator='last', denominator='first', type='difference'), feats).astype('float32')
    lag_last_diff_features = select(generate_difference_ratio_features(df, numerical, numerator='last', denominator='lag', type='difference'), feats).astype('float32')
    mean_last_diff_features = select(generate_difference_ratio_features(df, numerical, numerator='last', denominator='mean', type='difference'), feats).astype('float32')
    first_last_ratio_features = select(generate_difference_ratio_features(df, numerical, numerator='last', denominator='first', type='ratio'), feats).astype('float32')
    lag_last_ratio_features = select(generate_difference_ratio_features(df, numerical, numerator='last', denominator='lag', type='ratio'), feats).astype('float32')
    mean_last_ratio_features = select(generate_difference_ratio_features(df, numerical, numerator='last', denominator='mean', type='ratio'), feats).astype('float32')
    max_last_features = select(generate_difference_ratio_features(df, numerical, numerator='last', denominator='max', type='difference'), feats).astype('float32')
    min_last_features = select(generate_difference_ratio_features(df, numerical, numerator='last', denominator='min', type='difference'), feats).astype('float32')
    max_last_ratio_features = select(generate_difference_ratio_features(df, numerical, numerator='last', denominator='max',type='ratio'), feats).astype('float32')
    min_last_ratio_features = select(generate_difference_ratio_features(df, numerical, numerator='last', denominator='min',type='ratio'), feats).astype('float32')
    print("Created Difference/Ratio Features")

    min_diff_features = select(generate_features(df_diff, numerical, feature_type='min', prefix="diff"), feats).astype('float32')
    max_diff_features = select(generate_features(df_diff, numerical, feature_type='max', prefix="diff"), feats).astype('float32')
    mean_diff_features = select(generate_features(df_diff, numerical, feature_type='mean', prefix="diff"), feats).astype('float32')
    print("Created Differenced Features")

    max_df = df.groupby('customer_ID')[numerical].max()
    max_df = df[['customer_ID']].merge(max_df, on='customer_ID')
    rank_df = (df[numerical] == max_df[numerical]).astype(int)
    del max_df
    _ = gc.collect()
    for c in tqdm(numerical):
        rank_df[c] = df['ind'].values * rank_df[c]

    rank_df['customer_ID'] = df['customer_ID'].values
    argmax_features = select(generate_features(rank_df, numerical, feature_type='max', prefix="argmax").astype(int), feats)

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
    argmin_features = select(generate_features(rank_df, numerical, feature_type='max', prefix="argmin").astype(int), feats)

    del rank_df
    _ = gc.collect()
    print("Created Argmin/Argmax Features")

    rank_df = df.copy()
    for c in tqdm(numerical):
        rank_df[c] = rank_df[c].rank(pct=True)

    last_rank_features = select(generate_features(rank_df, numerical, feature_type='last', prefix="rank"), feats).astype('float32')
    mean_rank_features = select(generate_features(rank_df, numerical, feature_type='mean', prefix="rank"), feats).astype('float32')
    max_rank_features = select(generate_features(rank_df, numerical, feature_type='max', prefix="rank"), feats).astype('float32')
    min_rank_features = select(generate_features(rank_df, numerical, feature_type='min', prefix="rank"), feats).astype('float32')
    print("Created Rank Features")

    df = pd.concat([last_features, mean_features, min_features, max_features, std_features, first_features, count_features,
                    unique_features, recent_mean_features, recent_min_features, recent_max_features, recent_std_features,
                    first_last_diff_features, first_last_ratio_features, lag_last_diff_features, lag_last_ratio_features,
                    mean_last_diff_features, mean_last_ratio_features, min_diff_features, max_diff_features,
                    max_last_features, min_last_features, argmax_features, argmin_features, last_rank_features,
                    mean_rank_features, max_rank_features, min_rank_features, max_last_ratio_features, min_last_ratio_features,mean_diff_features,
                    last_null_features, sum_null_features, min_null_features, max_null_features, first_null_features], axis=1)

    del last_features, mean_features, min_features, max_features, std_features, first_features, count_features,\
        unique_features, recent_mean_features, recent_min_features, recent_max_features, recent_std_features,\
        first_last_diff_features, first_last_ratio_features, lag_last_diff_features, lag_last_ratio_features,\
        mean_last_diff_features, mean_last_ratio_features, min_diff_features, max_diff_features, \
        max_last_features, min_last_features, argmax_features, argmin_features, last_rank_features,\
        mean_rank_features, max_rank_features, min_rank_features, max_last_ratio_features, min_last_ratio_features, \
        mean_diff_features, last_null_features, sum_null_features, min_null_features, max_null_features, first_null_features

    gc.collect()
    print(f"Created {df.shape[1]} features")
    return df.reset_index(drop=True)

def main():
    print("Preparing Test Data")
    df = pd.read_feather("test_processed/test_with_index.f")
    sub = pd.read_csv("input/sample_submission.csv")
    feats = np.load("feature_selection/top_2500_features.npy")
    df = df.replace(-1, np.nan)

    test_missing_value_feature = pd.read_feather("test_features/test_missing_value_feature.f")
    test_missing_statement_feature = pd.read_feather("test_features/test_missing_statement_feature.f")
    test_missing_value_feature = sub[['customer_ID']].merge(test_missing_value_feature, on='customer_ID', how='left')
    test_missing_statement_feature = sub[['customer_ID']].merge(test_missing_statement_feature, on='customer_ID',
                                                                how='left')

    cols = [c for c in df.columns if c not in ['customer_ID', 'S_2', 'ind']]
    categoricals = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
    numerical = [c for c in cols if c not in categoricals]

    df = generate_train_test(df, cols, categoricals, numerical)

    cat_features = [c for c in df.columns if c in config.cat_features]
    for c in tqdm(cat_features):
        df[c] = df[c].fillna(-1).astype(int)

    print(f"Selected {df.shape[1]} raw features")

    df['missing_value_feature'] = test_missing_value_feature['missing_value_feature'].values
    df['missing_statement_feature'] = test_missing_statement_feature['missing_statement_feature'].values
    dist = pd.read_feather("test_features/dist_features.f")
    df = pd.concat([df, dist], axis=1)
    df = select(df, feats)
    print("Test shape", df.shape)
    df.to_feather("test_features/test.f")
    del df, test_missing_value_feature, test_missing_statement_feature, dist
    gc.collect()
    print("Test Data Complete")


if __name__ == "__main__":
    main()

