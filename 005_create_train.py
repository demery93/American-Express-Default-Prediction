import pandas as pd
import numpy as np

from feature_generator import generate_features, generate_difference_ratio_features, feature_selection
from config import config
from tqdm import tqdm

import gc

def generate_train_test(df, cols, categoricals, numerical):

    for c in numerical:
        df[c] = ((df[c] // 0.01) / 100).astype(np.float32)

    df_diff = df.copy()
    df_diff['ind'] = df_diff['ind'] + 1
    df_diff = df[['customer_ID','ind']].merge(df_diff, on=['customer_ID','ind'], how='left')
    df_diff[numerical] = df[numerical] - df_diff[numerical]

    last_features = generate_features(df, cols, feature_type='last', prefix="").astype('float32')
    mean_features = generate_features(df, numerical, feature_type='mean', prefix="").astype('float32')
    min_features = generate_features(df, numerical, feature_type='min', prefix="").astype('float32')
    max_features = generate_features(df, numerical, feature_type='max', prefix="").astype('float32')
    std_features = generate_features(df, numerical, feature_type='std', prefix="").astype('float32')
    count_features = generate_features(df, categoricals, feature_type='count', prefix="").astype('float32')
    unique_features = generate_features(df, categoricals, feature_type='nunique', prefix="").astype('float32')
    first_features = generate_features(df, cols, feature_type='first', prefix="").astype('float32')
    print("Created Base Features")

    recent_mean_features = generate_features(df, numerical, feature_type='mean', prefix="recent", recent=10).astype('float32')
    recent_min_features = generate_features(df, numerical, feature_type='min', prefix="recent", recent=10).astype('float32')
    recent_max_features = generate_features(df, numerical, feature_type='max', prefix="recent", recent=10).astype('float32')
    recent_std_features = generate_features(df, numerical, feature_type='std', prefix="recent", recent=10).astype('float32')
    print("Created Recent Features")

    df_null = df.copy()
    df_null[cols] = df_null[cols].isnull().astype(np.int8)
    last_null_features = generate_features(df_null, cols, feature_type='last', prefix="null").astype(np.int8)
    sum_null_features = generate_features(df_null, cols, feature_type='sum', prefix="null").astype(np.int8)
    min_null_features = generate_features(df_null, cols, feature_type='min', prefix="null").astype(np.int8)
    max_null_features = generate_features(df_null, cols, feature_type='max', prefix="null").astype(np.int8)
    first_null_features = generate_features(df_null, cols, feature_type='first', prefix="null").astype(np.int8)
    print("Created Null Features")

    first_last_diff_features = generate_difference_ratio_features(df, numerical, numerator='last', denominator='first', type='difference').astype('float32')
    lag_last_diff_features = generate_difference_ratio_features(df, numerical, numerator='last', denominator='lag', type='difference').astype('float32')
    mean_last_diff_features = generate_difference_ratio_features(df, numerical, numerator='last', denominator='mean', type='difference').astype('float32')
    first_last_ratio_features = generate_difference_ratio_features(df, numerical, numerator='last', denominator='first', type='ratio').astype('float32')
    lag_last_ratio_features = generate_difference_ratio_features(df, numerical, numerator='last', denominator='lag', type='ratio').astype('float32')
    mean_last_ratio_features = generate_difference_ratio_features(df, numerical, numerator='last', denominator='mean', type='ratio').astype('float32')
    max_last_features = generate_difference_ratio_features(df, numerical, numerator='last', denominator='max', type='difference').astype('float32')
    min_last_features = generate_difference_ratio_features(df, numerical, numerator='last', denominator='min', type='difference').astype('float32')
    max_last_ratio_features = generate_difference_ratio_features(df, numerical, numerator='last', denominator='max',type='ratio').astype('float32')
    min_last_ratio_features = generate_difference_ratio_features(df, numerical, numerator='last', denominator='min',type='ratio').astype('float32')
    print("Created Difference/Ratio Features")

    min_diff_features = generate_features(df_diff, numerical, feature_type='min', prefix="diff").astype('float32')
    max_diff_features = generate_features(df_diff, numerical, feature_type='max', prefix="diff").astype('float32')
    mean_diff_features = generate_features(df_diff, numerical, feature_type='mean', prefix="diff").astype('float32')
    print("Created Differenced Features")


    df = pd.concat([last_features, mean_features, min_features, max_features, std_features, first_features, count_features.astype(np.int8),
                    unique_features.astype(np.int8), recent_mean_features, recent_min_features, recent_max_features, recent_std_features,
                    first_last_diff_features, first_last_ratio_features, lag_last_diff_features, lag_last_ratio_features,
                    mean_last_diff_features, mean_last_ratio_features, min_diff_features, max_diff_features,
                    max_last_features, min_last_features, max_last_ratio_features, min_last_ratio_features,mean_diff_features,
                    last_null_features, sum_null_features, min_null_features, max_null_features, first_null_features], axis=1)

    del last_features, mean_features, min_features, max_features, std_features, first_features, count_features,\
        unique_features, recent_mean_features, recent_min_features, recent_max_features, recent_std_features,\
        first_last_diff_features, first_last_ratio_features, lag_last_diff_features, lag_last_ratio_features,\
        mean_last_diff_features, mean_last_ratio_features, min_diff_features, max_diff_features, \
        max_last_features, min_last_features, max_last_ratio_features, min_last_ratio_features, \
        mean_diff_features, last_null_features, sum_null_features, min_null_features, max_null_features, first_null_features

    gc.collect()
    print(f"Created {df.shape[1]} features")

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

    cat_features = [c for c in df.columns if c in config.cat_features]
    for c in tqdm(cat_features):
        df[c] = df[c].fillna(-1).astype(int)

    print(f"Selected {df.shape[1]} raw features")
    df['missing_value_feature'] = train_missing_value_feature['missing_value_feature'].values
    df['missing_statement_feature'] = train_missing_statement_feature['missing_statement_feature'].values
    dist = pd.read_feather("train_features/dist_features.f")
    df = pd.concat([df, dist], axis=1)
    print("Train shape", df.shape)

    df.to_feather("train_features/train.f")

    del df, train_missing_value_feature, train_missing_statement_feature, dist
    gc.collect()
    print("Train Data Complete")

if __name__ == "__main__":
    main()

