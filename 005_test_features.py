import pandas as pd
import numpy as np

from feature_generator import generate_features, generate_difference_ratio_features, feature_selection
from utils import cv_check, amex_metric_mod_lgbm
from config import config

import gc

df = pd.read_feather("train_processed/train_with_index.f")
df = df.replace(-1, np.nan)
labels = pd.read_csv("input/train_labels.csv")
cols = [c for c in df.columns if c not in ['customer_ID','S_2','ind']]
categoricals = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
numerical = [c for c in cols if c not in categoricals]

###################
## Last Features ##
###################
train = generate_features(df, cols, feature_type='last', prefix="")
features = [c for c in train.columns]
train[features] = train[features].astype(np.float32)
n_features = config.feature_numbers['last']
cat_features = [c for c in features if c in config.cat_features]
for c in cat_features:
    train[c] = train[c].fillna(-1).astype(int)
print(f"Selecting {n_features} out of {train.shape[1]} features")
features = feature_selection(train, labels.target.values, features, n_features, cat_features=cat_features)
np.save("feature_selection/features_0.npy", features)

cv = cv_check(pd.concat([labels[['target']], train.reset_index(drop=True)], axis=1), features)
print(f"CV AMEX Score after Last Features added: {np.round(cv['amex_metric-mean'][-1],4)}") #0.7888
print(f"CV Log Loss after Last Features added: {np.round(cv['binary_logloss-mean'][-1],4)}") #0.2203
train[features].reset_index(drop=True).to_feather("train_features/selected_features.f")

###################
## Mean Features ##
###################
features = np.load("feature_selection/features_0.npy")
mean_features = generate_features(df, numerical, feature_type='mean', prefix="")
train = pd.concat([train[features], mean_features.astype(np.float32)], axis=1)
del mean_features
_ = gc.collect()
features = [c for c in train.columns]
cat_features = [c for c in features if c in config.cat_features]
n_features = n_features + config.feature_numbers['mean'] #
print(f"Selecting {n_features} out of {train.shape[1]} features")
features = feature_selection(train, labels.target.values, features, n_features, cat_features=cat_features)
np.save("feature_selection/features_1.npy", features)

cv = cv_check(pd.concat([labels[['target']], train.reset_index(drop=True)], axis=1), features)
print(f"CV AMEX Score after Mean Features added: {np.round(cv['amex_metric-mean'][-1],4)}") #0.7916
print(f"CV Log Loss after Mean Features added: {np.round(cv['binary_logloss-mean'][-1],4)}") #0.2181
train[features].reset_index(drop=True).to_feather("train_features/selected_features.f")
############################
## Min, Max, Std Features ##
############################
features = np.load("feature_selection/features_1.npy")
min_features = generate_features(df, numerical, feature_type='min', prefix="")
max_features = generate_features(df, numerical, feature_type='max', prefix="")
std_features = generate_features(df, numerical, feature_type='std', prefix="")
train = pd.concat([train[features], min_features.astype(np.float32), max_features.astype(np.float32), std_features.astype(np.float32)], axis=1)
del min_features, max_features, std_features
_ = gc.collect()
features = [c for c in train.columns]
cat_features = [c for c in features if c in config.cat_features]
n_features = n_features + config.feature_numbers['min'] + config.feature_numbers['max'] + config.feature_numbers['std']
print(f"Selecting {n_features} out of {train.shape[1]} features")
features = feature_selection(train, labels.target.values, features, n_features, cat_features=cat_features)
np.save("feature_selection/features_2.npy", features)

cv = cv_check(pd.concat([labels[['target']], train.reset_index(drop=True)], axis=1), features)
print(f"CV AMEX Score after Min, Max, and Std Features added: {np.round(cv['amex_metric-mean'][-1],4)}") #0.7918
print(f"CV Log Loss after Min, Max, and Std Features added: {np.round(cv['binary_logloss-mean'][-1],4)}") #0.2177
train[features].reset_index(drop=True).to_feather("train_features/selected_features.f")

####################
## Extra Features ##
####################
features = np.load("feature_selection/features_2.npy")
count_features = generate_features(df, categoricals, feature_type='count', prefix="")
unique_features = generate_features(df, categoricals, feature_type='nunique', prefix="")
first_features = generate_features(df, cols, feature_type='first', prefix="")
train = pd.concat([train[features].astype(np.float32), count_features.astype(int), unique_features.astype(int), first_features.astype(np.float32)], axis=1)
del count_features, unique_features, first_features
_ = gc.collect()
features = [c for c in train.columns]
cat_features = [c for c in features if c in config.cat_features]
n_features = n_features + config.feature_numbers['count'] + config.feature_numbers['nunique'] + config.feature_numbers['first']
print(f"Selecting {n_features} out of {train.shape[1]} features")
for c in cat_features:
    train[c] = train[c].fillna(-1).astype(int)
features = feature_selection(train, labels.target.values, features, n_features, cat_features=cat_features)
np.save("feature_selection/features_3.npy", features)

cv = cv_check(pd.concat([labels[['target']], train.reset_index(drop=True)], axis=1), features)
print(f"CV AMEX Score after Count, Unique, First Features added: {np.round(cv['amex_metric-mean'][-1],4)}") #0.7917
print(f"CV Log Loss after Count, Unique, First Features added: {np.round(cv['binary_logloss-mean'][-1],4)}") #0.2176
train[features].reset_index(drop=True).to_feather("train_features/selected_features.f")

#####################
## Recent Features ##
#####################
features = np.load("feature_selection/features_3.npy")
mean_features = generate_features(df, numerical, feature_type='mean', prefix="recent", recent=10)
min_features = generate_features(df, numerical, feature_type='min', prefix="recent", recent=10)
max_features = generate_features(df, numerical, feature_type='max', prefix="recent", recent=10)
std_features = generate_features(df, numerical, feature_type='std', prefix="recent", recent=10)
train = pd.concat([train[features], mean_features.astype(np.float32), min_features.astype(np.float32), max_features.astype(np.float32), std_features.astype(np.float32)], axis=1)
del mean_features, min_features, max_features, std_features
_ = gc.collect()
features = [c for c in train.columns]
cat_features = [c for c in features if c in config.cat_features]
n_features = n_features + config.feature_numbers['recent_mean'] + config.feature_numbers['recent_min'] + config.feature_numbers['recent_max'] + config.feature_numbers['recent_std']
print(f"Selecting {n_features} out of {train.shape[1]} features")
for c in cat_features:
    train[c] = train[c].fillna(-1).astype(int)
features = feature_selection(train, labels.target.values, features, n_features, cat_features=cat_features)
np.save("feature_selection/features_4.npy", features)

cv = cv_check(pd.concat([labels[['target']], train.reset_index(drop=True)], axis=1), features)
print(f"CV AMEX Score after Recent Features added: {np.round(cv['amex_metric-mean'][-1],4)}") #0.7927
print(f"CV Log Loss after Recent Features added: {np.round(cv['binary_logloss-mean'][-1],4)}") #0.2173
train[features].reset_index(drop=True).to_feather("train_features/selected_features.f")

#########################
## Diff/Ratio Features ##
#########################
features = np.load("feature_selection/features_4.npy")
first_last_features = generate_difference_ratio_features(df, numerical, numerator='last', denominator='first', type='difference')
lag_last_features = generate_difference_ratio_features(df, numerical, numerator='last', denominator='lag', type='difference')
mean_last_features = generate_difference_ratio_features(df, numerical, numerator='last', denominator='mean', type='difference')
first_last_ratio_features = generate_difference_ratio_features(df, numerical, numerator='last', denominator='first', type='ratio')
lag_last_ratio_features = generate_difference_ratio_features(df, numerical, numerator='last', denominator='lag', type='ratio')
mean_last_ratio_features = generate_difference_ratio_features(df, numerical, numerator='last', denominator='mean', type='ratio')
train = pd.concat([train[features], first_last_features.astype(np.float32), lag_last_features.astype(np.float32), mean_last_features.astype(np.float32),
                   first_last_ratio_features.astype(np.float32), lag_last_ratio_features.astype(np.float32), mean_last_ratio_features.astype(np.float32)], axis=1)
del first_last_features, lag_last_features, mean_last_features,first_last_ratio_features,lag_last_ratio_features,mean_last_ratio_features
_ = gc.collect()
features = [c for c in train.columns]
cat_features = [c for c in features if c in config.cat_features]
n_features = n_features + config.feature_numbers['last_first_diff'] + config.feature_numbers['last_lag_diff'] + config.feature_numbers['last_mean_diff'] \
             + config.feature_numbers['last_first_ratio'] + config.feature_numbers['last_lag_ratio'] + config.feature_numbers['last_mean_ratio']
print(f"Selecting {n_features} out of {train.shape[1]} features")
features = feature_selection(train, labels.target.values, features, n_features, cat_features=cat_features)
np.save("feature_selection/features_5.npy", features)

cv = cv_check(pd.concat([labels[['target']], train.reset_index(drop=True)], axis=1), features)
print(f"CV AMEX Score after Diff/Ratio Features added: {np.round(cv['amex_metric-mean'][-1],4)}") #0.7937
print(f"CV Log Loss after Diff/Ratio Features added: {np.round(cv['binary_logloss-mean'][-1],4)}") #0.2172
train[features].reset_index(drop=True).to_feather("train_features/selected_features.f")

###########################
## Min/Max Diff Features ##
###########################
df_diff = df.copy()
df_diff['ind'] = df_diff['ind'] + 1
df_diff = df[['customer_ID','ind']].merge(df_diff, on=['customer_ID','ind'], how='left')
df_diff[numerical] = df[numerical] - df_diff[numerical]
features = np.load("feature_selection/features_5.npy")
min_features = generate_features(df_diff, numerical, feature_type='min', prefix="diff")
max_features = generate_features(df_diff, numerical, feature_type='max', prefix="diff")
train = pd.concat([train[features], min_features.astype(np.float32), max_features.astype(np.float32)], axis=1)
del min_features, max_features
_ = gc.collect()
features = [c for c in train.columns]
cat_features = [c for c in features if c in config.cat_features]
n_features = n_features + config.feature_numbers['min_diff'] + config.feature_numbers['max_diff']
print(f"Selecting {n_features} out of {train.shape[1]} features")
features = feature_selection(train, labels.target.values, features, n_features, cat_features=cat_features)
np.save("feature_selection/features_6.npy", features)

cv = cv_check(pd.concat([labels[['target']], train.reset_index(drop=True)], axis=1), features)
print(f"CV AMEX Score after Diff Features added: {np.round(cv['amex_metric-mean'][-1],4)}") #0.7938
print(f"CV Log Loss after Diff Features added: {np.round(cv['binary_logloss-mean'][-1],4)}") #0.2170
train[features].reset_index(drop=True).to_feather("train_features/selected_features.f")

################################
## Last/Max Last/Min Features ##
################################
features = np.load("feature_selection/features_6.npy")
max_last_features = generate_difference_ratio_features(df, numerical, numerator='last', denominator='max', type='difference')
min_last_features = generate_difference_ratio_features(df, numerical, numerator='last', denominator='min', type='difference')
max_last_ratio_features = generate_difference_ratio_features(df, numerical, numerator='last', denominator='max', type='ratio')
min_last_ratio_features = generate_difference_ratio_features(df, numerical, numerator='last', denominator='min', type='ratio')

train = pd.concat([train[features], max_last_features.astype(np.float32), min_last_features.astype(np.float32),
                   max_last_ratio_features.astype(np.float32), min_last_ratio_features.astype(np.float32)], axis=1)
del min_last_features, max_last_features, max_last_ratio_features, min_last_ratio_features
_ = gc.collect()
features = [c for c in train.columns]
cat_features = [c for c in features if c in config.cat_features]
n_features = n_features + config.feature_numbers['max_last_diff'] + config.feature_numbers['min_last_diff']
print(f"Selecting {n_features} out of {train.shape[1]} features")
features = feature_selection(train, labels.target.values, features, n_features, cat_features=cat_features)
np.save("feature_selection/features_7.npy", features)

cv = cv_check(pd.concat([labels[['target']], train.reset_index(drop=True)], axis=1), features)
print(f"CV AMEX Score after Min/Max Diff/Ratio Features added: {np.round(cv['amex_metric-mean'][-1],4)}") #0.7941
print(f"CV Log Loss after Min/Max Diff/Ratio Features added: {np.round(cv['binary_logloss-mean'][-1],4)}") #0.2168
train[features].reset_index(drop=True).to_feather("train_features/selected_features.f")

########################
## Self Rank Features ##
########################
features = np.load("feature_selection/features_7.npy")
#train = pd.read_feather("train_features/selected_features.f")

from tqdm import tqdm
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

train = pd.concat([train[features].reset_index(drop=True), argmax_features.reset_index(drop=True).astype(int), argmin_features.reset_index(drop=True).astype(int)], axis=1)
del argmax_features, argmin_features
_ = gc.collect()
features = [c for c in train.columns]
cat_features = [c for c in features if c in config.cat_features]

n_features = n_features + config.feature_numbers['argmax'] + config.feature_numbers['argmin']
print(f"Selecting {n_features} out of {train.shape[1]} features")
features = feature_selection(train, labels.target.values, features, n_features, cat_features=cat_features)
np.save("feature_selection/features_8.npy", features)

cv = cv_check(pd.concat([labels[['target']], train.reset_index(drop=True)], axis=1), features)
print(f"CV AMEX Score after Diff Features added: {np.round(cv['amex_metric-mean'][-1],4)}") #0.7942
print(f"CV Log Loss after Diff Features added: {np.round(cv['binary_logloss-mean'][-1],4)}") #0.2168
train[features].reset_index(drop=True).to_feather("train_features/selected_features.f")

##########################
## Global Rank Features ##
##########################
features = np.load("feature_selection/features_8.npy")
rank_df = df.copy()
for c in tqdm(numerical):
    rank_df[c] = rank_df[c].rank(pct=True)

last_rank_features = generate_features(rank_df, numerical, feature_type='last', prefix="rank")
mean_rank_features = generate_features(rank_df, numerical, feature_type='mean', prefix="rank")
max_rank_features = generate_features(rank_df, numerical, feature_type='max', prefix="rank")
min_rank_features = generate_features(rank_df, numerical, feature_type='min', prefix="rank")

train = pd.concat([train[features].reset_index(drop=True), last_rank_features.reset_index(drop=True).astype(np.float32),
                   mean_rank_features.reset_index(drop=True).astype(np.float32), max_rank_features.reset_index(drop=True).astype(np.float32),
                   min_rank_features.reset_index(drop=True).astype(np.float32)], axis=1)
del last_rank_features, mean_rank_features, max_rank_features, min_rank_features
_ = gc.collect()
features = [c for c in train.columns]
cat_features = [c for c in features if c in config.cat_features]
n_features = n_features + config.feature_numbers['rank_last'] + config.feature_numbers['rank_mean'] + config.feature_numbers['rank_min'] + config.feature_numbers['rank_max']
print(f"Selecting {n_features} out of {train.shape[1]} features")
features = feature_selection(train, labels.target.values, features, n_features, cat_features=cat_features)
np.save("feature_selection/features_9.npy", features)

cv = cv_check(pd.concat([labels[['target']], train.reset_index(drop=True)], axis=1), features)
print(f"CV AMEX Score after Rank Features added: {np.round(cv['amex_metric-mean'][-1],4)}") #0.794
print(f"CV Log Loss after Rank Features added: {np.round(cv['binary_logloss-mean'][-1],4)}") #0.2166
train[features].reset_index(drop=True).to_feather("train_features/selected_features.f")








