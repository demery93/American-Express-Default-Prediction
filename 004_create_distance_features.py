import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

train = pd.read_feather("train_processed/train_with_index.f")
labels = pd.read_csv("input/train_labels.csv")

def generate_self_distance(df, index, sc=None):
    cols = [c for c in df.columns if c not in ['customer_ID','S_2','ind']]
    categoricals = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
    numerical = [c for c in cols if c not in categoricals]

    if(sc == None):
        sc = StandardScaler()
        sc.fit(df[numerical])

    df[numerical] = sc.transform(df[numerical])

    first = df.groupby("customer_ID").first()
    last = df.groupby("customer_ID").last()
    mean = df.groupby("customer_ID").mean()
    min = df.groupby("customer_ID").min()
    max = df.groupby("customer_ID").max()

    first_last_distance = np.mean(np.sqrt((last[numerical] - first[numerical]) ** 2), axis=1)
    mean_last_distance = np.mean(np.sqrt((last[numerical] - mean[numerical]) ** 2), axis=1)
    min_last_distance = np.mean(np.sqrt((last[numerical] - min[numerical]) ** 2), axis=1)
    max_last_distance = np.mean(np.sqrt((last[numerical] - max[numerical]) ** 2), axis=1)
    min_max_distance = np.mean(np.sqrt((min[numerical] - max[numerical]) ** 2), axis=1)

    dist_df = pd.DataFrame({"first_last_distance":first_last_distance, "mean_last_distance":mean_last_distance,
                            "min_last_distance":min_last_distance, "max_last_distance": max_last_distance,
                            "min_max_distance":min_max_distance}, index=first_last_distance.index)

    dist_df = index[['customer_ID']].merge(dist_df.reset_index())
    dist_df.drop('customer_ID', inplace=True, axis=1)
    dist_df.reset_index(drop=True, inplace=True)

    print("Complete")
    return dist_df, sc

def main():
    train = pd.read_feather("train_processed/train_with_index.f")
    labels = pd.read_csv("input/train_labels.csv")
    dist_train, sc = generate_self_distance(train, labels)
    dist_train.to_feather("train_features/dist_features.f")
    test = pd.read_feather("test_processed/test_with_index.f")
    sub = pd.read_csv("input/sample_submission.csv")
    dist_test, sc = generate_self_distance(test, sub, sc=sc)
    dist_test.to_feather("test_features/dist_features.f")

if(__name__ == "__main__"):
    main()