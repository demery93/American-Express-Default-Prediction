import pandas as pd
import numpy as np

import gc

def main():
    train = pd.read_feather("train_processed/train.f")
    test = pd.read_feather("test_processed/test.f")
    ntrain = len(train)
    df = pd.concat([train, test], axis=0, ignore_index=True)
    del train, test
    gc.collect()

    df['S_2'] = pd.to_datetime(df['S_2'])
    df['month'] = df['S_2'].dt.month
    df['year'] = df['S_2'].dt.year
    df['year'] = df['year'] - df['year'].min()
    df['statement'] = df['year'] * 12 + df['month']
    df['diff_to_max'] = df.groupby("customer_ID")['statement'].transform(lambda x: x.max() - x)
    df['ind'] = 13 - df['diff_to_max']

    dropcols = ['month','year','statement','diff_to_max']
    df.drop(dropcols, axis=1, inplace=True)

    df[:ntrain].reset_index(drop=True).to_feather("train_processed/train_with_index.f")
    df[ntrain:].reset_index(drop=True).to_feather("test_processed/test_with_index.f")

if(__name__ == '__main__'):
    main()


