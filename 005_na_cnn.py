import gc

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from utils import amex_metric_mod

dropcols = ['B_19', 'B_2', 'B_20', 'B_22', 'B_26', 'B_27', 'B_3', 'B_30', 'B_33', 'B_38', 'D_41', 'D_54','D_104',
            'D_107', 'D_128', 'D_129', 'D_130', 'D_131', 'D_139', 'D_141', 'D_143', 'D_145','D_114', 'D_115', 'D_116',
            'D_117', 'D_118', 'D_119', 'D_120', 'D_121', 'D_122', 'D_123', 'D_124', 'D_125','S_2']

train = pd.read_feather("train_processed/train_with_index.f")
test = pd.read_feather("test_processed/test_with_index.f")

train = train.replace(-1, np.nan)
test = test.replace(-1, np.nan)

ntrain = len(train)
df = pd.concat([train, test], axis=0, ignore_index=True)
del train, test
gc.collect()

df.drop(dropcols, axis=1, inplace=True)

cols = [c for c in df.columns if c not in ['customer_ID','target','ind']+dropcols]
n = len(cols)

df[cols] = df[cols].isnull().astype(np.int8)

train = df[:ntrain].reset_index(drop=True)
test = df[ntrain:].reset_index(drop=True)

del df
gc.collect()
## Cross join unique customers and months
print("Cross Joining Train")
df_cust = train[['customer_ID']].drop_duplicates()
df_ind = train[['ind']].drop_duplicates()

df_cust['key'] = 1
df_ind['key'] = 1

df = df_cust.merge(df_ind, on='key').drop("key", axis=1)
train = df.merge(train, on=['customer_ID', 'ind'], how='left')
train[cols] = train[cols].fillna(0).astype(np.int8) # Encode with 0 to ignore signal from missing statements
del df, df_ind, df_cust
gc.collect()

print("Cross Joining Test")
df_cust = test[['customer_ID']].drop_duplicates()
df_ind = test[['ind']].drop_duplicates()

df_cust['key'] = 1
df_ind['key'] = 1

df = df_cust.merge(df_ind, on='key').drop("key", axis=1)
test = df.merge(test, on=['customer_ID', 'ind'], how='left')
test[cols] = test[cols].fillna(0).astype(np.int8) # Encode with 0 to ignore signal from missing statements
del df, df_ind, df_cust
gc.collect()

print("Train", train.shape, "Test", test.shape)
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)
train = train.sort_values(['customer_ID', 'ind']).reset_index(drop=True)
test = test.sort_values(['customer_ID', 'ind']).reset_index(drop=True)
train.drop(['ind'], axis=1, inplace=True)
test.drop(['ind'], axis=1, inplace=True)

targets = pd.read_csv("input/train_labels.csv")
train = train.merge(targets, on='customer_ID', how='left')

y = train[['target']].values.reshape((-1, 13, 1))
y = y[:,-1,:].flatten()

train_cust = train[['customer_ID']].values.reshape((-1, 13, 1))
train_cust = train_cust[:,-1,:].flatten()
test_cust = test[['customer_ID']].values.reshape((-1, 13, 1))
test_cust = test_cust[:,-1,:].flatten()
train.drop(['target'], axis=1, inplace=True)


train = train[cols].values.reshape((-1,13,n))
test = test[cols].values.reshape((-1,13,n))

print(f"{n} Features Used")
def get_model():
    inp = tf.keras.Input(shape=(13, n))

    # CNN Feature Extraction
    x1 = tf.keras.layers.Conv1D(100, kernel_size=2, dilation_rate=1, strides=1, activation='relu', padding='causal')(inp)
    x2 = tf.keras.layers.Conv1D(100, kernel_size=2, dilation_rate=2, strides=1, activation='relu', padding='causal')(inp)
    x = tf.keras.layers.concatenate([x1, x2])
    x = tf.keras.layers.Flatten()(x)
    activation = 'swish'
    reg = 4e-4
    x0 = tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(reg),
               activation=activation,
               )(x)
    x = tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(reg),
              activation=activation,
              )(x0)
    x = tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(reg),
              activation=activation,
              )(x)
    x = tf.keras.layers.Concatenate()([x, x0])
    x = tf.keras.layers.Dropout(0.1)(x)
    # x = BatchNormalization()(x)
    x = tf.keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.l2(reg),
              activation=activation,
              )(x)
    x = tf.keras.layers.Dense(1,  # kernel_regularizer=tf.keras.regularizers.l2(4e-4),
              activation='sigmoid',
              )(x)

    # COMPILE MODEL
    model = tf.keras.Model(inputs=inp, outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(loss=loss, optimizer=opt)

    return model

kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
# Iterate through each fold
oof = np.zeros(len(train))

# CUSTOM LEARNING SCHEUDLE
def lrfn(epoch):
    lr = [1e-3]*5 + [1e-4]*2 + [1e-5]*1
    return lr[epoch]

LR = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = False)
for fold, (trn_ind, val_ind) in enumerate(kfold.split(train, y)):
    print(f'Training fold {fold + 1}')
    x_train, x_val = train[trn_ind], train[val_ind]
    y_train, y_val = y[trn_ind], y[val_ind]
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f'trained_models/na_cnn_{fold}.h5', save_best_only=True,
                                                    save_weights_only=True, verbose=2)
    model = get_model()

    model.fit(x_train, y_train, batch_size=512, epochs=8, verbose=0, callbacks=[LR], validation_data=(x_val, y_val))
    del model
    gc.collect()
    model = get_model()
    oof[val_ind] = model.predict(x_val, batch_size=128)[:,0]

    score = amex_metric_mod(y_val, oof[val_ind])
    print(f"Fold {fold+1} score: {score}")
    del model, x_train, x_val, y_train, y_val
    gc.collect()

from sklearn.metrics import roc_auc_score, log_loss
print("AMEX: ",amex_metric_mod(y, oof)) #CV Score: 0.7863097411552613
print("AUC", roc_auc_score(y, oof))
print("Log Loss", log_loss(y, oof))
preds = np.zeros(test.shape[0])
for i in range(5):
    model = get_model()
    model.load_weights(f'trained_models/na_cnn_{i}.h5')
    preds1 = model.predict(test[:200000], batch_size=128, verbose=1)[:,0]
    gc.collect()
    preds2 = model.predict(test[200000:400000], batch_size=128, verbose=1)[:, 0]
    gc.collect()
    preds3 = model.predict(test[400000:600000], batch_size=128, verbose=1)[:, 0]
    gc.collect()
    preds4 = model.predict(test[600000:800000], batch_size=128, verbose=1)[:, 0]
    gc.collect()
    preds5 = model.predict(test[800000:], batch_size=128, verbose=1)[:, 0]
    gc.collect()
    pred = np.concatenate([preds1, preds2, preds3, preds4, preds5])
    preds += pred / 5
    del model, preds1, preds2, preds3, preds4, preds5, pred
    gc.collect()


train_feature = pd.DataFrame({"customer_ID":train_cust, "na_sequence_prediction":oof})
test_feature = pd.DataFrame({"customer_ID":test_cust, "na_sequence_prediction":preds})

train_feature.to_feather("train_features/na_feature.f")
test_feature.to_feather("test_features/na_feature.f")
'''
Fold 1 score: 0.7927592833544397
Fold 2 score: 0.7832263216340759
Fold 3 score: 0.7866089737963313
Fold 4 score: 0.7827444072662118
Fold 5 score: 0.7872506449874235

CV Score: 0.7863097411552613
'''

