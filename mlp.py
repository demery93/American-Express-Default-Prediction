import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
import random
import datetime
import math
from matplotlib.ticker import MaxNLocator
from colorama import Fore, Back, Style
import gc

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer, OneHotEncoder
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm
from config import config

import tensorflow as tf

INFERENCE = True
#nfeats = sys.argv[1]
nfeats = 500

base_dropout_rates = {'200':0.0,
                      '500':0.1,
                      '1000':0.5}

print(nfeats)

# From https://www.kaggle.com/code/inversion/amex-competition-metric-python
def amex_metric_component(y_true, y_pred, return_components=False) -> float:
    """Amex metric for ndarrays"""

    def top_four_percent_captured(df) -> float:
        """Corresponds to the recall for a threshold of 4 %"""
        df['weight'] = df['target'].apply(lambda x: 20 if x == 0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()

    def weighted_gini(df) -> float:
        df['weight'] = df['target'].apply(lambda x: 20 if x == 0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(df) -> float:
        """Corresponds to 2 * AUC - 1"""
        df2 = pd.DataFrame({'target': df.target, 'prediction': df.target})
        df2.sort_values('prediction', ascending=False, inplace=True)
        return weighted_gini(df) / weighted_gini(df2)

    df = pd.DataFrame({'target': y_true.ravel(), 'prediction': y_pred.ravel()})
    df.sort_values('prediction', ascending=False, inplace=True)
    g = normalized_weighted_gini(df)
    d = top_four_percent_captured(df)

    if return_components: return g, d, 0.5 * (g + d)
    return 0.5 * (g + d)

def amex_metric(y_true, y_pred):
    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:, 0] == 0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:, 0]) / np.sum(labels[:, 0])

    gini = [0, 0]
    for i in [1, 0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:, 0] == 0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] * weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1] / gini[0] + top_four)


best_score = 0


class MyCustomMetricCallback(tf.keras.callbacks.Callback):

    def __init__(self, save_path, train=None, validation=None, best_score=0):
        super(MyCustomMetricCallback, self).__init__()
        self.train = train
        self.validation = validation
        self.best_score = best_score
        self.save_path = save_path
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        if self.train:
            #             logs['my_metric_train'] = float('inf')
            #             X_train, y_train = self.train[0], self.train[1]
            #             y_pred = self.model.predict(X_train)
            #             score = amex_metric_tensor(y_train, y_pred)
            #             logs['my_metric_train'] = np.round(score, 5)
            pass

        if self.validation:
            X_valid, y_valid = self.validation[0], self.validation[1]
            y_pred = self.model.predict(X_valid).reshape((len(X_valid),))
            val_score = amex_metric(y_valid, y_pred)
            logs['my_metric_val'] = val_score
            if val_score > self.best_score:
                self.best_score = val_score
                self.best_epoch = epoch
                self.model.save(f"{self.save_path}.h5")
                print('best_val_score: ', self.best_score)
            elif self.best_epoch == 0:
                if epoch - self.best_epoch > 40:
                    self.model.stop_training = True
            elif epoch - self.best_epoch > 12:
                self.model.stop_training = True

            del X_valid, y_valid, y_pred, val_score
            gc.collect()

features = np.load(f"feature_selection/top_{nfeats}_features.npy")
train = pd.read_feather("train_features/train.f")
test = pd.read_feather("test_features/test.f")

train = train.replace(np.inf, 0)
test = test.replace(np.inf, 0)
train = train.replace(-np.inf, 0)
test = test.replace(-np.inf, 0)
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

target = pd.read_csv('input/train_labels.csv').target.values
print(f"target shape: {target.shape}")

cat_features = [c for c in features if c in config.cat_features]
features_not_cat = [f for f in features if f not in config.cat_features]

df_categorical_train = train[cat_features].astype(object)
df_categorical_test = test[cat_features].astype(object)
ohe = OneHotEncoder(drop='first', sparse=False, dtype=np.float32, handle_unknown='ignore')
ohe.fit(df_categorical_train)
df_categorical_train = pd.DataFrame(ohe.transform(df_categorical_train).astype(np.float16),index=df_categorical_train.index).rename(columns=str)
df_categorical_test = pd.DataFrame(ohe.transform(df_categorical_test).astype(np.float16),index=df_categorical_test.index).rename(columns=str)

train = pd.concat([train[features_not_cat], df_categorical_train], axis=1)
test = pd.concat([test[features_not_cat], df_categorical_test], axis=1)

del df_categorical_train, df_categorical_test
_ = gc.collect()

LR_START = 0.01

features = [f for f in train.columns if f != 'target' and f != 'customer_ID']
base_dropout = base_dropout_rates[str(nfeats)]
def get_model(n_inputs=len(features)):
    """Sequential neural network with a skip connection.

    Returns a compiled instance of tensorflow.keras.models.Model.
    """
    activation = 'swish'
    reg = 4e-4
    inputs = tf.keras.layers.Input(shape=(n_inputs,))
    x0 = tf.keras.layers.Dropout(base_dropout)(inputs)
    x0 = tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(reg),activation=activation)(x0)
    x = tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(reg),activation=activation)(x0)
    x = tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(reg),activation=activation)(x)
    x = tf.keras.layers.Concatenate()([x, x0])
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.l2(reg),activation=activation,)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs, x)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR_START),
                  loss=tf.keras.losses.BinaryCrossentropy())
    return model



ONLY_FIRST_FOLD = False
EPOCHS_EXPONENTIALDECAY = 100
VERBOSE = 0  # set to 0 for less output, or to 2 for more output
LR_END = 1e-5  # learning rate at the end of training
CYCLES = 1
EPOCHS = 200
DIAGRAMS = True
USE_PLATEAU = False  # set to True for early stopping, or to False for exponential learning rate decay
BATCH_SIZE = 2048

np.random.seed(1)
random.seed(1)
tf.random.set_seed(1)

def exponential_decay(epoch):
    # v decays from e^a to 1 in every cycle
    # w decays from 1 to 0 in every cycle
    # epoch == 0                  -> w = 1 (first epoch of cycle)
    # epoch == epochs_per_cycle-1 -> w = 0 (last epoch of cycle)
    # higher a -> decay starts with a steeper decline
    epochs = EPOCHS_EXPONENTIALDECAY
    a = 3
    epochs_per_cycle = epochs // CYCLES
    epoch_in_cycle = epoch % epochs_per_cycle
    if epochs_per_cycle > 1:
        v = math.exp(a * (1 - epoch_in_cycle / (epochs_per_cycle - 1)))
        w = (v - 1) / (math.exp(a) - 1)
    else:
        w = 1
    return w * LR_START + (1 - w) * LR_END


lr = tf.keras.callbacks.LearningRateScheduler(exponential_decay, verbose=0)
callbacks = [lr, tf.keras.callbacks.TerminateOnNaN()]
print(f"{len(features)} features")
history_list = []
score_list = []
y_pred_list = []
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
oof = np.zeros(len(train))
preds = np.zeros(len(test))
for fold, (trn_idx, val_idx) in enumerate(kf.split(train, target)):
    y_train, y_val = target[trn_idx], target[val_idx]
    x_train, x_val = train.iloc[trn_idx][features], train.iloc[val_idx][features]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    tf.keras.backend.clear_session()
    gc.collect()

    # Construct and compile the model
    model = get_model(x_train.shape[1])
    lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                           factor=0.3,
                           patience=5,
                           mode = 'min',
                           verbose=VERBOSE)
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                       patience=7,
                       min_delta=0.00001,
                       verbose=VERBOSE,
                       mode="min",
                       restore_best_weights=True)
    custom = MyCustomMetricCallback(save_path=f'trained_models/mlp_{nfeats}_{fold}', validation=(x_val, y_val),
                                    best_score=best_score)

    callbacks = [lr, es, tf.keras.callbacks.TerminateOnNaN(), custom]
    # Train the model
    history = model.fit(x_train, y_train,
                        validation_data=(x_val, y_val),
                        epochs=100,
                        verbose=0,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        callbacks=callbacks)
    del x_train, y_train
    model.load_weights(f'trained_models/mlp_{nfeats}_{fold}.h5')
    oof[val_idx] = model.predict(x_val, verbose=0)[:,0]
    preds += model.predict(scaler.transform(test), verbose=0)[:,0]

    # Evaluation: Execution time, loss and metrics
    lastloss = f"Training loss: {history.history['loss'][-1]:.4f} | Val loss: {history.history['val_loss'][-1]:.4f}"
    score = amex_metric(y_val, oof[val_idx])
    print(f"{Fore.GREEN}{Style.BRIGHT}Fold {fold+1}"
          f" | {len(history.history['loss']):3} ep"
          f" | {lastloss} | Score: {score:.5f}{Style.RESET_ALL}")
    score_list.append(score)


print(f"{Fore.GREEN}{Style.BRIGHT}OOF Score:                       {np.mean(score_list):.5f}{Style.RESET_ALL}")
#OOF Score:                       0.79210
g, d, amex = amex_metric_component(target, oof, return_components=True)
total_positive = (target == 1).sum()
total_negative = (target == 0).sum() * 20
fourpercent = int(0.04 * (total_positive + total_negative))

plt.figure(figsize=(6, 6))
fpr, tpr, _ = roc_curve(target, oof)
plt.plot(fpr, tpr, color='#c00000', lw=3) # curve
plt.fill_between(fpr, tpr, color='#ffc0c0') # area under the curve
plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--") # diagonal
plt.plot([fourpercent / total_negative, 0], [0, fourpercent / total_positive],
         color="green", lw=3, linestyle="-") # four percent line
four_percent_index = np.argmax((fpr * total_negative + tpr * total_positive >= fourpercent))
plt.scatter([fpr[four_percent_index]],
            [tpr[four_percent_index]],
            s=100) # intersection of roc curve with four percent line
plt.gca().set_aspect('equal')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("Receiver operating characteristic")
plt.show()

print(f"Area under the curve (AUC):         {roc_auc_score(target, oof):.5f}")
print(f"2*AUC-1:                            {2 * roc_auc_score(target, oof) - 1:.5f}")
print(f"Normalized Gini coefficient:        {g:.5f} (same as 2*AUC-1)")
print()
print(f"Positive samples in validation set: {total_positive:7}")
#print(f"Negative samples in validation set: {total_negative // 20:7}")
print(f"Negative samples weighted:          {total_negative:7} (unweighted: {total_negative // 20})")
print(f"Total samples weighted:             {total_positive + total_negative:7}")
print(f"4 % of Total samples weighted:      {fourpercent:7}")
print(f"True positives at this threshold:   {int(tpr[four_percent_index] * total_positive):7}")
print(f"False positives at this threshold:  {int(fpr[four_percent_index] * total_negative):7}")
print(f"Default rate captured at 4%:        {d:7.5f} (= {int(tpr[four_percent_index] * total_positive)} / {total_positive})")
print()
print(f"Competition score:                  {amex:7.5f} (= ({g:7.5f} + {d:7.5f}) / 2)")
'''
OOF Score:                       0.78974
Area under the curve (AUC):         0.96089
2*AUC-1:                            0.92178
Normalized Gini coefficient:        0.92178 (same as 2*AUC-1)
Positive samples in validation set:   11883
Negative samples weighted:           680160 (unweighted: 34008)
Total samples weighted:              692043
4 % of Total samples weighted:        27681
True positives at this threshold:      7849
False positives at this threshold:    19860
Default rate captured at 4%:        0.65794 (= 7849 / 11883)
Competition score:                  0.78986 (= (0.92178 + 0.65794) / 2)
'''
oof = pd.DataFrame({'customer_ID': train.index,
                    'prediction': oof})
oof.to_csv(f'output/validation_mlp_{nfeats}.csv', index=False)

sub = pd.DataFrame({'customer_ID': test.index,
                    'prediction': preds})
sub.to_csv(f'output/submission_mlp_{nfeats}.csv', index=False)

plt.figure(figsize=(16, 5))
plt.hist(sub.prediction, bins=np.linspace(0, 1, 21), density=True)
plt.title("Plausibility check", fontsize=20)
plt.xlabel('Prediction')
plt.ylabel('Density')
plt.show()