import os

class ParamConfig:
    def __init__(self):
        self.seed = 42
        self.NFOLDS = 5
        self.NBAG = 5
        self.TEST_SIZE_SPLIT = 0.2
        self.STEPS_TO_SELECT = 10
        self.feature_numbers = {'base_features': 830, # 1110
                                'recent_features':428, # 708
                                'diff_ratio_features':782, # 1062
                                'diff_features':959, # 1239
                                'arg_features':74, # 354
                                'rank_features':428} #708
        #3500/5181
        self.cat_features = ["B_30_last",
                             "B_38_last",
                             "D_114_last",
                             "D_116_last",
                             "D_117_last",
                             "D_120_last",
                             "D_126_last",
                             "D_63_last",
                             "D_64_last",
                             "D_66_last",
                             "D_68_last",
                             "B_30_first",
                             "B_38_first",
                             "D_114_first",
                             "D_116_first",
                             "D_117_first",
                             "D_120_first",
                             "D_126_first",
                             "D_63_first",
                             "D_64_first",
                             "D_66_first",
                             "D_68_first",
                             "S_2_weekday_first",
                             "S_2_weekday_last"
                            ]

        self.lightgbm_gbdt_params = {'1500': {'objective': 'binary',
                                             'metric': 'custom',
                                             'learning_rate': 0.02,
                                             'max_depth': 8,
                                             'num_leaves': 2 ** 8 - 1,
                                             'max_bin': 255,
                                             'min_child_weight': 10,
                                             'reg_lambda': 60,  # L2 regularization term on weights.
                                             'colsample_bytree': 0.5,
                                             'subsample': 0.9,
                                             'nthread': 8,
                                             'bagging_freq': 1,
                                             'verbose': -1,
                                             'seed': 42},
                                     '2000': {'objective': 'binary',
                                             'metric': 'custom',
                                             'learning_rate': 0.02,
                                             'max_depth': 8,
                                             'num_leaves': 2 ** 8 - 1,
                                             'max_bin': 255,
                                             'min_child_weight': 20,
                                             'reg_lambda': 70,  # L2 regularization term on weights.
                                             'colsample_bytree': 0.4,
                                             'subsample': 0.9,
                                             'nthread': 8,
                                             'bagging_freq': 1,
                                             'verbose': -1,
                                             'seed': 42},
                                     '2500': {'objective': 'binary',
                                                       'metric': 'custom',
                                                       'learning_rate': 0.02,
                                                       'max_depth': 8,
                                                       'num_leaves': 2 ** 8 - 1,
                                                       'max_bin': 255,
                                                       'min_child_weight': 24,
                                                       'reg_lambda': 80,  # L2 regularization term on weights.
                                                       'colsample_bytree': 0.3,
                                                       'subsample': 0.9,
                                                       'nthread': 8,
                                                       'bagging_freq': 1,
                                                       'verbose': -1,
                                                       'seed': 42
                                                       },
                                     '3000': {'objective': 'binary',
                                              'metric': 'custom',
                                              'learning_rate': 0.02,
                                              'max_depth': 6,
                                              'num_leaves': 2 ** 6 - 1,
                                              'max_bin': 255,
                                              'min_child_weight': 24,
                                              'reg_lambda': 80,  # L2 regularization term on weights.
                                              'colsample_bytree': 0.3,
                                              'subsample': 0.9,
                                              'nthread': 8,
                                              'bagging_freq': 1,
                                              'verbose': -1,
                                              'seed': 42
                                              }

        }
        self.lightgbm_dart_params = {'1500': {'objective': 'binary',
                                              'metric': "amex_metric",
                                              'boosting': 'dart',
                                              'seed': 42,
                                              'num_leaves': 2**6-1,
                                              'learning_rate': 0.01,
                                              'feature_fraction': 0.5,
                                              'bagging_freq': 10,
                                              'bagging_fraction': 0.80,
                                              'n_jobs': -1,
                                              'min_data_in_leaf': 10},
            '2000': {'objective': 'binary',
                                              'metric': "amex_metric",
                                              'boosting': 'dart',
                                              'seed': 42,
                                              'num_leaves': 120,
                                              'learning_rate': 0.01,
                                              'feature_fraction': 0.25,
                                              'bagging_freq': 10,
                                              'bagging_fraction': 0.60,
                                              'n_jobs': -1,
                                              'lambda_l2': 2,
                                              'min_data_in_leaf': 35},
                                     '2500': {'objective': 'binary',
                                              'metric': "amex_metric",
                                              'boosting': 'dart',
                                              'seed': 42,
                                              'num_leaves': 100,
                                              'learning_rate': 0.01,
                                              'feature_fraction': 0.20,
                                              'bagging_freq': 10,
                                              'bagging_fraction': 0.50,
                                              'n_jobs': -1,
                                              'lambda_l2': 2,
                                              'min_data_in_leaf': 40
                                              }

                                     }
        self.xgboost_params = {'1500': {'objective': 'binary:logistic',
                                          'tree_method': 'gpu_hist',
                                          'max_depth': 7,
                                          'subsample': 0.95,
                                          'colsample_bytree': 0.5,
                                          'gamma': 1.5,
                                          'min_child_weight': 15,
                                          'lambda': 10,
                                          'eta': 0.03},
                               '2000': {'objective': 'binary:logistic',
                                           'tree_method': 'gpu_hist',
                                           'max_depth': 5,
                                           'subsample': 0.9,
                                           'colsample_bytree': 0.3,
                                           'gamma': 1.5,
                                           'min_child_weight': 35,
                                           'lambda': 5,
                                           'eta': 0.03},
                               '2500': {'objective': 'binary:logistic',
                                        'tree_method': 'gpu_hist',
                                        'max_depth': 6,
                                        'subsample': 0.9,
                                        'colsample_bytree': 0.25,
                                        'gamma': 1.5,
                                        'min_child_weight': 40,
                                        'lambda': 5,
                                        'eta': 0.03}
                               }

## initialize a param config
config = ParamConfig()