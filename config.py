import os

class ParamConfig:
    def __init__(self):
        self.seed = 42
        self.NFOLDS = 5
        self.TEST_SIZE_SPLIT = 0.2
        self.STEPS_TO_SELECT = 5
        self.feature_numbers = {'last':150,
                                'first':50,
                                'mean':75,
                                'min':50,
                                'max':75,
                                'std':50,
                                'count':25,
                                'nunique':25,
                                'recent_min':50,
                                'recent_max': 50,
                                'recent_mean': 50,
                                'recent_std': 50,
                                'last_first_diff':50,
                                'last_first_ratio':50,
                                'last_mean_diff': 50,
                                'last_mean_ratio': 50,
                                'last_lag_diff': 50,
                                'last_lag_ratio': 50,
                                'max_diff': 50,
                                'min_diff': 50,
                                'max_last_diff':75,
                                'min_last_diff':75,
                                'argmax': 50,
                                'argmin': 50,
                                'rank_last':75,
                                'rank_mean':50,
                                'rank_min':50,
                                'rank_max':50
        }

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
                             "D_68_first"
                            ]

## initialize a param config
config = ParamConfig()