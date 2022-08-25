import os

class ParamConfig:
    def __init__(self):
        self.seed = 42
        self.NFOLDS = 5
        self.NBAG = 5
        self.TEST_SIZE_SPLIT = 0.2
        self.STEPS_TO_SELECT = 10

        self.dropcols = ['B_29','D_103', 'D_139']

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

## initialize a param config
config = ParamConfig()