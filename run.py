import os

PREPROCESS = True
ADD_INDEX = True
CREATE_INTERMEDIATE_MODEL = True
CREATE_DISTANCE_FEATURES = True
RETRAIN = True
REGEN_FEATURES = False
RETEST_FEATURES = False
SELECT_FEATURES = False

#################
## Preprocesss ##
#################
cmd = "python ./001_preprocess_train_test.py"
os.system(cmd)

#########################
## Add Statement Index ##
#########################
cmd = "python ./002_add_index.py"
os.system(cmd)

###########################################
## Add NA and Missing Statement Features ##
###########################################
cmd = "python ./003_intermediate_model.py"
os.system(cmd)

###########################
## Add Distance Features ##
###########################
cmd = "python ./004_create_distance_features.py"
os.system(cmd)

####################################
## Feature Creation and Selection ##
####################################
cmd = "python ./005_test_features.py"
os.system(cmd)

####################################
## Create Train and Test Datasets ##
####################################
cmd = "python ./006_create_train_test.py"
os.system(cmd)

#######################
## Feature Selection ##
#######################
cmd = "python ./007_feature_reduction.py"
os.system(cmd)

#######################
## Feature Selection ##
#######################
cmd = "python ./models/lightgbm_generic.py 500"
os.system(cmd)



