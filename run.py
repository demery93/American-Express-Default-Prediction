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
#os.system(cmd)

#########################
## Add Statement Index ##
#########################
cmd = "python ./002_add_index.py"
#os.system(cmd)

###########################################
## Add NA and Missing Statement Features ##
###########################################
cmd = "python ./003_intermediate_model.py"
#os.system(cmd)

###########################
## Add Distance Features ##
###########################
cmd = "python ./004_create_distance_features.py"
#os.system(cmd)

####################################
## Feature Creation and Selection ##
####################################
cmd = "python ./005_create_train.py"
os.system(cmd)

####################################
## Create Train and Test Datasets ##
####################################
cmd = "python ./006_feature_reduction.py"
os.system(cmd)

#######################
## Feature Selection ##
#######################
cmd = "python ./007_create_test.py"
os.system(cmd)

############################
## LightGBM Model w/ DART ##
############################
#cmd = "python ./models/100_model1.py"
cmd = "python ./100_model1.py"
os.system(cmd)

##################################
## LightGBM Model w/ Focal Loss ##
##################################
#cmd = "python ./models/101_model2.py"
cmd = "python ./101_model2.py"
os.system(cmd)

####################
## Catboost Model ##
####################
#cmd = "python ./models/102_model3.py"
cmd = "python ./102_model3.py"
os.system(cmd)



