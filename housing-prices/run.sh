export BASE_DATA=input/train.csv
export TRAINING_DATA=input/train_folds.csv
export TEST_DATA=input/test.csv

# export SHUFFLE=True
export PROBLEM_TYPE=single_col_regression
export NUM_FOLDS=5
export TARGET=SalePrice
export MODEL=$1

# python -m src.param_tuning #Deprecated, intergrated into src.train

# python -m src.create_folds

# python -m src.cross_validation #Deprecated, intergrated into src.create_folds

FOLD=0 python -m src.train
FOLD=1 python -m src.train
FOLD=2 python -m src.train
FOLD=3 python -m src.train
FOLD=4 python -m src.train

# export MODEL_LIST="LGBMRegressor,XGBRegressor,CatBoostRegressor,PytorchCNN1DRegressor"
export VOTING_TYPE="hard"

python -m src.predict
# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/leaderboard