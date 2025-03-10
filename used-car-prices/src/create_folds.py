import pandas as pd 
from sklearn import model_selection
import os
from . import cross_validation as cval

BASE_DATA = os.environ.get("BASE_DATA")
TRAINING_DATA = os.environ.get("TRAINING_DATA")
NUM_FOLDS = int(os.environ.get("NUM_FOLDS", 5))
TARGET = os.environ.get("TARGET")
PROBLEM_TYPE = os.environ.get("PROBLEM_TYPE")
SHUFFLE = os.environ.get("SHUFFLE",True)

if __name__ == "__main__":
    df = pd.read_csv(BASE_DATA)
    cv = cval.CrossValidation(df, target_cols=[TARGET], num_folds=NUM_FOLDS, problem_type="single_col_regression", shuffle=SHUFFLE)
    df_split = cv.split()
    # print(df_split.head(5))
    print(f"Following volume was created in each of the {NUM_FOLDS} folds:")
    print(df_split.kfold.value_counts())
    
    df.to_csv(TRAINING_DATA, index=False)

