"""
- binary classification
- multiclass classification
- multi label classification
- single column regression
- multi column regression
- holdout
"""

from sklearn import model_selection
import pandas as pd
import os

TARGET = os.environ.get("TARGET")


class CrossValidation:
    def __init__(
        self, 
        df, 
        target_cols, 
        problem_type="binary_classification",
        num_folds = 5,
        shuffle = True,
        random_state = 42
    ):
        self.dataframe = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.num_folds = num_folds
        self.shuffle = shuffle
        self.random_state = random_state

        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

        self.dataframe["kfold"] = -1


    def split(self):
        if self.problem_type in ["binary_classification", "multiclass_classification"]:
            target = self.target_cols[0]
            unique_values = self.dataframe[self.target_cols[0]].nunique()
            if unique_values == 1:
                raise Exception("Only one unique value found!")
            elif unique_values > 1:
                kf = model_selection.StratifiedKFold(n_splits=self.num_folds, 
                                                    shuffle=False)
                for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=self.dataframe[TARGET].values)):
                    self.dataframe.loc[val_idx, 'kfold'] = fold
        elif self.problem_type == "single_col_regression":
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")
            target = self.target_cols[0]
            kf = model_selection.KFold(n_splits=self.num_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=self.dataframe[TARGET].values)):
                self.dataframe.loc[val_idx, 'kfold'] = fold
        else:
            raise Exception("Problem type unknown!")

        return self.dataframe

if __name__=="__main__":
    df = pd.read_csv("input/train.csv")
    cv = CrossValidation(df, target_cols=[TARGET], problem_type="single_col_regression")
    df_split = cv.split()
    print(df_split.head(5))
    print(df_split.kfold.value_counts())
