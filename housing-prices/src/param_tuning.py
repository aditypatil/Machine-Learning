import pandas as pd
import numpy as np
from sklearn import ensemble
from . import metrics as mm
from sklearn import model_selection
from scipy.stats import uniform, loguniform, randint
from . import dispatcher
from . import categorical as le
import os
import scipy.stats as stats
import joblib
import json
from sklearn.metrics import make_scorer

# TRAINING_DATA = os.environ.get("TRAINING_DATA")
# TEST_DATA = os.environ.get("TEST_DATA")
# FOLD = int(os.environ.get("FOLD"))
# NUM_FOLDS = int(os.environ.get("NUM_FOLDS", 5))
# MODEL = os.environ.get("MODEL")
# TARGET = os.environ.get("TARGET")

class ParamTuner():
    def __init__(self, model_name, task="regression", optimization_method="RandomSearchCV", test_size=0.2, random_state=42, target='target'):
        self.model_name = model_name
        self.method = optimization_method
        self.test_size = test_size
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.task = task
        self.target = target
        self.scorer = make_scorer(mm.RegressionMetrics._rmsle)

    def _optimize(self, dataframe, n_iter=10, cv=5, scoring=None):
        # print(f"Columns in dataframe just after _optimize call: {len(dataframe.columns)}")
        best_score = -np.inf
        X = dataframe.drop(columns=[self.target]).values
        # print(f"Columns in dataframe after defining X _optimize call: {list(dataframe.columns)}")
        y = dataframe[self.target].values
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        model = dispatcher.MODELS[self.model_name]
        if self.method == "RandomSearchCV":
            param_grid = self._get_param_grid(self.model_name)
            print(f"Model Name: {self.model_name}")
            print(f"Parameter Grid: {param_grid}")
            from itertools import product
            total_combinations = np.prod([
                len(v) if isinstance(v, list) else
                (v.a - v.b if isinstance(v, stats.rv_discrete) else 10)
                for v in param_grid.values()
            ]) if param_grid else 1
            # Ensure n_iter does not exceed total combinations
            adjusted_n_iter = min(n_iter, total_combinations)
            if 'LGBM' in self.model_name:
                model.set_params(force_col_wise=True)
            print(f"Total iterations searching:{adjusted_n_iter}")
            search = model_selection.RandomizedSearchCV(model, param_grid, n_iter=adjusted_n_iter, cv=cv, scoring=scoring, n_jobs=-1, pre_dispatch="2*n_jobs", random_state=self.random_state)
            search.fit(X_train, y_train)
            
            y_pred = search.best_estimator_.predict(X_test)
            
            score = mm.ClassificationMetrics._auc(y_test, y_pred) if self.task == 'classification' else mm.RegressionMetrics._rmse(y_test, y_pred)
            print(f"{self.model_name}: Best Score = {score:.4f}")

            if score > best_score:
                best_score = score
                self.best_model = search.best_estimator_
                self.best_params = search.best_params_
        else:
            raise Exception("Invalid tuning method!")
        return self.best_params, self.best_model

    def _get_param_grid(self, model_name):
        """Defines hyperparameter grids for models."""
        params_grid = {
            # Classification Models
            "RandomForestClassifier": {
                "n_estimators": randint(50, 300),  # Number of trees
                "max_depth": randint(3, 20),       # Max depth of trees
                "max_features": ["sqrt", "log2", None],  # Number of features to consider
                "min_samples_split": randint(2, 10),     # Min samples to split
                "min_samples_leaf": randint(1, 5)        # Min samples per leaf
            },
            "XGBClassifier": {
                "n_estimators": randint(50, 200),
                "max_depth": randint(3, 10),
                "learning_rate": loguniform(1e-3, 0.3),  # Log scale for learning rate
                "subsample": uniform(0.5, 0.5),          # Fraction of samples (0.5 to 1.0)
                "colsample_bytree": uniform(0.5, 0.5)    # Fraction of features
            },
            "LGBMClassifier": {
                "n_estimators": randint(50, 200),
                "max_depth": randint(3, 15),
                "learning_rate": loguniform(1e-3, 0.3),
                "num_leaves": randint(15, 50),
                "min_child_samples": randint(5, 30)
            },
            "CatBoostClassifier": {
                "iterations": randint(100, 1000),
                "learning_rate": loguniform(1e-3, 0.3),
                "depth": randint(4, 10),
                "l2_leaf_reg": uniform(1, 10)  # Regularization term
            },
            "LogisticRegression": {
                "C": loguniform(1e-4, 10),  # Inverse of regularization strength
                "penalty": ["l1", "l2"],    # Type of regularization
                "solver": ["liblinear"]     # Solver compatible with l1 and l2
            },
            "SVMClassifier": {
                "C": loguniform(1e-3, 10),
                "gamma": loguniform(1e-4, 1),  # Kernel coefficient
                "kernel": ["rbf", "linear"]
            },
            "GradientBoostingClassifier": {
                "n_estimators": randint(50, 200),
                "learning_rate": loguniform(1e-3, 0.3),
                "max_depth": randint(3, 10),
                "subsample": uniform(0.5, 0.5)
            },
            "ExtraTreesClassifier": {
                "n_estimators": randint(50, 300),
                "max_depth": randint(3, 20),
                "max_features": ["sqrt", "log2", None],
                "min_samples_split": randint(2, 10)
            },
            "KNeighborsClassifier": {
                "n_neighbors": randint(3, 15),
                "weights": ["uniform", "distance"],
                "p": [1, 2]  # 1 for Manhattan, 2 for Euclidean
            },
            "MLPClassifier": {
                "hidden_layer_sizes": [(50,), (100,), (100, 50), (50, 50)],
                "learning_rate_init": loguniform(1e-4, 1e-2),
                "alpha": loguniform(1e-5, 1e-2),  # L2 penalty
                "activation": ["relu", "tanh"]
            },

            # Regression Models
            "LinearRegression": {
                "fit_intercept": [True, False],         # Whether to fit the intercept or not
                "positive": [True, False]               # Whether to force coefficients to be positive
            },
            "RandomForestRegressor": {
                "n_estimators": randint(50, 300),
                "max_depth": randint(3, 20),
                "max_features": ["sqrt", "log2", None],
                "min_samples_split": randint(2, 10),
                "min_samples_leaf": randint(1, 5)
            },
            "XGBRegressor": {
                "n_estimators": randint(50, 200),
                "max_depth": randint(3, 10),
                "learning_rate": loguniform(1e-3, 0.3),
                "subsample": uniform(0.5, 0.5),
                "colsample_bytree": uniform(0.5, 0.5)
            },
            "LGBMRegressor": {
                "n_estimators": randint(50, 200),
                "max_depth": randint(3, 15),
                "learning_rate": loguniform(1e-3, 0.3),
                "num_leaves": randint(15, 50),
                "min_child_samples": randint(5, 30)
            },
            "CatBoostRegressor": {
                "iterations": randint(100, 1000),
                "learning_rate": loguniform(1e-3, 0.3),
                "depth": randint(4, 10),
                "l2_leaf_reg": uniform(1, 10)
            },
            "ExtraTreesRegressor": {
                "n_estimators": randint(50, 300),
                "max_depth": randint(3, 20),
                "max_features": ["sqrt", "log2", None],
                "min_samples_split": randint(2, 10)
            },
            "GradientBoostingRegressor": {
                "n_estimators": randint(50, 200),
                "learning_rate": loguniform(1e-3, 0.3),
                "max_depth": randint(3, 10),
                "subsample": uniform(0.5, 0.5)
            },
            "SVR": {
                "C": loguniform(1e-3, 10),
                "gamma": loguniform(1e-4, 1),
                "kernel": ["rbf", "linear"]
            },
            "KNeighborsRegressor": {
                "n_neighbors": randint(3, 15),
                "weights": ["uniform", "distance"],
                "p": [1, 2]
            },
            "MLPRegressor": {
                "hidden_layer_sizes": [(50,), (100,), (100, 50), (50, 50)],
                "learning_rate_init": loguniform(1e-4, 1e-2),
                "alpha": loguniform(1e-5, 1e-2),
                "activation": ["relu", "tanh"]
            },
            "Ridge": {
                "alpha": loguniform(1e-3, 10)
            },
            "Lasso": {
                "alpha": loguniform(1e-3, 10)
            },
            "ElasticNet": {
                "alpha": loguniform(1e-3, 10),
                "l1_ratio": uniform(0, 1)
            },

            # PyTorch Neural Network Models (assuming they accept these params)
            "PytorchSimpleNNClassifier": {
                "hidden_sizes": [(64,), (128,), (128, 64), (64, 32)],
                "learning_rate": loguniform(1e-4, 1e-2),
                "num_epochs": randint(10, 100),
                "dropout_rate": uniform(0, 0.5)
            },
            "PytorchSimpleNNRegressor": {
                "hidden_sizes": [(64,), (128,), (128, 64), (64, 32)],
                "learning_rate": loguniform(1e-4, 1e-2),
                "num_epochs": randint(10, 100),
                "dropout_rate": uniform(0, 0.5)
            },
            "PytorchCNN1DRegressor": {
                "num_filters": randint(16, 64),  # Number of filters in conv layers
                "kernel_size": randint(3, 7),    # Size of convolutional kernel
                "learning_rate": loguniform(1e-4, 1e-2),
                "num_epochs": randint(10, 100),
                "dropout_rate": uniform(0, 0.5)
            }
        }
        return params_grid.get(model_name, {})

# if __name__=="__main__":
    # df_train = pd.read_csv(TRAINING_DATA)
    # df_test = pd.read_csv(TEST_DATA)
    # df_test[TARGET] = -1

    # print("\nEncoding catetgorical features... \n")
    # ytrain = df_train[TARGET].values

    # cat_cols = df_train.select_dtypes(include=['object']).columns.tolist()
    
    # print(df_train.shape)
    # print(df_test.shape)

    # train_idx = df_train["id"].reset_index(drop=True).values
    # test_idx = df_test["id"].reset_index(drop=True).values

    # full_data = pd.concat([df_train, df_test])

    # cols = [c for c in full_data.columns if c not in ["id", TARGET, "kfold"]]
    # cat_feats = le.CategoricalFeatures(full_data, 
    #                                 categorical_features=cols, 
    #                                 encoding_type="auto", 
    #                                 handle_na=True)
    # full_data_transformed = cat_feats.fit_transform()
    # print(full_data_transformed.head())

    # train_df = full_data_transformed[full_data_transformed["id"].isin(train_idx)]
    # test_df = full_data_transformed[full_data_transformed["id"].isin(test_idx)]

    # print(train_df.shape)
    # print(test_df.shape)

    # train_df = train_df.drop(["id", TARGET, "kfold"], axis=1)
    # train_df[TARGET] = ytrain

    # tuner = ParamTuner(model_name=MODEL, optimization_method="RandomSearchCV")
    # best_params, _ = tuner._optimize(train_df, n_iter=20)
    # print(best_params)
    # with open(f"models/best_params_{MODEL}.json", "w") as f:
    #     json.dump(self.best_params, f, indent=4)
    # joblib.dump(cat_feats, f"models/label_encoder.pkl")