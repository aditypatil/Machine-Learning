from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from catboost import CatBoostClassifier
from h2o.automl import H2OAutoML
import h2o
h2o.init(ip="localhost", port=54321)


# Define the CatBoost model
model = CatBoostClassifier(loss_function='Logloss', eval_metric='AUC', verbose=0)

# Define the parameter grid
param_grid = {
    'iterations': [200, 500, 1000],  # Number of boosting rounds
    'depth': [4, 6, 8, 10],  # Tree depth
    'learning_rate': [0.01, 0.03, 0.1, 0.2],  # Step size shrinkage
    'l2_leaf_reg': [1, 3, 5, 10, 20],  # L2 regularization
    'border_count': [32, 64, 128],  # Number of bins for continuous features
    'bagging_temperature': [0.5, 1.0, 2.0],  # Strength of randomization
    'colsample_bylevel': [0.5, 0.7, 1.0],  # Feature sampling at each split
    'random_strength': [1, 5, 10, 50],  # Noise added to score to avoid overfitting
    'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],  # Tree growing strategy
}

class AutoMLTuner:
    def __init__(self, models=None, task='classification', test_size=0.2, random_state=42):
        """
        Initializes the tuner.
        :param models: Dictionary of models to use. If None, defaults to common models.
        :param task: 'classification' or 'regression'
        :param test_size: Proportion of data to use for testing.
        :param random_state: Random seed for reproducibility.
        """
        self.task = task
        self.test_size = test_size
        self.random_state = random_state
        self.models = models if models else self._default_models()
        self.best_model = None
        self.best_params = None

    def _default_models(self):
        """Returns a dictionary of default models."""
        if self.task == 'classification':
            return {
                'XGBoost': XGBClassifier(eval_metric='logloss',verbose=0),
                'LightGBM': LGBMClassifier(verbose=0),
                'CatBoost': CatBoostClassifier(iterations=5000, loss_function='Logloss', eval_metric='AUC', verbose=0),
                'RandomForest': RandomForestClassifier(),
                'LogisticRegression': LogisticRegression(max_iter=5000),
                'GradientBoosting': GradientBoostingClassifier(),
                # 'SVM': SVC(max_iter=5000,seed=42),
                # 'KNN': KNeighborsClassifier(seed=42),
                'DecisionTree': DecisionTreeClassifier(),
                'H2OAutoML': H2OAutoML(max_models=20, seed=42)
            }
        else:
            return {
                'RandomForest': RandomForestRegressor(),
                'XGBoost': XGBRegressor(),
                'LightGBM': LGBMRegressor(),
                'Ridge': Ridge(max_iter=5000),
                'GradientBoosting': GradientBoostingRegressor(),
                'SVM': SVR(max_iter=5000),
                'KNN': KNeighborsRegressor(),
                'DecisionTree': DecisionTreeRegressor()

            }
    
    def _get_param_grid(self, model_name):
        """Defines hyperparameter grids for models."""
        param_grids = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5, 10]
            },
            'XGBoost': {
                'n_estimators': [200, 500, 1000],  # Boosting rounds
                'learning_rate': [0.01, 0.05, 0.1],  # Step size
                'max_depth': [3, 5, 7],  # Tree depth
                'min_child_weight': [1, 3, 5],  # Min samples per leaf
                'subsample': [0.6, 0.8, 1.0],  # Row sampling
                'colsample_bytree': [0.6, 0.8, 1.0],  # Feature sampling
                'reg_alpha': [0, 0.1, 1],  # L1 regularization
                'reg_lambda': [0, 0.1, 1],  # L2 regularization
            },
            'LightGBM' : {
            'n_estimators': [200, 500, 1000],  # Boosting rounds
            'learning_rate': [0.01, 0.05, 0.1],  # Step size
            'num_leaves': [20, 31, 50],  # Tree complexity
            'max_depth': [-1, 5, 10],  # Tree depth (-1 for unlimited)
            'subsample': [0.6, 0.8, 1.0],  # Row sampling
            'colsample_bytree': [0.6, 0.8, 1.0],  # Feature sampling
            'reg_alpha': [0, 0.1, 1],  # L1 regularization
            'reg_lambda': [0, 0.1, 1],  # L2 regularization
            },
            'LogisticRegression': {
                'C': [0.1, 1, 10],
                'solver': ['lbfgs', 'liblinear']
            },
            'Ridge': {
                'alpha': [0.1, 1, 10]
            },
            'GradientBoosting': {
                'n_estimators': [100, 300, 500],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf', 'poly']
            },
            'KNN': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            },
            "DecisionTree": {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'CatBoost': {
                'depth': [4, 6, 8, 10],  # Tree depth
                'learning_rate': [0.01, 0.03, 0.1, 0.2],  # Step size shrinkage
                'l2_leaf_reg': [1, 3, 5, 10, 20],  # L2 regularization
                'border_count': [32, 64, 128],  # Number of bins for continuous features
                'bagging_temperature': [0.5, 1.0, 2.0],  # Strength of randomization
                'colsample_bylevel': [0.5, 0.7, 1.0],  # Feature sampling at each split
                'random_strength': [1, 5, 10, 50],  # Noise added to score to avoid overfitting
                'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],  # Tree growing strategy
            }
        }
        return param_grids.get(model_name, {})

    def fit(self, X, y, model_name='test_all', n_iter=10, cv=5, scoring=None):
        """
        Fits the model using RandomizedSearchCV.
        :param X: Features
        :param y: Target variable
        :param model_name: Model to use, or 'test_all' to try all models.
        :param n_iter: Number of random search iterations.
        :param cv: Cross-validation folds.
        :param scoring: Metric for hyperparameter tuning.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        best_score = -np.inf

        models_to_test = self.models if model_name == 'test_all' else {model_name: self.models[model_name]}
        
        for name, model in models_to_test.items():
            print(f"Tuning {name}...")
            if name == 'H2OAutoML':

                X_train_h2o = h2o.H2OFrame(X_train)
                X_test_h2o = h2o.H2OFrame(X_test)
                y_train_h2o = h2o.H2OFrame(y_train.to_frame()).asfactor()
                y_test_h2o = h2o.H2OFrame(y_test.to_frame()).asfactor()
                # Convert target to categorical
                y_train_h2o = y_train_h2o.asfactor()
                y_test_h2o = y_test_h2o.asfactor()

                # Combine X_train with y_train in H2OFrame
                train_h2o = X_train_h2o.cbind(y_train_h2o)
                test_h2o = X_test_h2o.cbind(y_test_h2o)

                feature_cols = X_train_h2o.columns  # Use actual column names
                target_col = y_train_h2o.columns[0]  # Target column name
                model.train(x=feature_cols, y=target_col, training_frame=train_h2o)
                y_pred = model.predict(test_h2o).as_data_frame().values.flatten()
                y_test_array = y_test.to_numpy() if not isinstance(y_test, np.ndarray) else y_test

                if self.task == 'classification':
                    y_pred = (y_pred > 0.5).astype(int)  # Apply 0.5 threshold

                # Evaluate performance
                score = accuracy_score(y_test_array, y_pred) if self.task == 'classification' else r2_score(y_test_array, y_pred)
                print(f"{name}: Best Score = {score:.4f}")
                rmse = np.sqrt(mean_squared_error(y_test_array, y_pred))
                print(f"RMSE: {rmse:.4f}")
            else:
                param_grid = self._get_param_grid(name).get(name,None)
                from itertools import product
                total_combinations = np.prod([len(v) for v in param_grid.values()]) if param_grid else 1
                # Ensure n_iter does not exceed total combinations
                adjusted_n_iter = min(n_iter, total_combinations)

                if 'LightGBM' in name:
                    model.set_params(force_col_wise=True)
                
                search = RandomizedSearchCV(model, param_grid, n_iter=adjusted_n_iter, cv=cv, scoring=scoring, n_jobs=2, pre_dispatch="2*n_jobs", random_state=self.random_state)
                search.fit(X_train, y_train)
                
                y_pred = search.best_estimator_.predict(X_test)
                
                score = accuracy_score(y_test, y_pred) if self.task == 'classification' else r2_score(y_test, y_pred)
                print(f"{name}: Best Score = {score:.4f}")
                rmse = np.sqrt(mean_squared_error(y_test_array, y_pred))
                print(f"RMSE: {rmse:.4f}")
                
                if score > best_score:
                    best_score = score
                    self.best_model = search.best_estimator_
                    self.best_params = search.best_params_
        
        print(f"Best Model: {self.best_model}")
        return self.best_model

    def evaluate(self, X_test, y_test, path):
        """
        Evaluates the best model on test data and generates performance plots.
        :param X_test: Test features.
        :param y_test: Test labels.
        """
        if not self.best_model:
            raise ValueError("No model trained. Run fit() first.")
        
        y_pred = self.best_model.predict(X_test)
        
        if self.task == 'classification':
            print("Classification Report:\n", classification_report(y_test, y_pred))
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6,6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.savefig(f'{path}/confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            return {'accuracy': accuracy_score(y_test, y_pred), 'f1': f1_score(y_test, y_pred, average='weighted')}
        else:
            errors = y_test - y_pred
            plt.figure(figsize=(8,6))
            sns.histplot(errors, bins=30, kde=True)
            plt.xlabel('Prediction Error')
            plt.title('Prediction Error Distribution')
            plt.savefig(f'{path}/prediction_error_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            return {'rmse': np.sqrt(mean_squared_error(y_test, y_pred)), 'r2': r2_score(y_test, y_pred)}

# Example Usage:
# tuner = AutoMLTuner(task='classification')
# best_model = tuner.fit(X_train, y_train, model_name='test_all')
# results = tuner.evaluate(X_test, y_test)
