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
                'RandomForest': RandomForestClassifier(),
                'XGBoost': XGBClassifier(eval_metric='logloss'),
                'LightGBM': LGBMClassifier(),
                'LogisticRegression': LogisticRegression(max_iter=5000),
                'GradientBoosting': GradientBoostingClassifier(),
                'SVM': SVC(max_iter=5000),
                'KNN': KNeighborsClassifier(),
                'DecisionTree': DecisionTreeClassifier()
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
                'n_estimators': [100, 300, 500],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'LightGBM': {
                'num_leaves': [20, 40, 60],
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [100, 300, 500]
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
            param_grid = self._get_param_grid(name)
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
