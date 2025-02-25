# Tuning hyperparameter
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor 
from lightgbm import LGBMRegressor




models_and_params = {
    "Decision Tree": {
        "model": DecisionTreeRegressor(num),
        "params": {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    "Random Forest": {
        "model": RandomForestRegressor(random_state=42),
        "params": {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    "Gradient Boosting": {
        "model": GradientBoostingRegressor(random_state=42),
        "params": {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 10]
        }
    },
    "Lasso": {
        "model": Lasso(),
        "params": {
            'alpha': [0.01, 0.1, 1, 10, 100]
        }
    },
    "XGBoost": {
        "model": XGBRegressor(random_state=42, objective='reg:squarederror'),
        "params": {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'n_estimators': [100, 500, 1000],
            'max_depth': [3, 6, 9, 12],
            'min_child_weight': [1, 5, 10],
            'gamma': [0, 0.1, 0.5, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0]
        }
    },
    "LightGBM": {
        "model": LGBMRegressor(random_state=42, objective='regression'),
        "params": {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'num_leaves': [20, 30, 40, 50, 60],
            'min_child_samples': [10, 20, 30, 50],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0]
        }
    }
}

# Loop through models for hyperparameter tuning
best_estimators={}
results=[]

for model_name, model_info in models_and_params.items():
    print(f"Tuning {model_name}...")
    
    # Initialize RandomizedSearchCV
    search = RandomizedSearchCV(
        estimator=model_info["model"],
        param_distributions=model_info["params"],
        scoring='neg_mean_squared_error',
        cv=5,
        n_iter=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit the model
    search.fit(X_train, y_train)
    
    # Best estimator and parameters
    best_estimators[model_name] = search.best_estimator_
    best_params = search.best_params_
    
    # Evaluate on the test set
    y_pred = search.best_estimator_.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    results.append({
        "Model": model_name,
        "Best Parameters": best_params,
        "MSE": mse,
        "R² Score": r2
    })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Display results
print(results_df)