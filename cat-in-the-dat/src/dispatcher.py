from sklearn import ensemble
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

MODELS = {
    "RandomForestClassifier": ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, max_depth=10, max_features="sqrt", verbose=2),
    "RandomForestRegressor": ensemble.RandomForestRegressor(n_estimators=200, n_jobs=-1, max_depth=10, max_features="sqrt", verbose=2),
    "CatBoostRegressor": CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, verbose=100),
    "XGBRegressor": XGBRegressor(n_estimators=100, max_depth=5, tree_method='hist', enable_categorical=True),
    "ExtraTreesRegressor": ensemble.ExtraTreesRegressor(n_estimators=200, random_state=42, verbose=2)
}