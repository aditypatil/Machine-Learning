import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn import ensemble, linear_model, svm, neighbors, neural_network
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import torch
import torch.nn as nn
import numpy as np

from .utils import SimpleNNClassifier, SimpleNNRegressor, CNN1DRegressor



MODELS = {
    # Classification Models (Previous ones)
    "RandomForestClassifier": ensemble.RandomForestClassifier(
        n_estimators=200, max_depth=10, max_features="sqrt", n_jobs=-1, random_state=42, verbose=2
    ),
    "XGBClassifier": XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, tree_method='hist',
        enable_categorical=True, eval_metric='mlogloss', random_state=42
    ),
    "LGBMClassifier": LGBMClassifier(
        n_estimators=100, max_depth=7, learning_rate=0.1, num_leaves=31, verbose=2, random_state=42
    ),
    "CatBoostClassifier": CatBoostClassifier(
        iterations=1000, learning_rate=0.1, depth=6, verbose=100, random_state=42,
        auto_class_weights='Balanced'
    ),
    "LogisticRegression": linear_model.LogisticRegression(
        penalty="l2", C=1.0, solver='liblinear', max_iter=1000, random_state=42
    ),
    "SVMClassifier": svm.SVC(
        kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42
    ),
    "GradientBoostingClassifier": ensemble.GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
    ),
    "ExtraTreesClassifier": ensemble.ExtraTreesClassifier(
        n_estimators=200, max_depth=10, max_features="sqrt", n_jobs=-1, random_state=42, verbose=2
    ),
    "KNeighborsClassifier": neighbors.KNeighborsClassifier(
        n_neighbors=5, weights='distance', n_jobs=-1
    ),
    "MLPClassifier": neural_network.MLPClassifier(
        hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
        learning_rate_init=0.001, max_iter=500, random_state=42
    ),


    # Regression Models 
    "LinearRegression": linear_model.LinearRegression(
        fit_intercept=True, n_jobs=-1, positive=False 
    ),
    "RandomForestRegressor": ensemble.RandomForestRegressor(
        n_estimators=200, max_depth=10, max_features="sqrt", n_jobs=-1, random_state=42, verbose=2
    ),
    "XGBRegressor": XGBRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1, tree_method='hist',
        enable_categorical=True, random_state=42
    ),
    "LGBMRegressor": LGBMRegressor(
        n_estimators=100, max_depth=7, learning_rate=0.1, num_leaves=31, verbose=0, random_state=42
    ),
    "CatBoostRegressor": CatBoostRegressor(
        iterations=1000, learning_rate=0.1, depth=6, verbose=0, random_state=42
    ),
    "ExtraTreesRegressor": ensemble.ExtraTreesRegressor(
        n_estimators=200, max_depth=10, max_features="sqrt", n_jobs=-1, random_state=42, verbose=2
    ),
    "GradientBoostingRegressor": ensemble.GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
    ),
    "SVR": svm.SVR(kernel='rbf', C=1.0, gamma='scale'),
    "KNeighborsRegressor": neighbors.KNeighborsRegressor(
        n_neighbors=5, weights='distance', n_jobs=-1
    ),
    "MLPRegressor": neural_network.MLPRegressor(
        hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
        learning_rate_init=0.001, max_iter=500, random_state=42
    ),
    "Ridge": linear_model.Ridge(alpha=1.0, random_state=42),
    "Lasso": linear_model.Lasso(alpha=1.0, random_state=42),
    "ElasticNet": linear_model.ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),

    # PyTorch Neural Network Models
    # Note: input_dim needs to be specified when initializing these models based on your data
    "PytorchSimpleNNClassifier": SimpleNNClassifier,
    "PytorchSimpleNNRegressor": SimpleNNRegressor,
    "PytorchCNN1DRegressor": CNN1DRegressor
}

