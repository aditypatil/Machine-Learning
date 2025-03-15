import os
import pandas as pd
import numpy as np
from . import dispatcher
from . import metrics as mm
from . import categorical as le
from . import param_tuning
from . import feature_selector as fs
from sklearn import ensemble 
from sklearn import preprocessing
from sklearn import metrics
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import joblib
import json
from .utils import PytorchClassifierWrapper, PytorchRegressorWrapper

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
NUM_FOLDS = int(os.environ.get("NUM_FOLDS", 5))
MODEL = os.environ.get("MODEL")
TARGET = os.environ.get("TARGET")

NN_MODELS = [
    "PytorchSimpleNNClassifier",
    "PytorchSimpleNNRegressor",
    "PytorchCNN1DRegressor"
]

def train_and_predict(model_name, train_df, ytrain, valid_df, input_dim=None):
    """
    Train a model and make predictions, handling both scikit-learn and PyTorch models.
    
    Args:
        model_name (str): Name of the model from MODELS dictionary
        train_df (pd.DataFrame): Training features
        ytrain (np.ndarray): Training target
        valid_df (pd.DataFrame): Validation features
        input_dim (int, optional): Input dimension for PyTorch models
    
    Returns:
        tuple: (pred_proba, preds) for classifiers, (None, preds) for regressors
    """
    # Get best parameters
    def load_best_params(model_name):
        try:
            with open(f"model/best_params_{model_name}.json", "r") as f:
                best_params = json.load(f)
            print(f"Loaded best params for {model_name}: {best_params}")
            return best_params
        except FileNotFoundError:
            print(f"No saved best params found for {model_name}. Using default settings.")
            return None
    # Get the model
    if model_name in NN_MODELS:
        if input_dim is None:
            raise ValueError(f"input_dim must be specified for PyTorch model: {model_name}")
        reg = PytorchRegressorWrapper(
            pytorch_model_class=dispatcher.MODELS[model_name],  # Pass the class, not an instance
            input_dim=input_dim,
            epochs=20,
            batch_size=32,
            lr=0.002
        )
    else:
        reg = dispatcher.MODELS[model_name]
        print(f"Tuning {model_name}...")
        tuner = param_tuning.ParamTuner(model_name=model_name, optimization_method="RandomSearchCV", task="regression")
        train_full = pd.concat([train_df.reset_index(drop=True), pd.Series(ytrain, name=TARGET).reset_index(drop=True)], axis=1)
        best_params, _ = tuner._optimize(train_full, n_iter=30, cv=2)
        # best_params = load_best_params("XGBRegressor")
        print(f"Best parameters found: {best_params}")
        if best_params:
            reg.set_params(**best_params)
    
    # Train
    reg.fit(train_df, ytrain)
    
    # Predict
    if hasattr(reg, 'predict_proba'):  # For classifiers
        pred_proba = reg.predict_proba(valid_df)[:, 1]
        preds = reg.predict(valid_df)
        return pred_proba, preds, reg
    else:  # For regressors
        preds = reg.predict(valid_df)
        return None, preds, reg

FOLD_MAPPING = {i: [j for j in range(NUM_FOLDS) if j != i] for i in range(NUM_FOLDS)}

if __name__=='__main__':
    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)
    df_test[TARGET] = -1
    print("\nMapping Folds... \n")
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold==FOLD]
    df_test["kfold"] = -1

    print("\nEncoding catetgorical features... \n")
    ytrain = train_df[TARGET].values
    yvalid = valid_df[TARGET].values

    cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    
    print(train_df.shape)
    print(valid_df.shape)
    print(df_test.shape)

    train_idx = train_df["id"].reset_index(drop=True).values
    valid_idx = valid_df["id"].reset_index(drop=True).values
    test_idx = df_test["id"].reset_index(drop=True).values

    full_data = pd.concat([train_df, valid_df, df_test])

    cols = [c for c in full_data.columns if c not in ["id", TARGET, "kfold"]]
    cat_feats = le.CategoricalFeatures(full_data, 
                                    categorical_features=cols, 
                                    encoding_type="auto", 
                                    handle_na=True)
    full_data_transformed = cat_feats.fit_transform()
    print(full_data_transformed.head())

    train_df = full_data_transformed[full_data_transformed["id"].isin(train_idx)]
    valid_df = full_data_transformed[full_data_transformed["id"].isin(valid_idx)]
    test_df = full_data_transformed[full_data_transformed["id"].isin(test_idx)]

    print(train_df.shape)
    print(valid_df.shape)
    print(test_df.shape)

    train_df = train_df.drop(["id", TARGET, "kfold"], axis=1)
    valid_df = valid_df.drop(["id", TARGET, "kfold"], axis=1)

    # selector = fs.FeatureSelector(
    #     problem_type='regression',
    #     selection_methods=['variance', 'correlation'],
    #     n_features=None,
    #     auto_select=True
    # )
    # X_transformed = selector.fit_transform(train_df, ytrain, train_df.columns)
    # selected_features = selector.get_feature_names(train_df.columns)

    # train_df = train_df[X_transformed.columns]
    # valid_df = valid_df[train_df.columns]

    print(f"\n Starting Model Training Using {MODEL}... \n")

    input_dim = train_df.shape[1] if MODEL in NN_MODELS else None

    # Train and predict
    pred_proba, preds, reg = train_and_predict(MODEL, train_df, ytrain, valid_df, input_dim)

    # Data is ready to train
    # reg = dispatcher.MODELS[MODEL]

    # reg = ensemble.RandomForestRegressor(n_jobs=-1, verbose=2)
    # reg.fit(train_df, ytrain)
    # pred_proba = reg.predict_proba(valid_df)[:,1]
    # preds = reg.predict(valid_df)
    print(preds[:5])
    # print(pred_proba[:5])
    print(yvalid[:5])

    print(f"\n Evaluating Model on Validation data... \n")

    print(f"MAE : {mm.RegressionMetrics._mae(yvalid, preds)}")
    print(f"MSE : {mm.RegressionMetrics._mse(yvalid, preds)}")
    print(f"RMSE : {mm.RegressionMetrics._rmse(yvalid, preds)}")
    print(f"MSLE : {mm.RegressionMetrics._msle(yvalid, preds)}")
    print(f"RMSLE : {mm.RegressionMetrics._rmsle(yvalid, preds)}")
    print(f"R2 score : {mm.RegressionMetrics._r2(yvalid, preds)}")

    joblib.dump(cat_feats, f"models/{MODEL}_{FOLD}_label_encoder.pkl")
    joblib.dump(reg, f"models/{MODEL}_{FOLD}.pkl")
    joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")