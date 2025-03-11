import os
import pandas as pd
import numpy as np
from . import dispatcher
from . import metrics as mm
from . import categorical as le
from sklearn import ensemble 
from sklearn import preprocessing
from sklearn import metrics
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import joblib

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
NUM_FOLDS = int(os.environ.get("NUM_FOLDS", 5))
MODEL = os.environ.get("MODEL")
TARGET = os.environ.get("TARGET")

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

    valid_df = valid_df[train_df.columns]

    print(f"\n Starting Model Training Using {MODEL}... \n")

    # Data is ready to train
    reg = dispatcher.MODELS[MODEL]

    # reg = ensemble.RandomForestRegressor(n_jobs=-1, verbose=2)
    reg.fit(train_df, ytrain)
    # pred_proba = reg.predict_proba(valid_df)[:,1]
    preds = reg.predict(valid_df)
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

    # joblib.dump(label_encoders, f"models/{MODEL}_{FOLD}_label_encoder.pkl")
    # joblib.dump(reg, f"models/{MODEL}_{FOLD}.pkl")
    # joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")