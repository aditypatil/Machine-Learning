import os
import pandas as pd
import numpy as np
from . import dispatcher
from sklearn import ensemble 
from sklearn import preprocessing
from sklearn import metrics
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import joblib


TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")

FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3],
}

if __name__=='__main__':
    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)

    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold==FOLD]

    ytrain = train_df.price.values
    yvalid = valid_df.price.values

    train_df = train_df.drop(["id", "price", "kfold"], axis=1)
    valid_df = valid_df.drop(["id", "price", "kfold"], axis=1)

    valid_df = valid_df[train_df.columns]

    label_encoders = {}
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[c].astype(str).tolist() + valid_df[c].astype(str).tolist())
        train_df[c] = lbl.transform(train_df[c].astype(str).tolist())
        valid_df[c] = lbl.transform(valid_df[c].astype(str).tolist())
        label_encoders[c] = lbl

    # Data is ready to train
    # reg = XGBRegressor(njobs=-1, n_estimators=100, learning_rate=0.1, max_depth=5, verbose=2)
    # Define categorical columns
    cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()

    # Initialize and train CatBoost
    reg = dispatcher.MODELS[MODEL]

    # reg = ensemble.RandomForestRegressor(n_jobs=-1, verbose=2)
    reg.fit(train_df, ytrain)
    preds = reg.predict(valid_df)[:]
    # print("ROC_AUC Score = {}".format(metrics.roc_auc_score(yvalid, preds)))
    print("RMSE = {}".format(metrics.mean_squared_error(yvalid, preds)))
    # print("Accuracy = {}".format(metrics.accuracy_score(yvalid, preds)))

    joblib.dump(label_encoders, f"models/{MODEL}_{FOLD}_label_encoder.pkl")
    joblib.dump(reg, f"models/{MODEL}_{FOLD}.pkl")
    joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")