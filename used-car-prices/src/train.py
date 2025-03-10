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
MODEL = os.environ.get("MODEL")
TARGET = os.environ.get("TARGET")

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
    df_test[TARGET] = -1

    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold==FOLD]
    df_test["kfold"] = -1

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

    cols = [c for c in full_data.columns if c not in ["id", "target", "kfold"]]
    cat_feats = le.CategoricalFeatures(full_data, 
                                    categorical_features=cols, 
                                    encoding_type="auto", 
                                    handle_na=True)
    full_data_transformed = cat_feats.fit_transform()
    print(full_data_transformed.head())

    train_df = full_data_transformed[full_data_transformed["id"].isin(train_idx)]
    valid_df = full_data_transformed[full_data_transformed["id"].isin(valid_idx)]
    test_df = full_data_transformed[full_data_transformed["id"].isin(test_idx)]



    train_df = train_df.drop(["id", TARGET, "kfold"], axis=1)
    valid_df = valid_df.drop(["id", TARGET, "kfold"], axis=1)

    valid_df = valid_df[train_df.columns]

    # label_encoders = {}
    # for c in train_df.columns:
    #     lbl = preprocessing.LabelEncoder()
    #     lbl.fit(train_df[c].astype(str).tolist() + valid_df[c].astype(str).tolist())
    #     train_df[c] = lbl.transform(train_df[c].astype(str).tolist())
    #     valid_df[c] = lbl.transform(valid_df[c].astype(str).tolist())
    #     label_encoders[c] = lbl
    


    # Data is ready to train
    # reg = XGBRegressor(njobs=-1, n_estimators=100, learning_rate=0.1, max_depth=5, verbose=2)
    # Define categorical columns




    # Initialize and train CatBoost
    reg = dispatcher.MODELS[MODEL]

    # reg = ensemble.RandomForestRegressor(n_jobs=-1, verbose=2)
    reg.fit(train_df, ytrain)
    pred_proba = reg.predict_proba(valid_df)[:,1]
    preds = reg.predict(valid_df)
    print(preds[:5])
    print(pred_proba[:5])
    print(yvalid[:5])

    print(f"Accuracy : {mm.ClassificationMetrics._accuracy(yvalid, preds)}")
    print(f"Precision : {mm.ClassificationMetrics._precision(yvalid, preds)}")
    print(f"Recall : {mm.ClassificationMetrics._recall(yvalid, preds)}")
    print(f"F1 score : {mm.ClassificationMetrics._f1(yvalid, preds)}")
    print(f"KS-score : {mm.ClassificationMetrics._ks_score(yvalid, pred_proba)}")
    print(f"ROC-AUC-score : {mm.ClassificationMetrics._auc(yvalid, pred_proba)}")
    print(f"LogLoss : {mm.ClassificationMetrics._logloss(yvalid, pred_proba)}")

    # joblib.dump(label_encoders, f"models/{MODEL}_{FOLD}_label_encoder.pkl")
    # joblib.dump(reg, f"models/{MODEL}_{FOLD}.pkl")
    # joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")