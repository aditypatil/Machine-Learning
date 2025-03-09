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
MODEL = os.environ.get("MODEL")

def predict():
    df = pd.read_csv(TEST_DATA)
    test_idx = df["id"].values
    predictions = None

    for FOLD in range(5):
        encoders = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_label_encoder.pkl"))
        cols = joblib.load(os.path.join("models" ,f"{MODEL}_{FOLD}_columns.pkl"))
        for c in cols:
            lbl = encoders[c]
            # df.loc[:, c] = lbl.transform(df[c].values.tolist())
            df[c] = df[c].map(lambda s: lbl.transform([s])[0] if s in lbl.classes_ else -1)

        # Initialize testing
        reg = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}.pkl"))
        
        df = df[cols]
        preds = reg.predict(df)[:]
        if FOLD==0:
            predictions = preds
        else:
            predictions += preds
    
    predictions /= 5

    submission = pd.DataFrame(np.column_stack((test_idx, predictions)), columns = ["id", "target"])

    return submission 

if __name__ == "__main__":
    submission = predict()
    submission.id = submission.id.astype(int)
    submission.to_csv(f"models/{MODEL}.csv", index=False)