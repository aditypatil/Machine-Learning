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
from sklearn.ensemble import VotingRegressor

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
TARGET = os.environ.get("TARGET")
MODEL = os.environ.get("MODEL", "")
MODEL_LIST = os.environ.get("MODEL_LIST", "").split(",")
# If MODEL_LIST is empty or contains only empty strings, use MODEL as fallback
if not MODEL_LIST or all(not x.strip() for x in MODEL_LIST):
    MODEL_LIST = [os.environ.get("MODEL", "").strip()]  # Use MODEL, ensuring it's stripped of whitespace

# Optional: Ensure MODEL_LIST doesn't contain empty strings
MODEL_LIST = [m for m in MODEL_LIST if m]
VOTING_TYPE = os.environ.get("VOTING_TYPE", "hard")  # Default to hard voting if not specified

# MODEL = os.environ.get("MODEL")

def predict(voting_type=VOTING_TYPE):
    df = pd.read_csv(TEST_DATA)  
    test_idx = df["id"].values
    all_predictions = {}
    
    # First, collect predictions from all models
    for model_name in MODEL_LIST:
        model_predictions = None
        for FOLD in range(5):
            # Load encoders and columns
            encoders = joblib.load(os.path.join("models", f"{model_name}_{FOLD}_label_encoder.pkl"))
            cols = joblib.load(os.path.join("models", f"{model_name}_{FOLD}_columns.pkl"))
            
            # Transform test data
            df_test_transformed = encoders.transform(df)
            df_test_transformed = df_test_transformed[cols]
            
            # Load and predict with model
            reg = joblib.load(os.path.join("models", f"{model_name}_{FOLD}.pkl"))
            preds = reg.predict(df_test_transformed)
            
            if FOLD == 0:
                model_predictions = preds
            else:
                model_predictions += preds
        
        # Average predictions across folds for this model
        model_predictions /= 5
        all_predictions[model_name] = model_predictions

    # Apply voting
    if len(MODEL_LIST) > 1:
        if voting_type.lower() == "soft":
            # Soft voting: average predictions across all models
            final_predictions = np.mean(list(all_predictions.values()), axis=0)
        else:
            # Hard voting: for regression, we'll use median as a robust alternative
            final_predictions = np.median(list(all_predictions.values()), axis=0)
    else:
        # If only one model, use its predictions
        final_predictions = list(all_predictions.values())[0]

    # Create submission DataFrame
    submission = pd.DataFrame(
        np.column_stack((test_idx, final_predictions)), 
        columns=["id", TARGET]
    )
    
    return submission 



# if __name__ == "__main__":
#     submission = predict()
#     submission.id = submission.id.astype(int)
#     submission.to_csv(f"models/{MODEL}.csv", index=False)
if __name__ == "__main__":
    # Allow command-line override of voting type if needed
    voting_type = os.environ.get("VOTING_TYPE", "hard")
    submission = predict(voting_type=voting_type)
    submission.id = submission.id.astype(int)
    if len(MODEL_LIST)>1:
        submission.to_csv(f"models/submission_{'_'.join(MODEL_LIST)}_{voting_type}.csv", index=False)
    else:
        submission.to_csv(f"models/{MODEL}.csv", index=False)





