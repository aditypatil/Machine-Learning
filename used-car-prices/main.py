import numpy as np
import pandas as pd
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from auto_ml import AutoMLTuner

path = r'/Users/adityapatil/GitHub/Machine-Learning/loan-approval'
target_var = 'loan_status'

# Redirect print output to a file
sys.stdout = open(f"{path}/logs/main.log", "w")

# Sample print statements
print("Code Begins\n")
print(f"All print outputs will be saved in '{path}/logs/main.log'.")

df_train = pd.read_csv(f'{path}/data/train.csv')
df_eval = pd.read_csv(f'{path}/data/test.csv')

# Print Data Head

print(df_train.head(5))
print(df_eval.head(5))

df_train.drop('id', axis=1, inplace=True)
df_eval.drop('id', axis=1, inplace=True)

column_types_counts = df_train.dtypes.value_counts()
print(column_types_counts)
print("\n")

# Making sure if test data has missing values?
train_missing_values=df_train.isnull().sum()
train_missing_values=train_missing_values[train_missing_values>0]
print("Missing Values in training data:\n")
print(train_missing_values)
print("\n")
eval_missing_values=df_eval.isnull().sum()
eval_missing_values=eval_missing_values[eval_missing_values>0]
print("Missing Values in eval data:\n")
print(eval_missing_values)
print("\n")

print('df_train shape:', df_train.shape,'\n')
print('df_eval shape:', df_eval.shape,'\n')

# Making sure categorical data types are converted to numeric for easier processing
from sklearn.preprocessing import LabelEncoder
df_combined = pd.concat([df_train, df_eval], axis=0)
# Convert all categorical columns to string type to avoid mixed data types
for col in df_combined.select_dtypes(include='object').columns:
    df_combined[col] = df_combined[col].astype(str)
# Apply label encoding (sklearn) on the combined dataset
label_encoders = {}
for col in df_combined.select_dtypes(include='object').columns:
    # Fit the encoder on the combined dataset
    df_combined[col] = LabelEncoder().fit_transform(df_combined[col])
    # Store the label encoder for later use
    label_encoders[col] = LabelEncoder()
# Separate the train-test datasets again
df_train_encoded = df_combined[:len(df_train)]
df_eval_encoded = df_combined[len(df_train):].drop(target_var, axis=1)

column_types_counts = df_train_encoded.dtypes.value_counts()
print(column_types_counts,'\n')
print(df_train.columns.tolist(),'\n')

X = df_train_encoded.drop(target_var, axis=1) 
y = df_train_encoded[target_var]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Distribution of Target Variable
print(df_train_encoded[target_var].value_counts())
print("\n")

# Correlation Plot

#Correlation between numerical features and target
numeric_features=df_train.select_dtypes(include=['int64','float64'])
correlation_matrix = numeric_features.corr()
correlation_with_target = correlation_matrix[target_var].sort_values(ascending=False)
print('Correlation with target:\n',correlation_with_target)
print('Correlation matrix:\n',correlation_matrix)
print("\n")
#visualize correlation
if len(correlation_with_target)<=20:
	plt.figure(figsize=(20,8))
	correlation_with_target.drop(target_var).plot(kind='bar', color='red')
	plt.title('Correlation with Loan Approval')
	plt.xlabel('Features')
	plt.ylabel('Correlation')
	plt.savefig(f'{path}/corr_plot.png', dpi=300, bbox_inches='tight')
	plt.close()
	# plt.figure(figsize=(20,20))
	# sns.pairplot(df_train[df_train.select_dtypes(include=['int64']).columns],hue=target_var)
	# plt.savefig(f'{path}/pair_plot.png', dpi=300, bbox_inches='tight')
	# plt.close()

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)
print('X_eval shape:', df_eval_encoded.shape,'\n')

# Training models

tuner = AutoMLTuner(task='classification')
best_model = tuner.fit(X_train, y_train, model_name='test_all')
results = tuner.evaluate(X_test, y_test, path)

import pickle

# Save the model
with open(f'{path}/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Predict using the trained model
df_eval['loan_status'] = best_model.predict(df_eval_encoded)

df_output = df_eval[['id', 'loan_status']]

# Save as CSV
df_output.to_csv(f'{path}/loan_status_preds.csv', index=False)


# ---------------------------- CODE ENDS -----------------------------
# Reset stdout to default (optional)
sys.stdout.close()
sys.stdout = sys.__stdout__