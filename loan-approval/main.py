import numpy as np
import pandas as pd
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

path = r'/Users/adityapatil/GitHub/Machine-Learning/loan-approval'
target_var = 'loan_status'

# Redirect print output to a file
sys.stdout = open(f"{path}/logs/main.log", "w")

# Sample print statements
print("Code Begins\n")
print(f"All print outputs will be saved in '{path}/logs/main.log'.")

df_train = pd.read_csv(f'{path}/data/train.csv')
df_test = pd.read_csv(f'{path}/data/test.csv')

# Print Data Head

print(df_train.head(5))
print(df_test.head(5))

df_train.drop('id', axis=1, inplace=True)
df_test.drop('id', axis=1, inplace=True)

column_types_counts = df_train.dtypes.value_counts()
print(column_types_counts)

#making sure if test data has missing values?
train_missing_values=df_train.isnull().sum()
train_missing_values=train_missing_values[train_missing_values>0]
print("Missing Values:\n")
print(train_missing_values)
print("\n")

# Distribution of Target Variable

print(df_train[target_var].value_counts())

# Correlation Plot

#Correlation between numerical features and target
numeric_features=df_train.select_dtypes(include=['int64','float64'])
correlation_matrix = numeric_features.corr()
correlation_with_target = correlation_matrix[target_var].sort_values(ascending=False)
print(correlation_with_target)
#visualize correlation
plt.figure(figsize=(20,8))
correlation_with_target.drop(target_var).plot(kind='bar', color='red')
plt.title('Correlation with Loan Approval')
plt.xlabel('Features')
plt.ylabel('Correlation')
plt.savefig(f'{path}/corr_plot.png', dpi=300, bbox_inches='tight')
plt.close()
plt.figure(figsize=(20,20))
sns.pairplot(df_train[df_train.select_dtypes(include=["int"]).columns],hue=target_var)
plt.savefig(f'{path}/pair_plot.png', dpi=300, bbox_inches='tight')
plt.close()




# ---------------------------- CODE ENDS -----------------------------
# Reset stdout to default (optional)
sys.stdout.close()
sys.stdout = sys.__stdout__