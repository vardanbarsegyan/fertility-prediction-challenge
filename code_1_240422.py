#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 20:25:25 2024

@author: vardanbarsegyan
"""

# Prefer data challenge 
# 2024-01

# Install pachages 
#conda install anaconda::pandas
#conda install conda-forge::matplotlib
#conda install anaconda::numpy
#conda install anaconda::scikit-learn
#conda install conda-forge::shap
#conda install conda-forge::joblib


np.random.seed(13579)  


# Load libraris 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap
#from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler # preprocess numerical vars 
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.linear_model import LogisticRegression
import json # distionnaries 
from sklearn.model_selection import cross_validate
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from joblib import dump, load


#%%
# Load the data 
PreFer_train_data = pd.read_csv('/Users/vardanbarsegyan/Library/Mobile Documents/com~apple~CloudDocs/Papers/12_fertility_challenge/data/training_data/PreFer_train_data.csv', 
                      encoding="cp1252", 
                      low_memory=False)
 
PreFer_train_outcome = pd.read_csv('/Users/vardanbarsegyan/Library/Mobile Documents/com~apple~CloudDocs/Papers/12_fertility_challenge/data/training_data/PreFer_train_outcome.csv', 
                      encoding="cp1252", 
                      low_memory=False)


PreFer_train_background_data = pd.read_csv('/Users/vardanbarsegyan/Library/Mobile Documents/com~apple~CloudDocs/Papers/12_fertility_challenge/data/other_data/PreFer_train_background_data.csv', 
                      encoding="cp1252", 
                      low_memory=False)


# Load the codebook
PreFer_codebook = pd.read_csv('/Users/vardanbarsegyan/Library/Mobile Documents/com~apple~CloudDocs/Papers/12_fertility_challenge/data/codebooks/PreFer_codebook.csv', 
                      encoding="cp1252", 
                      low_memory=False)

PreFer_codebook_summary = pd.read_csv('/Users/vardanbarsegyan/Library/Mobile Documents/com~apple~CloudDocs/Papers/12_fertility_challenge/data/codebooks/PreFer_codebook_summary.csv', 
                      encoding="cp1252", 
                      low_memory=False)
#%%
# Explore the types of variables
PreFer_codebook.type_var.value_counts()
PreFer_codebook.columns
PreFer_codebook.dataset.value_counts()



#%%
# Define the numeric and categoric variables
PreFer_codebook_main = PreFer_codebook[PreFer_codebook['dataset'] == 'PreFer_train_data.csv']

# Filter the codebook to get only numeric and categorical variables
PreFer_codebook_main_types = PreFer_codebook_main[PreFer_codebook_main['type_var'].isin(['numeric', 'categorical'])]

# Select these variables from the main dataset
PreFer_train_data_vartypes = PreFer_train_data[PreFer_codebook_main_types['var_name'].tolist()]

# numeric vars
numeric_vars_temp = PreFer_codebook_main_types[PreFer_codebook_main_types['type_var'] == 'numeric']
numeric_vars = numeric_vars_temp['var_name'].tolist()
numeric_vars = [numeric_vars for numeric_vars in numeric_vars if numeric_vars != 'nomem_encr']


# categorical_vars
categorical_vars_temp = PreFer_codebook_main_types[PreFer_codebook_main_types['type_var'] == 'categorical']
categorical_vars = categorical_vars_temp['var_name'].tolist()
len(categorical_vars_temp) 



#%%
# ENCODE 
# One-hot encode the categorical variables
data_encoded = pd.get_dummies(PreFer_train_data_vartypes, 
                              columns = categorical_vars)

# Standardize the numerical variables
scaler = StandardScaler()
data_encoded[numeric_vars] = scaler.fit_transform(data_encoded[numeric_vars])

#%%
# Merge the X and y vars 
merged_df = pd.merge(data_encoded, PreFer_train_outcome, on='nomem_encr', how='inner')


#%%
merged_df.new_child.value_counts()

# Replace NaN values  with 0
merged_df['new_child'] = merged_df['new_child'].fillna(0)



# Define the features (X) and the target (y)
X = merged_df.drop(columns='new_child')
y = merged_df['new_child']

# All data to train 
X_train = X.copy() 
y_train = y.copy()


#%%
# Select 500 random variables 
#X_train_selectvars = X_train.iloc[:, :500]



#%%
# Initialize the model
#model = HistGradientBoostingClassifier()


#param_grid = {
#    'learning_rate': [0.1, 0.01, 0.5]
#}


# Implement GridSearchCV
#grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
#                           cv=3, n_jobs=-1, verbose=2, scoring='f1')

# Fit the model
#grid_search.fit(X_train, y_train)


# Get the best parameters and best model from grid search
#best_params = grid_search.best_params_
#best_model = grid_search.best_estimator_
best_model = HistGradientBoostingClassifier(learning_rate=0.01)

# Fit the model
best_model.fit(X_train, y_train)


# Next
best_model.score(X_train, y_train)
best_model.predict(X_train)


# Evaluate the model 
visual1 = ConfusionMatrixDisplay.from_estimator(best_model, 
                              X_train, 
                              y_train)

best_model.score(X_train, y_train)

# Print ROC curve, it tells you how well you can balance false and true positives
RocCurveDisplay.from_predictions(
    y_train,
    best_model.predict_proba(X_train)[:, 1],
    color="cornflowerblue",
)
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()


# Save the model
joblib.dump(best_model, "best_model.joblib")


