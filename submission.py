"""
This is an example script to generate the outcome variable given the input dataset.

This script should be modified to prepare your own submission that predicts 
the outcome for the benchmark challenge by changing the clean_df and predict_outcomes function.

The predict_outcomes function takes a Pandas data frame. The return value must
be a data frame with two columns: nomem_encr and outcome. The nomem_encr column
should contain the nomem_encr column from the input data frame. The outcome
column should contain the predicted outcome for each nomem_encr. The outcome
should be 0 (no child) or 1 (having a child).

clean_df should be used to clean (preprocess) the data.

run.py can be used to test your submission.
"""

# List your libraries and modules here. Don't forget to update environment.yml!
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib



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



def clean_df(df, background_df=None):
    """
    Preprocess the input dataframe to feed the model.
    # If no cleaning is done (e.g. if all the cleaning is done in a pipeline) leave only the "return df" command

    Parameters:
    df (pd.DataFrame): The input dataframe containing the raw data (e.g., from PreFer_train_data.csv or PreFer_fake_data.csv).
    background (pd.DataFrame): Optional input dataframe containing background data (e.g., from PreFer_train_background_data.csv or PreFer_fake_background_data.csv).

    Returns:
    pd.DataFrame: The cleaned dataframe with only the necessary columns and processed variables.
    """


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


    return df


def predict_outcomes(df, background_df=None, model_path="model.joblib"):
    """Generate predictions using the saved model and the input dataframe.

    The predict_outcomes function accepts a Pandas DataFrame as an argument
    and returns a new DataFrame with two columns: nomem_encr and
    prediction. The nomem_encr column in the new DataFrame replicates the
    corresponding column from the input DataFrame. The prediction
    column contains predictions for each corresponding nomem_encr. Each
    prediction is represented as a binary value: '0' indicates that the
    individual did not have a child during 2021-2023, while '1' implies that
    they did.

    Parameters:
    df (pd.DataFrame): The input dataframe for which predictions are to be made.
    background_df (pd.DataFrame): The background dataframe for which predictions are to be made.
    model_path (str): The path to the saved model file (which is the output of training.py).

    Returns:
    pd.DataFrame: A dataframe containing the identifiers and their corresponding predictions.
    """

    ## This script contains a bare minimum working example
    if "nomem_encr" not in df.columns:
        print("The identifier variable 'nomem_encr' should be in the dataset")

    # Load the model
    model = joblib.load(model_path)

    # Preprocess the fake / holdout data
    df = clean_df(df, background_df)

    # Exclude the variable nomem_encr if this variable is NOT in your model
    vars_without_id = df.columns[df.columns != 'nomem_encr']

    # Generate predictions from model, should be 0 (no child) or 1 (had child)
    predictions = model.predict(df[vars_without_id])

    # Output file should be DataFrame with two columns, nomem_encr and predictions
    df_predict = pd.DataFrame(
        {"nomem_encr": df["nomem_encr"], "prediction": predictions}
    )

    # Return only dataset with predictions and identifier
    return df_predict
