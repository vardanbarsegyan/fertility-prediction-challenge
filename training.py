"""
This is an example script to train your model given the (cleaned) input dataset.

This script will not be run on the holdout data, 
but the resulting model model.joblib will be applied to the holdout data.

It is important to document your training steps here, including seed, 
number of folds, model, et cetera
"""

def train_save_model(cleaned_df, outcome_df):
    """
    Trains a model using the cleaned dataframe and saves the model to a file.

    Parameters:
    cleaned_df (pd.DataFrame): The cleaned data from clean_df function to be used for training the model.
    outcome_df (pd.DataFrame): The data with the outcome variable (e.g., from PreFer_train_outcome.csv or PreFer_fake_outcome.csv).
    """
    
    
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

