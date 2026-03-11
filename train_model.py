import pandas as pd


# Load cleaned dataset
df = pd.read_csv("cleaned_drinks.csv")

# Input features 
X = df.drop(["total_litres_of_pure_alcohol", "country"], axis=1)

# Target variable
y = df["total_litres_of_pure_alcohol"]


# Convert categorical column to numerical columns
X = pd.get_dummies(X, columns=["continent"])


# Split dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Model 1: Linear Regression
lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train, y_train)

# Predict on test data
y_pred_lr = lr_model.predict(X_test)

# Calculate R2 score
lr_r2 = r2_score(y_test, y_pred_lr)

print("Linear Regression R2 Score:", lr_r2)

# Model 2: Random Forest
rf_model = RandomForestRegressor(random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on test data
y_pred_rf = rf_model.predict(X_test)

# R2 score
rf_r2 = r2_score(y_test, y_pred_rf)

print("Random Forest R2 Score:", rf_r2)


from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning for Random Forest
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Random Forest Parameters:", grid_search.best_params_)

# Evaluate tuned model
best_rf = grid_search.best_estimator_

y_pred_rf_tuned = best_rf.predict(X_test)

from sklearn.metrics import r2_score
rf_tuned_r2 = r2_score(y_test, y_pred_rf_tuned)

print("Tuned Random Forest R2 Score:", rf_tuned_r2)



import pickle

# Save the best model (Linear Regression)
with open("best_model.pkl", "wb") as f:
    pickle.dump(lr_model, f)

print("Best model saved as best_model.pkl")
