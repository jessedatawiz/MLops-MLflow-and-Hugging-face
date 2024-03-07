import argparse
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Random Forest Regressor training with hyperparameter tuning')
parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in the forest')
parser.add_argument('--max_depth', type=int, default=None, help='Maximum depth of the tree')
args = parser.parse_args()

# Generate synthetic data
X, y = np.random.rand(100, 10), np.random.rand(100)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [args.n_estimators],
    'max_depth': [args.max_depth] if args.max_depth else [None, 10, 20, 30]
}

# Initialize the RandomForestRegressor
rf = RandomForestRegressor(random_state=42)

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

LOCAL_PORT = 'http://127.0.0.1:5000'
mlflow.set_tracking_uri(LOCAL_PORT)

# Start MLflow run for tracking
with mlflow.start_run():
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Log parameters, metrics, and model
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("mse", -grid_search.best_score_)
    mlflow.sklearn.log_model(grid_search.best_estimator_, "model")
    
    # Make predictions and calculate test MSE
    predictions = grid_search.predict(X_test)
    mse_test = mean_squared_error(y_test, predictions)
    
    # Log test MSE
    mlflow.log_metric("mse_test", mse_test)

    # Print out metrics
    print(f"MSE (CV): {-grid_search.best_score_}")
    print(f"MSE (Test): {mse_test}")
