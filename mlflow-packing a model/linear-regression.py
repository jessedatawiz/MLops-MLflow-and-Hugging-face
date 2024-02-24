import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import mlflow

LOCAL_PORT = 'http://127.0.0.1:5000'
mlflow.set_tracking_uri(LOCAL_PORT)

# Generate synthetic data
X = np.array([[i] for i in range(-100, 100)])
y = np.array([2 * i**2 + np.random.randn() for i in range(-100, 100)])

# Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create linear regression object
regr = LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))

# Plot outputs
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.savefig("regression_line.png")

# Log experiment with MLflow
with mlflow.start_run():
    mlflow.log_param("coef", regr.coef_[0])
    mlflow.log_metric("mse", mean_squared_error(y_test, y_pred))
    mlflow.log_artifact("regression_line.png")
    mlflow.set_tag("my_custom_run_name", "Special_Run_01")

print("Experiment logged with MLflow.")
