import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

LOCAL_PORT = 'http://127.0.0.1:5000'
mlflow.set_tracking_uri(LOCAL_PORT)

# Enable autologging
mlflow.sklearn.autolog()

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Train a model
with mlflow.start_run(run_name="RF_Iris_Model"):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {accuracy}")

    # Log parameters, metrics, and model
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

    # Register the Model
    client = mlflow.tracking.MlflowClient()
    client.create_registered_model("sk-learn-random-forest-reg-model")
