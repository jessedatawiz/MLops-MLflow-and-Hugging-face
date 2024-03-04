import mlflow

client = mlflow.tracking.MlflowClient()
model_name = 'Iris Random Forest'
model_version = 1  # This might be different for your model

# Transition the model version to the Staging stage
client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage="Staging"
)
