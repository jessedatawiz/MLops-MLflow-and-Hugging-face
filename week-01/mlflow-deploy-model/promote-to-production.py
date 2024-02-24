# promote_to_production.py
import mlflow

model_name = "LinearRegressionModel"
version = 2  # This would be set appropriately based on the model's version

client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=version,
    stage="Production",
    archive_existing_versions=True
)
