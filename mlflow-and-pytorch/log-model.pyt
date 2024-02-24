import mlflow.pytorch

# Log the model
with mlflow.start_run() as run:
    mlflow.pytorch.log_model(model, "model")
