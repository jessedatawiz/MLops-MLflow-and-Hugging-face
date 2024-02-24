# register_model.py
import mlflow

run_id = "b3b9c7506b694bd0b8286789518f9a26"
model_uri = f"runs:/{run_id}/model"
model_name = "LinearRegressionModel"
mlflow.register_model(model_uri=model_uri, name=model_name)
