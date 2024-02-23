from mlflow import log_param, log_artifact, log_metric
import mlflow

if __name__ == '__main__':

    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    log_param('threshold', 3)
    log_param('verbosity', "DEBUG")

    log_metric('timestamp', 10000)
    log_metric('TTC', 33)

    log_artifact('test-dataset.csv') 

