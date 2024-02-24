import pandas as pd
import numpy as np
import joblib
from urllib.parse import urlparse
from pathlib import Path

from mlProject.config.configuration import ModelEvaluationConfig
from mlProject.utils.common import save_json

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow
import mlflow.sklearn


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
    
    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(y_pred=pred, y_true=actual))
        mae = mean_absolute_error(y_pred=pred, y_true=actual)
        r2 = r2_score(actual, pred)

        return rmse, mae, r2
        
    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        X_test = test_data.drop([self.config.target_column], axis=1)
        y_test = test_data[[self.config.target_column]]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():

            predicted_qualities = model.predict(X_test)

            (rmse, mae, r2) = self.eval_metrics(y_test, predicted_qualities)

            # Saving metrics as local
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metrics_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticNet Model")
            else:
                mlflow.sklearn.log_model(model, "model")