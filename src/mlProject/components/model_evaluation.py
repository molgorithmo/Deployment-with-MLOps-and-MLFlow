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
        elastic_net_model = joblib.load(self.config.elastic_net_model_path)
        linear_regression_model =joblib.load(self.config.linear_regression_model_path)
        support_vector_machine_model = joblib.load(self.config.support_vector_machine_model_path)

        X_test = test_data.drop([self.config.target_column], axis=1)
        y_test = test_data[[self.config.target_column]]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():

            if(self.config.train_elastic_net):
                predicted_qualities = elastic_net_model.predict(X_test)
            

                (rmse, mae, r2) = self.eval_metrics(y_test, predicted_qualities)

                # Saving metrics as local
                scores = {"en_rmse": rmse, "en_mae": mae, "en_r2": r2}
                save_json(path=Path(self.config.metrics_file_name), data=scores)

                mlflow.log_params(self.config.all_params)

                mlflow.log_metric("en_rmse", rmse)
                mlflow.log_metric("en_r2", r2)
                mlflow.log_metric("en_mae", mae)

                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(elastic_net_model, "model", registered_model_name="ElasticNet Model")
                else:
                    mlflow.sklearn.log_model(elastic_net_model, "model")

            if(self.config.train_linear_regression):
                predicted_qualities = linear_regression_model.predict(X_test)

                (rmse, mae, r2) = self.eval_metrics(y_test, predicted_qualities)

                # Saving metrics as local
                scores = {"lr_rmse": rmse, "lr_mae": mae, "lr_r2": r2}
                save_json(path=Path(self.config.metrics_file_name), data=scores)

                mlflow.log_params(self.config.all_params)

                mlflow.log_metric("lr_rmse", rmse)
                mlflow.log_metric("lr_r2", r2)
                mlflow.log_metric("lr_mae", mae)

                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(linear_regression_model, "model", registered_model_name="Linear Regression Model")
                else:
                    mlflow.sklearn.log_model(linear_regression_model, "model")
            
            if(self.config.train_support_vector_machine):
                predicted_qualities = support_vector_machine_model.predict(X_test)

                (rmse, mae, r2) = self.eval_metrics(y_test, predicted_qualities)

                # Saving metrics as local
                scores = {"svr_rmse": rmse, "svr_mae": mae, "svr_r2": r2}
                save_json(path=Path(self.config.metrics_file_name), data=scores)

                mlflow.log_params(self.config.all_params)

                mlflow.log_metric("svr_rmse", rmse)
                mlflow.log_metric("svr_r2", r2)
                mlflow.log_metric("svr_mae", mae)

                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(support_vector_machine_model, "model", registered_model_name="Support Vector Regressor Model")
                else:
                    mlflow.sklearn.log_model(support_vector_machine_model, "model")