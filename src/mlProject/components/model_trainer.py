import os
import joblib
import pandas as pd
from mlProject import logger
from mlProject.entity.config_entity import ModelTrainerConfig

from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.svm import SVR

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        #test_data = pd.read_csv(self.config.test_data_path)

        X_train = train_data.drop([self.config.target_column], axis=1)
        #X_test = train_data.drop([self.config.target_column], axis=1)
        y_train = train_data[[self.config.target_column]]
        #y_test = test_data[[self.config.target_column]]
        
        if(self.config.train_elastic_net):
            logger.info(f"Currently training {self.config.model_name_elastic_net}.")
            en = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
            en.fit(X_train, y_train)

            logger.info("Training complete!")
            joblib.dump(en, os.path.join(self.config.root_dir, self.config.model_name_elastic_net))
        else:
            logger.info(f"{self.config.model_name_elastic_net} was not trained.")

        if(self.config.train_linear_regression):
            logger.info(f"Currently training {self.config.model_name_linear_regression}.")
            lr = LinearRegression()
            lr.fit(X_train, y_train)

            logger.info("Training complete!")
            joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name_linear_regression))
        else:
            logger.info(f"{self.config.model_name_linear_regression} was not trained.")

        if(self.config.train_linear_regression):
            logger.info(f"Currently training {self.config.model_name_support_vector}.")
            svr = SVR(C=self.config.C)
            svr.fit(X_train, y_train)

            logger.info("Training complete!")
            joblib.dump(svr, os.path.join(self.config.root_dir, self.config.model_name_support_vector))
        else:
            logger.info(f"{self.config.model_name_support_vector} was not trained.")

        
