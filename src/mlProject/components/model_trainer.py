import os
import joblib
import pandas as pd
from mlProject import logger
from mlProject.entity.config_entity import ModelTrainerConfig

from sklearn.linear_model import ElasticNet

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        X_train = train_data.drop([self.config.target_column], axis=1)
        #X_test = train_data.drop([self.config.target_column], axis=1)
        y_train = train_data[[self.config.target_column]]
        #y_test = test_data[[self.config.target_column]]
        
        if(self.config.train_elastic_net):
            logger.info(f"Currently training {self.config.model_name}.")
            lr = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
            lr.fit(X_train, y_train)
            logger.info("Training complete!")
        else:
            logger.info(f"{self.config.model_name} was not trained.")

        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))