import os
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from mlProject.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
    
    def train_test_splitting(self):
        data = pd.read_csv(self.config.data_path)

        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"))
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"))

        logger.info(f"<<< Splitted data into train and test sets\nShape of train data: {train.shape}\nShape of test data: {test.shape}>>>")