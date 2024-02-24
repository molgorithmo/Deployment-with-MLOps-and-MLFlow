from mlProject.constants import *
from mlProject.utils.common import read_yaml, create_directories
from mlProject.entity.config_entity import *

class ConfigurationManager:
    def __init__(
            self,
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH,
            schema_filepath = SCHEMA_FILE_PATH,
            model_train_filepath = MODEL_TRAIN_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        self.model_train = read_yaml(model_train_filepath)

        create_directories([self.config.artifact_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
            unzip_data_dir=config.unzip_data_dir
        )

        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir = config.unzip_data_dir,
            all_schema=schema
        )

        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path
        )

        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        elastic_net_params = self.params.ElasticNet
        support_vector_params = self.params.SupportVectorMachine
        schema = self.schema.TARGET_COLUMN
        model_train = self.model_train

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            model_name_elastic_net=config.model_name_elastic_net,
            model_name_linear_regression=config.model_name_linear_regression,
            model_name_support_vector=config.model_name_support_vector,
            C=support_vector_params.C,
            alpha=elastic_net_params.alpha,
            l1_ratio=elastic_net_params.l1_ratio,
            target_column=schema.name,
            train_elastic_net=model_train.train_elastic_net,
            train_linear_regression=model_train.train_linear_regression,
            train_support_vector_machine=model_train.train_support_vector_machine
        )

        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params
        schema = self.schema.TARGET_COLUMN
        model_train = self.model_train

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            elastic_net_model_path=config.elastic_net_model_path,
            linear_regression_model_path=config.linear_regression_model_path,
            support_vector_machine_model_path=config.support_vector_machine_model_path,
            train_elastic_net=model_train.train_elastic_net,
            train_linear_regression=model_train.train_linear_regression,
            train_support_vector_machine=model_train.train_support_vector_machine,
            all_params=params,
            metrics_file_name=config.metrics_file_name,
            target_column=schema.name,
            mlflow_uri="https://dagshub.com/molgorithmo/Deployment-with-MLOps-and-MLFlow.mlflow"
        )

        return model_evaluation_config


