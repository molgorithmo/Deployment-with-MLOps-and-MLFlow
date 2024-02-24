from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: Path
    local_data_file: Path
    unzip_dir: Path
    unzip_data_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name_elastic_net: str
    model_name_linear_regression: str
    model_name_support_vector: str
    alpha: float
    l1_ratio: float
    C: float
    target_column: str
    train_elastic_net: bool
    train_linear_regression: bool
    train_support_vector_machine: bool

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    elastic_net_model_path: Path
    linear_regression_model_path: Path
    support_vector_machine_model_path: Path
    train_elastic_net: bool
    train_linear_regression: bool
    train_support_vector_machine: bool
    all_params: dict
    metrics_file_name: Path
    target_column: str
    mlflow_uri: str