from mlProject import logger
from mlProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from mlProject.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from mlProject.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from mlProject.pipeline.stage_04_model_training import ModelTrainerTrainingPipeline
from mlProject.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline

STAGE_NAME = "Main"

try:
    logger.info(f"/////////////// Stage: {STAGE_NAME} has started. ///////////////")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.main()
    model_training = ModelTrainerTrainingPipeline()
    model_training.main()
    model_evaluation = ModelEvaluationTrainingPipeline()
    model_evaluation.main()
    logger.info(f"/////////////// Stage: {STAGE_NAME} is completed. ///////////////")
except Exception as e:
    logger.exception(e)
    raise e
