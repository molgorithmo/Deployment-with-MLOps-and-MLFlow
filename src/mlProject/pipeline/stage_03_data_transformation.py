from mlProject.config.configuration import ConfigurationManager
from mlProject.components.data_transformation import DataTransformation
from mlProject import logger
from pathlib import Path

STAGE_NAME = "Data Transformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(config=data_transformation_config)
                data_transformation.train_test_splitting()
            else:
                raise Exception("The schema of the data is not valid.")
            
        except Exception as e:
            print(e)

if __name__ == '__main__':
    try:
        logger.info(f"/////////////// Stage {STAGE_NAME} has started. ///////////////")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f"/////////////// Stage {STAGE_NAME} completed. ///////////////")
    except Exception as e:
        logger.exception(e)
        raise e
