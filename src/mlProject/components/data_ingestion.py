import os
import urllib.request as request
import zipfile
from mlProject import logger
from mlProject.utils.common import get_size
from mlProject.config.configuration import DataIngestionConfig
from pathlib import Path

class DataIngestion:
    def __init__(self, config:DataIngestionConfig) -> None:
        self.config = config
    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url=self.config.source_url,
                filename=self.config.local_data_file
            )
            logger.info(f"{filename} downladed! with following info: {headers}")
        else:
            logger.info(f"File already exists with file size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        """
        This method extract the files from the given path.
        """
        if not os.path.exists(self.config.unzip_data_dir):
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
        else:
            logger.info("File already unzipped!")