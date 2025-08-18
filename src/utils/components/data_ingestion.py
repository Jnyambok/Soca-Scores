#==== Let's ingest the data ===== #
import os
import sys
import pandas as pd
from pathlib import Path

from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    project_root = Path().absolute().parent.parent
    data_dir = project_root / "datasets" 
    common_data_dir = data_dir / "common_data"
    csv_file = common_data_dir / "english_league_data_url.csv"

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def load_csv(self):
        logging.info("Entered the data ingestion component")
        try:
            if self.ingestion_config.csv_file.exists():
                logging.info(f"✅ Found file: {self.ingestion_config.csv_file}")
                urls = pd.read_csv(self.ingestion_config.csv_file)
                logging.info(f'📊File has been loaded successfully. The url dataset has {urls.shape[0]} rows and {urls.shape[1]} columns')
            else:
                print(f"❌ File not found: {self.ingestion_config.csv_file}")
        except Exception as e:
            raise CustomException(e,sys)


if __name__ == "__main__":
    obj = DataIngestion()



