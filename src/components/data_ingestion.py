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
    project_root = Path(__file__).resolve().parent.parent.parent  # jnyambok-soca-scores/
    data_dir = project_root / "datasets" 
    common_data_dir = data_dir / "common_data"
    csv_file = common_data_dir / "english_league_data_urls.csv"

# ===== DATA SOURCE IDENTIFICATION =====#
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def load_csv(self):
        print("Identifying the data source...............")
        logging.info("Entered the loading csv microcomponent")
        try:
            if self.ingestion_config.csv_file.exists():
                logging.info(f"Found file: {self.ingestion_config.csv_file}")
                urls = pd.read_csv(self.ingestion_config.csv_file)
                logging.info(f'File has been loaded successfully. The url dataset has {urls.shape[0]} rows and {urls.shape[1]} columns')
            else:
                logging.info(f"File not found: {self.ingestion_config.csv_file}")
        except Exception as e:
            raise CustomException(e, sys) from e
        print("Step 1 complete: Data Source has been identified!")
        return urls

# ===== APPENDING METADATA - ADDING COMPETITION NAME AND SEASON ID =====#
    def add_metadata(self,urls):
        print("Adding metadata to urls")
        logging.info("Entering the adding metadata microcomponent")
        dataframes=[]
        for index,row in urls.iterrows():
            season_id = row["Season_ID"]
            seasons_url = row["Seasons_url"]
            competition_name = row["Competition_name"]
            logging.info(f"\n Processing: {season_id} - {competition_name}")
            logging.info(f"URL : {seasons_url}")
            # === Metadata collected ===#

            try:
                epl_data = pd.read_csv(seasons_url)
                epl_data["season_id"] = season_id
                epl_data["competition_name"] = competition_name
                dataframes.append(epl_data)
                logging.info(f"Success: {epl_data.shape[0]} matches loaded")
                logging.info(f" Columns: {epl_data.shape[1]} (including metadata)")
            except Exception as e:
                raise CustomException(e,sys) from e
        print("Step 2: Metadata successfully appended!")
        return dataframes



if __name__ == "__main__":
    obj = DataIngestion()
    urls = obj.load_csv()
    obj.add_metadata(urls)


