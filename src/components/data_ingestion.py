#==== Let's ingest the data ===== #

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
    ingested_dir = data_dir / "ingested_data"
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
        print("Adding metadata to urls.........")
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
    
# ===== COMBINING ALL THE DATASETS =====#
    def combine_dataframes(self,dataframes):
        try:
            print("Combining all the datasets.......")
            final_df = pd.concat(dataframes,ignore_index=True)
            final_df = final_df.drop('Div',axis=1)
            logging.info("Datasets successfully ingested and concatenated!")
            logging.info(f"Final dataset shape : {final_df.shape}")
            #Validating the final dataset with metadata
            display_columns = ['season_id','competition_name','Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR']
            available_columns = [ col for col in display_columns if col in final_df.columns]
            logging.info(final_df[available_columns].sample(10).to_string(index=False))
        except Exception as e:
            raise CustomException(e,sys) from e
        print("Step 3 : Datasets successfully concatenated!")
        return final_df
    

# ===== SAVING OUTPUT =====#
    def save_datasets(self,final_df):
        print("Saving the concatenated datasets.......")
        try:
            output_dir = self.ingestion_config.ingested_dir
            output_dir.parent.mkdir(exist_ok=True)
            output_file = output_dir/ "enhanced_dataset.csv"

            final_df.to_csv(output_file,index=False)
            logging.info(f"Enhanced Dataset saved to: {output_file}")
        except Exception as e:
            raise CustomException(e,sys) from e
        print("Step 4: Enhanced dataset saved to ingested folder!")




if __name__ == "__main__":
    obj = DataIngestion()
    urls = obj.load_csv()
    dataframes = obj.add_metadata(urls)
    final_df=obj.combine_dataframes(dataframes)
    obj.save_datasets(final_df)
