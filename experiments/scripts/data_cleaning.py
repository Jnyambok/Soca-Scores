#==== Let's ingest the data ===== #

import sys
import pandas as pd
from pathlib import Path

from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass


@dataclass
class DataCleaningConfig:
    project_root = Path(__file__).resolve().parent.parent.parent  # jnyambok-soca-scores/
    data_dir = project_root / "datasets" 
    common_data_dir = data_dir / "common_data"
    ingested_dir = data_dir / "ingested_data"
    csv_file = common_data_dir / "english_league_data_urls.csv"
    ingested_csv_file = ingested_dir / "enhanced_dataset.csv"
    feature_catalog_csv_file = common_data_dir / "feature_catalog.csv"

class DataCleaning:
    def __init__(self):
        self.ingestion_config = DataCleaningConfig()
    

    def filter_data_by_non_betting_features(self):
        print("Stage 1: Filtering feature catalog to exclude Betting_odd=True features.")
        
        # Load the CSV files first
        feature_catalog_df = pd.read_csv(self.ingestion_config.feature_catalog_csv_file)
        enhanced_data_df = pd.read_csv(self.ingestion_config.ingested_csv_file,low_memory=False)
        
        # Now filter
        feature_catalog_filtered = feature_catalog_df[feature_catalog_df["Betting_odd"]==False].copy()
        feature_names_to_keep = feature_catalog_filtered['Feature_name'].unique()
        enhanced_data = enhanced_data_df.loc[:, enhanced_data_df.columns.isin(feature_names_to_keep)].copy()
        
        return print(enhanced_data.columns)



if __name__ == "__main__":
    obj = DataCleaning()
    cleaned = obj.filter_data_by_non_betting_features()
    




