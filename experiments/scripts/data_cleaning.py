#==== Let's ingest the data ===== #

"""
Module for cleaning the English Premier League Data and ingesting it into the Neon DB 

"""

#Used when PEP8 import order is not respected
#(standard imports first, then third-party libraries,
#then local imports).

#Standard imports
import sys
from pathlib import Path
from dataclasses import dataclass

#Third Party imports
import pandas as pd
from src.logger import logging

#local imports
from src.exception import CustomException



@dataclass
class DataCleaningConfig:
    """
    Contains configuration paths for data cleaning operations.

    Attributes:
        project_root: Root directory of the project
        data_dir: Main datasets directory
        common_data_dir: Directoty for common/shared data
        ingested_dir: Directory for ingested/processed data from the .csv files
        csv_file: Path to the URLs CSV file
    """

    project_root:Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = None
    common_data_dir: Path = None
    ingested_dir: Path = None
    ingested_csv: Path = None
    csv_file: Path = None
    feature_catalog:Path = None

    def __post_init__(self) -> None:
        """
        Initialize computed attributes after dataclass initialization.
        
        """
        self.data_dir = self.project_root / "datasets"
        self.common_data_dir = self.data_dir / "common_data"
        self.ingested_dir = self.data_dir / "ingested_data"
        self.csv_file = self.common_data_dir / "english_league_data_urls.csv"
        self.feature_catalog = self.common_data_dir/ "feature_catalog.csv"
        self.ingested_csv = self.ingested_dir / "enhanced_dataset.csv"

        #Create directories if they dont exist
        self.ingested_dir.mkdir(parents=True,exist_ok=True)


class DataCleaning:
    """
    Contains various methods for data cleaning

    This module follows these steps
    1. Load URLS from csv

    """
    def __init__(self) -> None:
        """
       Initialize Data Ingestion with configuration
       
        :param self: Description
        """
        self.cleaning_config = DataCleaningConfig()
   
    def remove_betting_features(self) -> pd.DataFrame:
        """
        Step 1: Removing unnecessary betting features
        
        :param self: Description
        :return: DataFrame with bookmaker's odds removed
        :rtype: DataFrame
        """
        try:
            print("Data Cleaning: Step 1: Filtering feature catalog to exclude Betting_odd=True features.......")
            logging.info("Starting the deletion of betting features.....")

            #Loading the CSV files
            catalog = pd.read_csv(self.cleaning_config.feature_catalog)
            ingested_data_df = pd.read_csv(self.cleaning_config.ingested_csv,low_memory=False)

            #Filtering logic: Convert to String to avoid boolean singelton comparison issues
            is_not_betting_feature = catalog['Betting_odd'].astype(str).str.lower().isin(['false'])
            feature_catalog_filtered = catalog[is_not_betting_feature].copy()
            feature_names_to_keep = feature_catalog_filtered['Feature_name'].unique()
            enhanced_data = ingested_data_df.loc[:, ingested_data_df.columns.isin(feature_names_to_keep)].copy()

            columns_retained = enhanced_data.columns
            logging.info(f" Here are the retained columns after removing betting odds : {columns_retained}")
            logging.info(f"The new number of columns are {enhanced_data.shape[1]} and the new number of the rows are {enhanced_data.shape[0]}" )
            return enhanced_data

        except Exception as e:
            raise CustomException(e,sys) from e


if __name__ == "__main__":
    obj = DataCleaning()
    enhanced = obj.remove_betting_features()
