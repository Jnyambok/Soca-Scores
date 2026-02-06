#==== Let's ingest the data ===== #

"""
Module for cleaning the English Premier League Data and ingesting it into the Neon DB 

"""



#Standard imports
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List

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

    critical_columns: List[str] = ('HomeTeam', 'AwayTeam', 'Date', 'FTHG', 'FTAG')
    string_columns: List[str] = ('HomeTeam', 'AwayTeam', 'FTR', 'HTR', 'Referee', 'Month', 'Day')
    numeric_columns: List[str] = ('FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'HST', 'AST','HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'Year')

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
            logging.info("======================================================")
            logging.info("Data Cleaning Step 1: Starting the deletion of betting features.....")

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
            logging.info("Data Cleaning Step 1: Betting features succcessfully deleted. STEP COMPLETED")
            logging.info("===============================================================")
            print("Data Cleaning: Step 1: Betting features have been deleted...STEP 1 COMPLETED")
            return enhanced_data
        except Exception as e:
            raise CustomException(e,sys) from e
    
    def handle_missing_data(self, enhanced_data:pd.DataFrame, threshold:float=10.0)-> pd.DataFrame:
        """
        Step 2: Handling missing data
        Removes rows that are missing a significant chunk
        :params : self, enhanced_data, threshold value -> value to ensure
        :rtype:  enhanced_data Dataframe -> Missing data has been handled

        """
        try:
            print("Data Cleaning: Step 2: Handling data features missing significant chunks of volume.......")
            logging.info("======================================================")
            logging.info("Data Cleaning Step 2: Handling of missing features.....")

            #Dropping columns with > threshold missing
            total = enhanced_data.isnull().sum().sort_values(ascending=False)
            missing_percent = (enhanced_data.isnull().sum() / len(enhanced_data)) * 100
            missing_data = pd.concat([total,missing_percent],axis=1, keys=['Total','Percent'])
            logging.info(f"These are the columns that are missing{missing_data.head(20)}")

            cols_to_drop = missing_percent[missing_percent>threshold].index
            enhanced_data = enhanced_data.drop(columns=cols_to_drop)
            print(f"Dropped {len(cols_to_drop)} columns.")
            logging.info(f"Dropped {len(cols_to_drop)} columns.")
            print(f"Dropped {cols_to_drop} columns.")
            logging.info(f"Dropped {cols_to_drop} columns.")
            logging.info(f"{enhanced_data.head()}")

            #Handling Nullable types
            enhanced_data_with_no_nulls = enhanced_data.dropna(subset=list(self.cleaning_config.critical_columns))
            logging.info(f"Original shape: {enhanced_data.shape}")
            logging.info(f"New shape: {enhanced_data_with_no_nulls.shape}")           


            logging.info("Data Cleaning Step 2: Data missing significant chunk of data and nulls handled. STEP COMPLETED")
            logging.info("===============================================================")
            #return enhanced_data
            #print("Data Cleaning: Step 3: Data missing significant chunk of data handled...STEP 2 COMPLETED")
       
        except Exception as e:
            raise CustomException(e,sys) from e
        print("Data Cleaning: Step 2: Data missing significant chunk of data handled...STEP 2 COMPLETED")
        return enhanced_data_with_no_nulls
    
    def declare_dtypes_early(self,enhanced_data_with_no_nulls:pd.DataFrame)->pd.DataFrame:  #Not a pure function but falls in "standardization"
        """
        Step 3: Standardizing data types and augments data features
        :params -> Enhanced_data_with_no_nulls
        :rtyps -> standardized_data_with_no_nulls

        """
        try:
            print("Data Cleaning: Step 3: Standardizing data types and augmenting date features.......")
            logging.info("======================================================")
            logging.info("Data Cleaning Step 3: Standardizing data types.....")

            #Standardizing - Handling date types and augmenting them
            enhanced_data_with_no_nulls['Date'] = pd.to_datetime(enhanced_data_with_no_nulls['Date'],format = 'mixed')
            enhanced_data_with_no_nulls['Month'] = enhanced_data_with_no_nulls['Date'].dt.strftime('%B')
            enhanced_data_with_no_nulls['Year'] = enhanced_data_with_no_nulls['Date'].dt.year
            enhanced_data_with_no_nulls['Day'] = enhanced_data_with_no_nulls['Date'].dt.strftime('%A')
            logging.info(f"Augmented date types : {enhanced_data_with_no_nulls.sample(10)}")

            #Standardizing - vectorized type conversion
            logging.info("Standardizing String Columns")
            for col in self.cleaning_config.string_columns:
                if col in enhanced_data_with_no_nulls.columns:
                    enhanced_data_with_no_nulls[col] = enhanced_data_with_no_nulls[col].astype(str).str.strip()
            
            #Standardizing - numeric columns
            logging.info("Standarding Numeric Columns")
            for col in self.cleaning_config.numeric_columns:
                if col in enhanced_data_with_no_nulls.columns:
                    enhanced_data_with_no_nulls[col] = pd.to_numeric(
                        enhanced_data_with_no_nulls[col],errors="coerce").astype("Int64")
            logging.info("======================================================")
            logging.info("Data Cleaning Step 3: Standardized data types handled. STEP COMPLETED")
            
            return enhanced_data_with_no_nulls
        
        except Exception as e:
            raise CustomException(e,sys) from e
        





if __name__ == "__main__":
    obj = DataCleaning()
    enhanced = obj.remove_betting_features()
    enhanced_no_nulls = obj.handle_missing_data(enhanced)
    standardized_no_nulls = obj.declare_dtypes_early(enhanced_no_nulls)
