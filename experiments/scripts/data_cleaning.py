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
        print("Data Cleaning: Step 1: Filtering feature catalog to exclude Betting_odd=True features.......")
        logging.info("Entering the Filtering out the non betting features module.....")
        
        # Load the CSV files first
        feature_catalog_df = pd.read_csv(self.ingestion_config.feature_catalog_csv_file)
        enhanced_data_df = pd.read_csv(self.ingestion_config.ingested_csv_file,low_memory=False)
        
        # Now filter
        # Had to Convert the False to avoid the boolean confusion brought about by Singleton Comparison
        is_not_betting_odd = feature_catalog_df['Betting_odd'].astype(str).str.lower().isin(['false'])
        feature_catalog_filtered = feature_catalog_df[is_not_betting_odd].copy()
        feature_names_to_keep = feature_catalog_filtered['Feature_name'].unique()
        enhanced_data = enhanced_data_df.loc[:, enhanced_data_df.columns.isin(feature_names_to_keep)].copy()


        columns_retained = enhanced_data.columns
        logging.info(f" Here are the retained columns after removing betting odds : {columns_retained}")
        logging.info(f"The new number of columns are {enhanced_data.shape[1]} and the new number of the rows are {enhanced_data.shape[0]}" )
        return enhanced_data
    

    def removing_columns_missing_significant_chunks(self,enhanced_data):
        print("Data Cleaning:Step 2: Removing columns missing significant chunks on it.......")
        try:
            total = enhanced_data.isnull().sum().sort_values(ascending=False)
            missing_percent = (enhanced_data.isnull().sum() / len(enhanced_data)) * 100
            missing_data = pd.concat([total,missing_percent],axis=1, keys=['Total','Percent'])
            logging.info(f"These are the columns that are missing{missing_data.head(20)}")
            #Deleting the columns
            columns_to_drop = missing_percent[missing_percent > 10].index
            enhanced_data_with_no_nulls = enhanced_data.drop(columns=columns_to_drop).copy()
            print(f"Dropped {len(columns_to_drop)} columns.")
            logging.info(f"Dropped {len(columns_to_drop)} columns.")
            print(f"Dropped {columns_to_drop} columns.")
            logging.info(f"Dropped {columns_to_drop} columns.")
            logging.info(f"{enhanced_data_with_no_nulls.head()}")
        except Exception as e:
            raise CustomException(e,sys) from e
        print("Data Cleaning:Step 2: Removing columns missing significant chunks on it✅")
        return enhanced_data_with_no_nulls
    

    def declare_dtypes_early(self,enhanced_data_with_no_nulls):
        print("Data Cleaning:Step 3: Enforcing data types.......")
        try:
            logging.info("Enforcing and Augmenting Date Data types")
            enhanced_data_with_no_nulls['Date'] = pd.to_datetime(enhanced_data_with_no_nulls['Date'],format = 'mixed')
            enhanced_data_with_no_nulls['Month'] = enhanced_data_with_no_nulls['Date'].dt.strftime('%B')
            enhanced_data_with_no_nulls['Year'] = enhanced_data_with_no_nulls['Date'].dt.year
            enhanced_data_with_no_nulls['Day'] = enhanced_data_with_no_nulls['Date'].dt.strftime('%A')
            logging.info(f"Augmented date types : {enhanced_data_with_no_nulls.sample(10)}")

            # == Fixing String Columns == #
            logging.info("Enforcing string columns to avoid edge cases....")
            string_cols = ['HomeTeam', 'AwayTeam', 'FTR', 'HTR', 'Referee', 'Month', 'Day'] 
            str_map = ({col: str for col in string_cols})
            enhanced_data_with_no_nulls = enhanced_data_with_no_nulls.astype(str_map) 
            logging.info(f"Here are the data types: {enhanced_data_with_no_nulls.dtypes}")

            # == Fixing Numeric Columns == #
            logging.info("Enforcing numeric columns......")
            num_cols = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'HS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR','Year']
            num_map = ({num: 'Int64' for num in num_cols})
            enhanced_data_with_no_nulls = enhanced_data_with_no_nulls.astype(num_map)
            logging.info(f"Here are the data types: {enhanced_data_with_no_nulls.dtypes}")
        except Exception as e:
            raise CustomException(e,sys) from e
        print("Data Cleaning:Step 3: Enforcing data types.......✅")
        logging.info("Data Cleaning:Step 3: Enforcing data types.......✅")
        return enhanced_data_with_no_nulls






if __name__ == "__main__":
    obj = DataCleaning()
    enhanced_data = obj.filter_data_by_non_betting_features()
    enhanced_data_with_no_nulls = obj.removing_columns_missing_significant_chunks(enhanced_data)
    enhanced_data_with_no_nulls = obj.declare_dtypes_early(enhanced_data_with_no_nulls)

    




