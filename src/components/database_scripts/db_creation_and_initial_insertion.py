"""
Database Insertion Script for EPL Data

This module handles all database operations for the Soca-Scores project,
including connection management, table creation, and data insertion.

Author: Data Engineering Team
Date: January 2026
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from contextlib import contextmanager

import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
import os

# Local imports
from src.logger import logging
from src.exception import CustomException


@dataclass
class DatabaseConfig:
    """Configuration for database operations."""
    db_connection_string: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Load database credentials from environment variables."""
        load_dotenv()
        self.db_connection_string = os.getenv('NEON_CONNECTION_STRING')
        
        if not self.db_connection_string:
            raise ValueError("NEON_CONNECTION_STRING not found in environment variables")


class DatabaseInsertion:
    """Handles all database insertion and cleaning operations."""
    
    TABLE_NAME = 'epl_data'
    
    # Column configuration
    COLUMN_ORDER = [
        'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 
        'HTHG', 'HTAG', 'HTR', 'Referee', 'HomeShots', 'AwayShots', 
        'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 
        'Month', 'Year', 'Day'
    ]
    
    # Column rename mapping
    COLUMN_RENAMES = {
        'HS': 'HomeShots',
        'AS': 'AwayShots'
    }
    
    CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS epl_data(
            id SERIAL PRIMARY KEY,
            date DATE NULL,
            hometeam VARCHAR(255) NOT NULL,
            awayteam VARCHAR(255) NOT NULL,
            fthg SMALLINT,
            ftag SMALLINT,
            ftr CHAR(1),
            hthg SMALLINT,
            htag SMALLINT,
            htr CHAR(1),
            referee VARCHAR(255),
            homeshots SMALLINT,
            awayshots SMALLINT,
            hst SMALLINT,
            ast SMALLINT,
            hf SMALLINT,
            af SMALLINT,
            hc SMALLINT,
            ac SMALLINT,
            hy SMALLINT,
            ay SMALLINT,
            hr SMALLINT,
            ar SMALLINT,
            month VARCHAR(255),
            year SMALLINT,
            day VARCHAR(255)
        );
    """
    
    def __init__(self) -> None:
        """Initialize database configuration."""
        self.db_config = DatabaseConfig()
        logging.info("Database configuration initialized")
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        Ensures proper cleanup even if errors occur.
        
        Yields:
            psycopg2.connection: Database connection object
            
        Raises:
            CustomException: If connection fails
        """
        conn = None
        try:
            conn = psycopg2.connect(self.db_config.db_connection_string)
            logging.info("✅ Database connection established")
            yield conn
        except Exception as e:
            logging.error(f"❌ Database connection failed: {e}")
            raise CustomException(e, sys) from e
        finally:
            if conn is not None:
                conn.close()
                logging.info("🔒 Database connection closed")
    
    def ensure_table_exists(self, conn) -> None:
        """
        Create table if it doesn't exist.
        
        Args:
            conn: Database connection
            
        Raises:
            CustomException: If table creation fails
        """
        try:
            with conn.cursor() as cur:
                cur.execute(self.CREATE_TABLE_SQL)
                conn.commit()
                logging.info("✅ Table 'epl_data' ensured to exist")
        except Exception as e:
            conn.rollback()
            logging.error(f"❌ Table creation failed: {e}")
            raise CustomException(e, sys) from e
   
    def prepare_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, list]:
        """
        Prepare data for insertion by renaming, reordering, and converting to tuples.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (cleaned DataFrame, list of tuples for insertion)
            
        Raises:
            CustomException: If data preparation fails
        """
        try:
            logging.info(f"Preparing data for insertion: {data.shape}")
            
            # Make a copy to avoid SettingWithCopyWarning
            data = data.copy()
            data = data.reset_index(drop=True)
            
            # Rename columns
            data.rename(columns=self.COLUMN_RENAMES, inplace=True)
            logging.info("✅ Columns renamed")
            
            # Reorder columns to match database schema
            available_columns = [col for col in self.COLUMN_ORDER if col in data.columns]
            data = data[available_columns]
            logging.info(f"✅ Columns reordered: {len(available_columns)} columns")
            
            # Validate column count
            if len(available_columns) == 0:
                raise ValueError("No valid columns found in DataFrame")
            
            # Convert to tuples for batch insertion
            data_to_insert = [tuple(row) for row in data.values]
            logging.info(f"✅ Created {len(data_to_insert)} tuples for insertion")
            
            return data, data_to_insert
            
        except Exception as e:
            logging.error(f"❌ Data preparation failed: {e}")
            raise CustomException(e, sys) from e
    
    def insert_data(self, conn, data: pd.DataFrame, data_tuples: list) -> int:
        """
        Insert cleaned data into the database using batch insertion.
        
        Args:
            conn: Database connection
            data: DataFrame with column information
            data_tuples: List of tuples to insert
            
        Returns:
            Number of rows inserted
            
        Raises:
            CustomException: If insertion fails
        """
        try:
            with conn.cursor() as cur:
                # Build dynamic INSERT statement
                columns = [col.lower() for col in data.columns]
                columns_str = ', '.join(columns)
                placeholders = ', '.join(['%s'] * len(columns))
                insert_query = f"INSERT INTO {self.TABLE_NAME} ({columns_str}) VALUES ({placeholders})"
                
                logging.info(f"Executing INSERT query with {len(data_tuples)} rows")
                logging.info(f"Columns: {columns}")
                
                # Use execute_batch for better performance
                execute_batch(cur, insert_query, data_tuples, page_size=1000)
                rows_inserted = cur.rowcount
                
                conn.commit()
                logging.info(f"✅ Successfully inserted {rows_inserted} rows into '{self.TABLE_NAME}'")
                
                return rows_inserted
                
        except Exception as e:
            conn.rollback()
            logging.error(f"❌ Insertion failed: {e}")
            raise CustomException(e, sys) from e
    
    def load_and_insert(self, csv_file_path: str) -> int:
        """
        Complete workflow: Load CSV, prepare data, and insert into database.
        Uses context manager to ensure proper connection handling.
        
        Args:
            csv_file_path: Path to cleaned CSV file
            
        Returns:
            Number of rows inserted
            
        Raises:
            CustomException: If any step fails
        """
        try:
            logging.info("="*60)
            logging.info("Starting Database Insertion Process")
            logging.info("="*60)
            
            # Load data
            logging.info(f"Loading data from: {csv_file_path}")
            data = pd.read_csv(csv_file_path)
            logging.info(f"✅ Loaded {data.shape[0]} rows, {data.shape[1]} columns")
            
            # Prepare data first (before opening connection)
            data, data_tuples = self.prepare_data(data)
            
            # Use context manager for database operations
            with self.get_connection() as conn:
                # Ensure table exists
                self.ensure_table_exists(conn)
                
                # Insert data
                rows_inserted = self.insert_data(conn, data, data_tuples)
            
            logging.info("="*60)
            logging.info("✅ Database insertion completed successfully!")
            logging.info(f"   Total rows inserted: {rows_inserted}")
            logging.info("="*60)
            
            return rows_inserted
            
        except Exception as e:
            logging.error(f"❌ Error during database insertion: {e}")
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    try:
        logging.info("🚀 Starting database insertion script")
        
        inserter = DatabaseInsertion()
        
        # Define path to cleaned data CSV
        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
        CLEANED_CSV = PROJECT_ROOT / "datasets" / "cleaned_ingested_data" / "cleaned_ingested_data.csv"
        
        logging.info(f"Looking for CSV at: {CLEANED_CSV}")
        
        if not CLEANED_CSV.exists():
            raise FileNotFoundError(f"CSV file not found: {CLEANED_CSV}")
        
        rows = inserter.load_and_insert(str(CLEANED_CSV))
        print(f"\n✅ Successfully inserted {rows} rows into the database!")
        
    except FileNotFoundError as e:
        logging.error(f"❌ File error: {e}")
        print(f"❌ File error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)