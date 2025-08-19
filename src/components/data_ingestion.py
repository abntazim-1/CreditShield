import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException, DataLoadingError, DataValidationError
from src.logger import get_logger
from src.components.data_transformation import DataTransformation
from src.utils import save_object, evaluate_models
from src.components.model_trainer import ModelTrainer

# Initialize logger
logger = get_logger(__name__)


@dataclass
class DataIngestionConfig:
    """Configuration class for data ingestion parameters."""
    
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    source_data_path: str = os.path.join('notebooks', 'data', 'credit_risk_dataset.csv')
    test_size: float = 0.2
    random_state: int = 42
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not (0 < self.test_size < 1):
            raise DataValidationError("test_size must be between 0 and 1")
        
        if not os.path.exists(self.source_data_path):
            raise DataLoadingError(f"Source data file not found: {self.source_data_path}")


class DataIngestion:
    """
    Data Ingestion class responsible for loading, splitting, and saving data.
    
    This class handles the initial data loading process, performs train-test split,
    and saves the processed data to specified locations.
    """
    
    def __init__(self, config: Optional[DataIngestionConfig] = None):
        """
        Initialize DataIngestion with configuration.
        
        Args:
            config: DataIngestionConfig object. If None, uses default config.
        """
        self.ingestion_config = config or DataIngestionConfig()
        logger.info("DataIngestion initialized with config")
    
    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Validate the loaded dataframe.
        
        Args:
            df: Pandas dataframe to validate
            
        Raises:
            DataValidationError: If dataframe is invalid
        """
        if df.empty:
            raise DataValidationError("Loaded dataframe is empty")
        
        if df.isnull().all().any():
            raise DataValidationError("Dataframe contains columns with all null values")
        
        logger.info(f"Dataframe validation passed. Shape: {df.shape}")
    
    def _create_directories(self) -> None:
        """Create necessary directories for saving data."""
        directories = [
            os.path.dirname(self.ingestion_config.train_data_path),
            os.path.dirname(self.ingestion_config.test_data_path),
            os.path.dirname(self.ingestion_config.raw_data_path)
        ]
        
        for directory in set(directories):  # Use set to avoid duplicates
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            except OSError as e:
                raise DataLoadingError(
                    f"Failed to create directory: {directory}",
                    error_detail=e,
                    context={"directory": directory}
                )
    
    def _load_data(self) -> pd.DataFrame:
        """
        Load data from the source file.
        
        Returns:
            Loaded pandas dataframe
            
        Raises:
            DataLoadingError: If data loading fails
        """
        try:
            logger.info(f"Loading data from: {self.ingestion_config.source_data_path}")
            df = pd.read_csv(self.ingestion_config.source_data_path)
            logger.info(f"Successfully loaded data. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            raise DataLoadingError(
                f"Data file not found: {self.ingestion_config.source_data_path}",
                context={"file_path": self.ingestion_config.source_data_path}
            )
        except pd.errors.EmptyDataError:
            raise DataLoadingError("Data file is empty")
        except pd.errors.ParserError as e:
            raise DataLoadingError(
                "Failed to parse CSV file",
                error_detail=e,
                context={"file_path": self.ingestion_config.source_data_path}
            )
        except Exception as e:
            raise DataLoadingError(
                "Unexpected error during data loading",
                error_detail=e,
                context={"file_path": self.ingestion_config.source_data_path}
            )
    
    def _save_data_splits(
        self, 
        df: pd.DataFrame, 
        train_set: pd.DataFrame, 
        test_set: pd.DataFrame
    ) -> None:
        """
        Save raw data and train-test splits.
        
        Args:
            df: Original dataframe
            train_set: Training data
            test_set: Testing data
        """
        try:
            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logger.info(f"Raw data saved to: {self.ingestion_config.raw_data_path}")
            
            # Save train set
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            logger.info(f"Training data saved to: {self.ingestion_config.train_data_path}")
            
            # Save test set
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logger.info(f"Testing data saved to: {self.ingestion_config.test_data_path}")
            
        except Exception as e:
            raise DataLoadingError(
                "Failed to save data splits",
                error_detail=e
            )
    
    def initiate_data_ingestion(self) -> Tuple[str, str]:
        """
        Main method to initiate data ingestion process.
        
        Returns:
            Tuple containing paths to train and test data files
            
        Raises:
            CustomException: If any step in the ingestion process fails
        """
        logger.info("Starting data ingestion process")
        
        try:
            # Create necessary directories
            self._create_directories()
            
            # Load data
            df = self._load_data()
            
            # Validate data
            self._validate_dataframe(df)
            
            # Perform train-test split
            logger.info("Initiating train-test split")
            train_set, test_set = train_test_split(
                df,
                test_size=self.ingestion_config.test_size,
                random_state=self.ingestion_config.random_state
            )
            
            logger.info(f"Train set shape: {train_set.shape}, Test set shape: {test_set.shape}")
            
            # Save data splits
            self._save_data_splits(df, train_set, test_set)
            
            logger.info("Data ingestion completed successfully")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except (DataLoadingError, DataValidationError):
            # Re-raise custom exceptions
            raise
        except Exception as e:
            raise CustomException(
                "Unexpected error during data ingestion",
                error_detail=e,
                context={"config": self.ingestion_config.__dict__}
            )



if __name__ == "__main__":
    
    obj=DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion() 
    data_transformation = DataTransformation()
    train_arr, test_arr, _, transformed_columns = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr, transformed_columns)) 