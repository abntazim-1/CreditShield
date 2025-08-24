import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin

from src.exception import CustomException, DataValidationError, FeatureEngineeringError
from src.logger import get_logger
from src.utils import save_object
from src.components.imbalance_handler import ImbalanceHandler, ImbalanceConfig

# Initialize logger
logger = get_logger(__name__)


@dataclass
class DataTransformationConfig:
    """Configuration class for data transformation parameters."""
    
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

    
    # Imputation settings
    numeric_imputation_strategy: str = "mean"  # mean, median, constant
    categorical_imputation_strategy: str = "most_frequent"  # most_frequent, constant
    
    # Encoding settings
    drop_first_dummy: bool = True
    handle_unknown_categories: str = "ignore"  # ignore, error
    
    # Scaling settings
    scaling_method: str = "standard"  # standard, minmax, robust
    
    # K-means binning settings
    kmeans_n_bins: int = 4
    kmeans_random_state: int = 42
    enable_kmeans_binning: bool = False
    
    # Data validation settings
    min_samples: int = 100
    max_missing_ratio: float = 0.5
    
    # Imbalance handling settings
    handle_imbalance: bool = True
    imbalance_technique: str = "SMOTE"  # SMOTE, ADASYN, RandomOverSampler, etc.
    imbalance_strategy: str = "auto"  # auto, oversample, undersample, combine, none


class MeanImputer(BaseEstimator, TransformerMixin):
    """Custom transformer for mean imputation with verbose logging."""
    
    def __init__(self, columns: List[str], verbose: bool = False):
        self.columns = columns
        self.verbose = verbose
        self.mean_values_ = {}
    
    def fit(self, X, y=None):
        """Fit the imputer by calculating means."""
        if isinstance(X, pd.DataFrame):
            for col in self.columns:
                if col in X.columns and pd.api.types.is_numeric_dtype(X[col]):
                    self.mean_values_[col] = X[col].mean()
                    if self.verbose:
                        missing_count = X[col].isna().sum()
                        logger.info(f"Column '{col}': {missing_count} missing values, mean = {self.mean_values_[col]:.4f}")
        return self
    
    def transform(self, X):
        """Transform the data by imputing missing values."""
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            for col in self.columns:
                if col in X_transformed.columns and col in self.mean_values_:
                    missing_before = X_transformed[col].isna().sum()
                    X_transformed[col] = X_transformed[col].fillna(self.mean_values_[col])
                    if self.verbose and missing_before > 0:
                        logger.info(f"Imputed {missing_before} missing values in '{col}' with mean = {self.mean_values_[col]:.4f}")
            return X_transformed
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Custom transformer for categorical encoding with verbose logging."""
    
    def __init__(self, drop_first: bool = True, verbose: bool = False):
        self.drop_first = drop_first
        self.verbose = verbose
        self.categorical_cols_ = []
        self.encoder_ = None
    
    def fit(self, X, y=None):
        """Fit the encoder by identifying categorical columns."""
        if isinstance(X, pd.DataFrame):
            self.categorical_cols_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
            if self.categorical_cols_:
                self.encoder_ = OneHotEncoder(drop='first' if self.drop_first else None, 
                                            handle_unknown='ignore', sparse_output=False)
                self.encoder_.fit(X[self.categorical_cols_])
                if self.verbose:
                    logger.info(f"Encoding {len(self.categorical_cols_)} categorical columns: {self.categorical_cols_}")
        return self
    
    def transform(self, X):
        """Transform the data by encoding categorical variables."""
        if isinstance(X, pd.DataFrame) and self.categorical_cols_ and self.encoder_:
            X_transformed = X.copy()
            
            # Get encoded features
            encoded_features = self.encoder_.transform(X[self.categorical_cols_])
            encoded_feature_names = self.encoder_.get_feature_names_out(self.categorical_cols_)
            
            # Create DataFrame with encoded features
            encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=X.index)
            
            # Drop original categorical columns and add encoded ones
            X_transformed = X_transformed.drop(columns=self.categorical_cols_)
            X_transformed = pd.concat([X_transformed, encoded_df], axis=1)
            
            # Convert boolean columns to integers
            bool_cols = X_transformed.select_dtypes(include='bool').columns
            if len(bool_cols) > 0:
                X_transformed[bool_cols] = X_transformed[bool_cols].astype(int)
                if self.verbose:
                    logger.info(f"Converted {len(bool_cols)} boolean columns to integers (0/1)")
            
            return X_transformed
        return X


class KMeansBinner(BaseEstimator, TransformerMixin):
    """Custom transformer for K-means based binning."""
    
    def __init__(self, columns: List[str], n_bins: int = 4, random_state: int = 42, 
                 replace: bool = False, verbose: bool = False):
        self.columns = columns
        self.n_bins = n_bins
        self.random_state = random_state
        self.replace = replace
        self.verbose = verbose
        self.kmeans_models_ = {}
    
    def fit(self, X, y=None):
        """Fit K-means models for specified columns."""
        if isinstance(X, pd.DataFrame):
            for col in self.columns:
                if col in X.columns and pd.api.types.is_numeric_dtype(X[col]):
                    values = X[col].dropna().values.reshape(-1, 1)
                    if len(values) > 0:
                        kmeans = KMeans(n_clusters=self.n_bins, random_state=self.random_state, n_init=10)
                        kmeans.fit(values)
                        self.kmeans_models_[col] = kmeans
                        if self.verbose:
                            logger.info(f"Fitted K-means for column '{col}' with {self.n_bins} clusters")
        return self
    
    def transform(self, X):
        """Transform the data by applying K-means binning."""
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            
            for col in self.columns:
                if col in self.kmeans_models_ and col in X_transformed.columns:
                    # Create bin labels
                    bin_labels = pd.Series(np.nan, index=X_transformed.index, dtype="Int64")
                    non_null_mask = X_transformed[col].notna()
                    
                    if non_null_mask.any():
                        values = X_transformed.loc[non_null_mask, col].values.reshape(-1, 1)
                        bin_labels.loc[non_null_mask] = self.kmeans_models_[col].predict(values)
                    
                    if self.replace:
                        X_transformed[col] = bin_labels
                    else:
                        X_transformed[f"{col}_bin"] = bin_labels
                    
                    if self.verbose:
                        logger.info(f"Applied K-means binning to column '{col}'")
            
            return X_transformed
        return X


class DataTransformation:
    """
    Data transformation class for preprocessing data for machine learning.
    
    This class handles missing value imputation, categorical encoding, feature scaling,
    and optional K-means binning for continuous variables.
    """
    
    def __init__(self, config: Optional[DataTransformationConfig] = None):
        """
        Initialize DataTransformation with configuration.
        
        Args:
            config: DataTransformationConfig object. If None, uses default config.
        """
        self.transformation_config = config or DataTransformationConfig()
        logger.info("DataTransformation initialized")
    
    def _validate_data(self, df: pd.DataFrame, data_type: str = "training") -> None:
        """
        Validate input data quality.
        
        Args:
            df: DataFrame to validate
            data_type: Type of data (training/testing) for logging
            
        Raises:
            DataValidationError: If data doesn't meet quality standards
        """
        if df.empty:
            raise DataValidationError(f"{data_type} data is empty")
        
        if len(df) < self.transformation_config.min_samples:
            raise DataValidationError(
                f"{data_type} data has insufficient samples: {len(df)} < {self.transformation_config.min_samples}"
            )
        
        # Check for excessive missing values
        missing_ratios = df.isnull().mean()
        problematic_cols = missing_ratios[missing_ratios > self.transformation_config.max_missing_ratio]
        
        if not problematic_cols.empty:
            logger.warning(f"Columns with high missing ratio in {data_type} data: {dict(problematic_cols)}")
        
        logger.info(f"{data_type} data validation passed. Shape: {df.shape}")
    
    def _remove_duplicates(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Remove duplicate rows from DataFrame.
        
        Args:
            df: Input DataFrame
            verbose: Whether to log duplicate information
            
        Returns:
            DataFrame with duplicates removed
        """
        initial_count = len(df)
        duplicate_count = df.duplicated().sum()
        
        if duplicate_count > 0:
            df_cleaned = df.drop_duplicates()
            if verbose:
                logger.info(f"Removed {duplicate_count} duplicate rows. Shape: {initial_count} → {len(df_cleaned)}")
            return df_cleaned
        else:
            if verbose:
                logger.info("No duplicate rows found")
            return df
    
    def _get_transformer_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """
        Create preprocessing pipeline based on data characteristics.
        
        Args:
            X: Training data
            
        Returns:
            Configured preprocessing pipeline
        """
        # Identify column types
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Identified {len(numeric_features)} numeric and {len(categorical_features)} categorical features")
        
        # Create preprocessing steps
        steps = []
        
        # Add mean imputation for numeric columns
        if numeric_features:
            steps.append(('mean_imputer', MeanImputer(numeric_features, verbose=True)))
        
        # Add categorical encoding
        if categorical_features:
            steps.append(('categorical_encoder', CategoricalEncoder(
                drop_first=self.transformation_config.drop_first_dummy, 
                verbose=True
            )))
        
        # Add K-means binning if enabled
        if self.transformation_config.enable_kmeans_binning and numeric_features:
            steps.append(('kmeans_binner', KMeansBinner(
                columns=numeric_features,
                n_bins=self.transformation_config.kmeans_n_bins,
                random_state=self.transformation_config.kmeans_random_state,
                replace=False,
                verbose=True
            )))
        
        # Add scaling (will be applied to all numeric features after transformations)
        if self.transformation_config.scaling_method == "standard":
            scaler = StandardScaler()
        elif self.transformation_config.scaling_method == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif self.transformation_config.scaling_method == "robust":
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        steps.append(('scaler', scaler))
        
        return Pipeline(steps)
    
    def initiate_data_transformation(self, train_path: str, test_path: str) -> Tuple[np.ndarray, np.ndarray, str, List[str]]:
        """
        Initiate the data transformation process.
        
        Args:
            train_path: Path to training data CSV file
            test_path: Path to test data CSV file
            
        Returns:
            Tuple containing:
            - Transformed training data array
            - Transformed test data array  
            - Path to saved preprocessor object
            
        Raises:
            CustomException: If transformation process fails
        """
        try:
            logger.info("Starting data transformation")
            
            # Load data
            logger.info("Loading training and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logger.info(f"Training data shape: {train_df.shape}")
            logger.info(f"Test data shape: {test_df.shape}")
            
            # Validate data
            self._validate_data(train_df, "training")
            self._validate_data(test_df, "test")
            
            # Remove duplicates
            train_df = self._remove_duplicates(train_df)
            test_df = self._remove_duplicates(test_df)
            
            # Assuming target column is the last column
            target_column_name = "loan_status"
            logger.info(f"Target column identified: {target_column_name}")
            
            # Separate features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logger.info("Creating preprocessing pipeline")
            
            # Create and fit preprocessing pipeline
            preprocessing_pipeline = self._get_transformer_pipeline(input_feature_train_df)
            
            logger.info("Fitting preprocessor on training data")
            preprocessing_pipeline.fit(input_feature_train_df)
            
            # Extract feature names before scaling and then scale
            logger.info("Extracting feature names before scaling")
            # Build a pipeline of all steps except the final scaler to preserve DataFrame columns
            pre_scaler_pipeline = Pipeline(preprocessing_pipeline.steps[:-1])
            
            logger.info("Transforming training data (pre-scaler)")
            features_train_df = pre_scaler_pipeline.transform(input_feature_train_df)
            logger.info("Transforming test data (pre-scaler)")
            features_test_df = pre_scaler_pipeline.transform(input_feature_test_df)

            # Capture transformed feature names
            if isinstance(features_train_df, pd.DataFrame):
                transformed_feature_names: List[str] = features_train_df.columns.tolist()
            else:
                # Fallback: if a numpy array is returned unexpectedly
                transformed_feature_names = [f"feature_{i}" for i in range(features_train_df.shape[1])]  # type: ignore

            # Scale features using the already-fitted scaler in the full pipeline
            logger.info("Applying scaler to transformed features")
            scaler = preprocessing_pipeline.named_steps.get('scaler')
            if scaler is None:
                raise FeatureEngineeringError("Scaler step not found in preprocessing pipeline")
            input_feature_train_arr = scaler.transform(features_train_df)
            input_feature_test_arr = scaler.transform(features_test_df)
            
            # Handle class imbalance if enabled
            if self.transformation_config.handle_imbalance:
                logger.info("Checking for class imbalance")
                
                imbalance_config = ImbalanceConfig(
                    technique=self.transformation_config.imbalance_technique,
                    resampling_strategy=self.transformation_config.imbalance_strategy
                )
                
                imbalance_handler = ImbalanceHandler(imbalance_config)
                
                # Apply resampling to training data only
                input_feature_train_arr, target_feature_train_resampled, imbalance_info = imbalance_handler.apply_resampling(
                    input_feature_train_arr, target_feature_train_df
                )
                
                if imbalance_info["resampling_applied"]:
                    logger.info(f"Resampling applied: {imbalance_info['technique']}")
                    logger.info(f"Training data shape changed: {imbalance_info['original_shape']} → {imbalance_info['new_shape']}")
                    target_feature_train_df = target_feature_train_resampled
            
            # Combine features and target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logger.info(f"Final training array shape: {train_arr.shape}")
            logger.info(f"Final test array shape: {test_arr.shape}")
            
            # Save preprocessing object
            logger.info("Saving preprocessing object")
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_pipeline
            )
            
            logger.info("Data transformation completed successfully")
            
            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path,
                transformed_feature_names + [target_column_name]
            )
            
        except Exception as e:
            raise FeatureEngineeringError(
                "Error occurred during data transformation",
                error_detail=e,
                context={
                    "train_path": train_path,
                    "test_path": test_path,
                    "config": self.transformation_config.__dict__
                }
            )


# if __name__ == "__main__":
#     config = DataTransformationConfig()
#     data_transformation = DataTransformation(config)
#     train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
#         config.train_data_path, config.test_data_path
#     )

#     print(f"Preprocessor created at: {preprocessor_path}")