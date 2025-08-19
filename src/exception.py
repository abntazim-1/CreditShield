"""
Custom exceptions for the credit risk project.

This module defines custom exception classes to handle specific error scenarios
in the credit risk analysis application.
"""

import sys
from typing import Optional, Dict, Any
from src.logger import get_logger

logger = get_logger(__name__)


class CustomException(Exception):
    """
    Base custom exception class for the credit risk project.
    
    This class provides enhanced error handling with automatic logging,
    detailed error messages, and context information.
    """
    
    def __init__(
        self,
        error_message: str,
        error_detail: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize custom exception.
        
        Args:
            error_message: Human-readable error message
            error_detail: Original exception that caused this error
            context: Additional context information
        """
        super().__init__(error_message)
        self.error_message = error_message
        self.error_detail = error_detail
        self.context = context or {}
        
        # Get detailed error information
        self.error_info = self._get_error_details()
        
        # Log the exception
        logger.error(f"{self.__class__.__name__}: {self.error_message}")
        if self.error_detail:
            logger.error(f"Original error: {str(self.error_detail)}")
        if self.context:
            logger.error(f"Context: {self.context}")
    
    def _get_error_details(self) -> str:
        """
        Extract detailed error information including file, line number, and traceback.
        
        Returns:
            Formatted error details string
        """
        try:
            _, _, tb = sys.exc_info()
            if tb is not None:
                filename = tb.tb_frame.f_code.co_filename
                line_number = tb.tb_lineno
                function_name = tb.tb_frame.f_code.co_name
                
                return (
                    f"Error occurred in file [{filename}] "
                    f"at line number [{line_number}] "
                    f"in function [{function_name}]"
                )
        except Exception:
            pass
        
        return "Error details unavailable"
    
    def __str__(self) -> str:
        """Return formatted error message."""
        return f"{self.error_message}\nDetails: {self.error_info}"


class DataValidationError(CustomException):
    """
    Exception raised when data validation fails.
    
    This exception is used when input data doesn't meet expected criteria,
    such as missing required columns, invalid data types, or out-of-range values.
    """
    
    def __init__(
        self,
        error_message: str = "Data validation failed",
        error_detail: Optional[Exception] = None,
        validation_errors: Optional[Dict[str, str]] = None
    ):
        context = {"validation_errors": validation_errors} if validation_errors else None
        super().__init__(error_message, error_detail, context)


class ModelTrainingError(CustomException):
    """
    Exception raised when model training fails.
    
    This exception is used when machine learning model training encounters
    issues such as convergence problems, insufficient data, or invalid parameters.
    """
    
    def __init__(
        self,
        error_message: str = "Model training failed",
        error_detail: Optional[Exception] = None,
        model_params: Optional[Dict[str, Any]] = None
    ):
        context = {"model_parameters": model_params} if model_params else None
        super().__init__(error_message, error_detail, context)


class DataLoadingError(CustomException):
    """
    Exception raised when data loading fails.
    
    This exception is used when there are issues loading data from files,
    databases, or external APIs.
    """
    
    def __init__(
        self,
        error_message: str = "Data loading failed",
        error_detail: Optional[Exception] = None,
        file_path: Optional[str] = None
    ):
        context = {"file_path": file_path} if file_path else None
        super().__init__(error_message, error_detail, context)


class ModelPredictionError(CustomException):
    """
    Exception raised when model prediction fails.
    
    This exception is used when trained models fail to make predictions
    due to issues like incompatible input data or model loading problems.
    """
    
    def __init__(
        self,
        error_message: str = "Model prediction failed",
        error_detail: Optional[Exception] = None,
        input_shape: Optional[tuple] = None
    ):
        context = {"input_shape": input_shape} if input_shape else None
        super().__init__(error_message, error_detail, context)


class ConfigurationError(CustomException):
    """
    Exception raised when configuration is invalid.
    
    This exception is used when configuration files are missing,
    corrupted, or contain invalid values.
    """
    
    def __init__(
        self,
        error_message: str = "Configuration error",
        error_detail: Optional[Exception] = None,
        config_key: Optional[str] = None
    ):
        context = {"config_key": config_key} if config_key else None
        super().__init__(error_message, error_detail, context)


class FeatureEngineeringError(CustomException):
    """
    Exception raised when feature engineering fails.
    
    This exception is used when feature transformation, scaling,
    or selection processes encounter issues.
    """
    
    def __init__(
        self,
        error_message: str = "Feature engineering failed",
        error_detail: Optional[Exception] = None,
        feature_names: Optional[list] = None
    ):
        context = {"feature_names": feature_names} if feature_names else None
        super().__init__(error_message, error_detail, context)


# Exception handler decorator
def handle_exceptions(exception_type: type = CustomException):
    """
    Decorator to automatically handle exceptions in functions.
    
    Args:
        exception_type: Type of custom exception to raise
    
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if isinstance(e, CustomException):
                    # Re-raise custom exceptions
                    raise
                else:
                    # Wrap other exceptions in custom exception
                    raise exception_type(
                        error_message=f"Error in function '{func.__name__}': {str(e)}",
                        error_detail=e
                    )
        return wrapper
    return decorator


"""
Custom exceptions for the credit risk project.

This module defines custom exception classes to handle specific error scenarios
in the credit risk analysis application.
"""

import sys
from typing import Optional, Dict, Any
from src.logger import get_logger

logger = get_logger(__name__)


class CustomException(Exception):
    """
    Base custom exception class for the credit risk project.
    
    This class provides enhanced error handling with automatic logging,
    detailed error messages, and context information.
    """
    
    def __init__(
        self,
        error_message: str,
        error_detail: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize custom exception.
        
        Args:
            error_message: Human-readable error message
            error_detail: Original exception that caused this error
            context: Additional context information
        """
        super().__init__(error_message)
        self.error_message = error_message
        self.error_detail = error_detail
        self.context = context or {}
        
        # Get detailed error information
        self.error_info = self._get_error_details()
        
        # Log the exception
        logger.error(f"{self.__class__.__name__}: {self.error_message}")
        if self.error_detail:
            logger.error(f"Original error: {str(self.error_detail)}")
        if self.context:
            logger.error(f"Context: {self.context}")
    
    def _get_error_details(self) -> str:
        """
        Extract detailed error information including file, line number, and traceback.
        
        Returns:
            Formatted error details string
        """
        try:
            _, _, tb = sys.exc_info()
            if tb is not None:
                filename = tb.tb_frame.f_code.co_filename
                line_number = tb.tb_lineno
                function_name = tb.tb_frame.f_code.co_name
                
                return (
                    f"Error occurred in file [{filename}] "
                    f"at line number [{line_number}] "
                    f"in function [{function_name}]"
                )
        except Exception:
            pass
        
        return "Error details unavailable"
    
    def __str__(self) -> str:
        """Return formatted error message."""
        return f"{self.error_message}\nDetails: {self.error_info}"


class DataValidationError(CustomException):
    """
    Exception raised when data validation fails.
    
    This exception is used when input data doesn't meet expected criteria,
    such as missing required columns, invalid data types, or out-of-range values.
    """
    
    def __init__(
        self,
        error_message: str = "Data validation failed",
        error_detail: Optional[Exception] = None,
        validation_errors: Optional[Dict[str, str]] = None
    ):
        context = {"validation_errors": validation_errors} if validation_errors else None
        super().__init__(error_message, error_detail, context)


class ModelTrainingError(CustomException):
    """
    Exception raised when model training fails.
    
    This exception is used when machine learning model training encounters
    issues such as convergence problems, insufficient data, or invalid parameters.
    """
    
    def __init__(
        self,
        error_message: str = "Model training failed",
        error_detail: Optional[Exception] = None,
        model_params: Optional[Dict[str, Any]] = None
    ):
        context = {"model_parameters": model_params} if model_params else None
        super().__init__(error_message, error_detail, context)


class DataLoadingError(CustomException):
    """
    Exception raised when data loading fails.
    
    This exception is used when there are issues loading data from files,
    databases, or external APIs.
    """
    
    def __init__(
        self,
        error_message: str = "Data loading failed",
        error_detail: Optional[Exception] = None,
        file_path: Optional[str] = None
    ):
        context = {"file_path": file_path} if file_path else None
        super().__init__(error_message, error_detail, context)


class ModelPredictionError(CustomException):
    """
    Exception raised when model prediction fails.
    
    This exception is used when trained models fail to make predictions
    due to issues like incompatible input data or model loading problems.
    """
    
    def __init__(
        self,
        error_message: str = "Model prediction failed",
        error_detail: Optional[Exception] = None,
        input_shape: Optional[tuple] = None
    ):
        context = {"input_shape": input_shape} if input_shape else None
        super().__init__(error_message, error_detail, context)


class ConfigurationError(CustomException):
    """
    Exception raised when configuration is invalid.
    
    This exception is used when configuration files are missing,
    corrupted, or contain invalid values.
    """
    
    def __init__(
        self,
        error_message: str = "Configuration error",
        error_detail: Optional[Exception] = None,
        config_key: Optional[str] = None
    ):
        context = {"config_key": config_key} if config_key else None
        super().__init__(error_message, error_detail, context)


class FeatureEngineeringError(CustomException):
    """
    Exception raised when feature engineering fails.
    
    This exception is used when feature transformation, scaling,
    or selection processes encounter issues.
    """
    
    def __init__(
        self,
        error_message: str = "Feature engineering failed",
        error_detail: Optional[Exception] = None,
        feature_names: Optional[list] = None
    ):
        context = {"feature_names": feature_names} if feature_names else None
        super().__init__(error_message, error_detail, context)


# Exception handler decorator
def handle_exceptions(exception_type: type = CustomException):
    """
    Decorator to automatically handle exceptions in functions.
    
    Args:
        exception_type: Type of custom exception to raise
    
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if isinstance(e, CustomException):
                    # Re-raise custom exceptions
                    raise
                else:
                    # Wrap other exceptions in custom exception
                    raise exception_type(
                        error_message=f"Error in function '{func.__name__}': {str(e)}",
                        error_detail=e
                    )
        return wrapper
    return decorator


if __name__ == "__main__":
    logger.info("Exception module loaded successfully")