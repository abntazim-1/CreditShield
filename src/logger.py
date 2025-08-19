import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = __name__,
    log_level: int = logging.INFO,
    log_dir: Optional[str] = None,
    log_filename: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Logger name (usually __name__)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files (default: ./logs)
        log_filename: Custom log filename (default: timestamped)
        console_output: Whether to output to console
        file_output: Whether to output to file
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_output:
        try:
            # Set up log directory
            if log_dir is None:
                log_dir = os.path.join(os.getcwd(), "logs")
            
            # Create directory if it doesn't exist
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            
            # Generate log filename
            if log_filename is None:
                timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                log_filename = f"app_{timestamp}.log"
            
            # Full path to log file
            log_file_path = os.path.join(log_dir, log_filename)
            
            # Create file handler
            file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            logger.info(f"Logging to file: {log_file_path}")
            
        except (OSError, PermissionError) as e:
            logger.error(f"Failed to create log file: {e}")
    
    return logger


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get a logger instance. If not already configured, set up with defaults.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


# Create a default logger for the module
logger = setup_logger(__name__)


class LoggerMixin:
    """Mixin class to add logging capability to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for the class."""
        return get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")


# Environment-specific configuration
def configure_production_logging():
    """Configure logging for production environment."""
    return setup_logger(
        name="production",
        log_level=logging.WARNING,
        log_dir="/var/log/credi_risk",
        console_output=False,
        file_output=True,
    )


def configure_development_logging():
    """Configure logging for development environment."""
    return setup_logger(
        name="development",
        log_level=logging.DEBUG,
        console_output=True,
        file_output=True,
    )


if __name__ == "__main__":
    logger.info("Logging has started")