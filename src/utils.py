"""
Utility functions for the credit risk project.

This module contains common utility functions for model evaluation, 
object serialization, file operations, and other shared functionality.
"""

import os
import sys
import pickle
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib

from src.exception import CustomException, DataLoadingError, ModelTrainingError
from src.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


def save_object(file_path: str, obj: Any) -> None:
    """
    Save a Python object to a pickle file.
    
    Args:
        file_path: Path where the object should be saved
        obj: Object to be saved
        
    Raises:
        CustomException: If saving fails
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        logger.info(f"Object saved successfully to {file_path}")
        
    except Exception as e:
        raise CustomException(
            f"Error occurred while saving object to {file_path}",
            error_detail=e,
            context={"file_path": file_path, "object_type": type(obj).__name__}
        )


def load_object(file_path: str) -> Any:
    """
    Load a Python object from a pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Loaded object
        
    Raises:
        DataLoadingError: If loading fails
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        
        logger.info(f"Object loaded successfully from {file_path}")
        return obj
        
    except Exception as e:
        raise DataLoadingError(
            f"Error occurred while loading object from {file_path}",
            error_detail=e,
            context={"file_path": file_path}
        )


def save_json(data: Dict[str, Any], file_path: str, indent: int = 4) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save the JSON file
        indent: JSON indentation level
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        
        logger.info(f"JSON data saved to {file_path}")
        
    except Exception as e:
        raise CustomException(
            f"Error saving JSON to {file_path}",
            error_detail=e
        )


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"JSON data loaded from {file_path}")
        return data
        
    except Exception as e:
        raise DataLoadingError(
            f"Error loading JSON from {file_path}",
            error_detail=e
        )


def save_yaml(data: Dict[str, Any], file_path: str) -> None:
    """
    Save dictionary to YAML file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save the YAML file
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"YAML data saved to {file_path}")
        
    except Exception as e:
        raise CustomException(
            f"Error saving YAML to {file_path}",
            error_detail=e
        )


def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    Load dictionary from YAML file.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Loaded dictionary
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        logger.info(f"YAML data loaded from {file_path}")
        return data
        
    except Exception as e:
        raise DataLoadingError(
            f"Error loading YAML from {file_path}",
            error_detail=e
        )


def ensure_dir(file_path: str) -> None:
    """
    Ensure that the directory for a file path exists.
    
    Args:
        file_path: Path to file (directory will be created)
    """
    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)


def get_file_size(file_path: str) -> str:
    """
    Get human-readable file size.
    
    Args:
        file_path: Path to file
        
    Returns:
        Human-readable file size string
    """
    try:
        size = os.path.getsize(file_path)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    except Exception:
        return "Unknown"


def evaluate_classification_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Evaluate classification model performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for AUC calculation)
        average: Averaging method for multi-class metrics
        
    Returns:
        Dictionary containing evaluation metrics
    """
    try:
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
        }
        
        # Add AUC if probabilities are provided
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:  # Multi-class
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average=average)
            except Exception as e:
                logger.warning(f"Could not calculate AUC: {e}")
                metrics['roc_auc'] = None
        
        logger.info("Classification model evaluation completed")
        return metrics
        
    except Exception as e:
        raise ModelTrainingError(
            "Error during classification model evaluation",
            error_detail=e
        )


def evaluate_regression_model(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate regression model performance.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary containing evaluation metrics
    """
    try:
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2_score': r2_score(y_true, y_pred)
        }
        
        # Add additional metrics
        metrics['mean_absolute_percentage_error'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        logger.info("Regression model evaluation completed")
        return metrics
        
    except Exception as e:
        raise ModelTrainingError(
            "Error during regression model evaluation",
            error_detail=e
        )


def evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: Dict[str, Any],
    param_grids: Optional[Dict[str, Dict]] = None,
    cv_folds: int = 5,
    scoring: str = 'accuracy',
    n_jobs: int = -1
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate multiple models with optional hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        models: Dictionary of model names and model objects
        param_grids: Dictionary of parameter grids for each model
        cv_folds: Number of cross-validation folds
        scoring: Scoring metric for evaluation
        n_jobs: Number of parallel jobs
        
    Returns:
        Dictionary containing model evaluation results
    """
    try:
        model_reports = {}
        
        n_train, n_features = X_train.shape
        n_test = X_test.shape[0]

        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")
            
            # Hyperparameter tuning if parameter grid is provided
            if param_grids and model_name in param_grids:
                logger.info(f"Performing hyperparameter tuning for {model_name}")
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grids[model_name],
                    cv=cv_folds,
                    scoring=scoring,
                    n_jobs=n_jobs,
                    verbose=0
                )
                fit_start = time.time()
                grid_search.fit(X_train, y_train)
                fit_time_sec = time.time() - fit_start
                best_model = grid_search.best_estimator_
                logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            else:
                best_model = model
                fit_start = time.time()
                best_model.fit(X_train, y_train)
                fit_time_sec = time.time() - fit_start
            
            # Cross-validation score
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_folds, scoring=scoring, n_jobs=n_jobs)
            
            # Test predictions
            pred_start = time.time()
            y_pred = best_model.predict(X_test)
            predict_time_sec = time.time() - pred_start
            
            # Get prediction probabilities if available (for classification)
            y_pred_proba = None
            if hasattr(best_model, 'predict_proba'):
                try:
                    y_pred_proba = best_model.predict_proba(X_test)
                except Exception:
                    pass
            
            # Determine if this is a classification or regression problem
            is_classification = len(np.unique(y_train)) < 20  # Simple heuristic
            
            if is_classification:
                metrics = evaluate_classification_model(y_test, y_pred, y_pred_proba)
            else:
                metrics = evaluate_regression_model(y_test, y_pred)
            
            # Add cross-validation metrics
            metrics.update({
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist(),
                'fit_time_sec': float(fit_time_sec),
                'predict_time_sec': float(predict_time_sec),
                'n_train_samples': int(n_train),
                'n_test_samples': int(n_test),
                'n_features': int(n_features)
            })
            
            model_reports[model_name] = metrics
            
            logger.info(f"Completed evaluation for {model_name}")
        
        logger.info("Model evaluation completed for all models")
        return model_reports
        
    except Exception as e:
        raise ModelTrainingError(
            "Error during model evaluation",
            error_detail=e,
            context={"models": list(models.keys())}
        )


def get_best_model(
    model_reports: Dict[str, Dict[str, float]],
    metric: str = 'accuracy',
    higher_is_better: bool = True
) -> Tuple[str, Dict[str, float]]:
    """
    Get the best performing model based on a specific metric.
    
    Args:
        model_reports: Dictionary of model evaluation results
        metric: Metric to use for comparison
        higher_is_better: Whether higher values of the metric are better
        
    Returns:
        Tuple containing best model name and its metrics
    """
    try:
        if not model_reports:
            raise ValueError("No model reports provided")
        
        # Filter models that have the required metric
        valid_reports = {
            name: report for name, report in model_reports.items()
            if metric in report and report[metric] is not None
        }
        
        if not valid_reports:
            raise ValueError(f"No models have the metric '{metric}'")
        
        # Find best model
        if higher_is_better:
            best_model_name = max(valid_reports.keys(), key=lambda k: valid_reports[k][metric])
        else:
            best_model_name = min(valid_reports.keys(), key=lambda k: valid_reports[k][metric])
        
        best_metrics = valid_reports[best_model_name]
        
        logger.info(f"Best model: {best_model_name} with {metric}: {best_metrics[metric]:.4f}")
        
        return best_model_name, best_metrics
        
    except Exception as e:
        raise ModelTrainingError(
            "Error finding best model",
            error_detail=e,
            context={"metric": metric, "models": list(model_reports.keys())}
        )


def generate_model_report(
    model_reports: Dict[str, Dict[str, float]],
    sort_by: str = 'accuracy',
    ascending: bool = False,
    save_path: Optional[str] = os.path.join('artifacts', 'model_report.csv')
) -> pd.DataFrame:
    """
    Convert model_reports into a tidy DataFrame, sort by a metric, and optionally save as CSV.

    Args:
        model_reports: Output of evaluate_models
        sort_by: Metric column to sort by
        ascending: Sort order
        save_path: If provided, CSV will be saved at this path

    Returns:
        Pandas DataFrame with one row per model and metrics as columns
    """
    try:
        if not model_reports:
            raise ValueError("Empty model_reports provided")

        df = pd.DataFrame.from_dict(model_reports, orient='index')
        df.index.name = 'model'
        if sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=ascending)

        if save_path:
            ensure_dir(save_path)
            df.to_csv(save_path)
            logger.info(f"Model report saved to {save_path}")

        return df
    except Exception as e:
        raise ModelTrainingError(
            "Error generating model report",
            error_detail=e,
            context={"sort_by": sort_by, "save_path": save_path}
        )


def print_model_comparison(model_reports: Dict[str, Dict[str, float]]) -> None:
    """
    Print a formatted comparison of model performance.
    
    Args:
        model_reports: Dictionary of model evaluation results
    """
    try:
        if not model_reports:
            logger.warning("No model reports to display")
            return
        
        # Get all unique metrics
        all_metrics = set()
        for report in model_reports.values():
            all_metrics.update(report.keys())
        
        # Remove list metrics for display
        display_metrics = {m for m in all_metrics if not isinstance(next(iter(model_reports.values())).get(m), list)}
        
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)
        
        # Print header
        print(f"{'Model':<20}", end="")
        for metric in sorted(display_metrics):
            print(f"{metric:<12}", end="")
        print()
        print("-"*80)
        
        # Print model results
        for model_name, metrics in model_reports.items():
            print(f"{model_name:<20}", end="")
            for metric in sorted(display_metrics):
                value = metrics.get(metric, 0.0)
                if value is not None:
                    print(f"{value:<12.4f}", end="")
                else:
                    print(f"{'N/A':<12}", end="")
            print()
        
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error printing model comparison: {e}")


def create_directory_structure(base_path: str, directories: List[str]) -> None:
    """
    Create project directory structure.
    
    Args:
        base_path: Base directory path
        directories: List of subdirectories to create
    """
    try:
        for directory in directories:
            full_path = os.path.join(base_path, directory)
            os.makedirs(full_path, exist_ok=True)
            logger.info(f"Created directory: {full_path}")
    
    except Exception as e:
        raise CustomException(
            "Error creating directory structure",
            error_detail=e,
            context={"base_path": base_path, "directories": directories}
        )
