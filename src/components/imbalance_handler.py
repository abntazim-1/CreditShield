"""
Imbalance detection and handling utilities for the credit risk project.

This module provides comprehensive tools for detecting class imbalance and
applying various resampling techniques to address it.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, Union
from collections import Counter
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

# Resampling libraries
from imblearn.over_sampling import (
    RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
)
from imblearn.under_sampling import (
    RandomUnderSampler, EditedNearestNeighbours, RepeatedEditedNearestNeighbours,
    AllKNN, CondensedNearestNeighbour, OneSidedSelection, NeighbourhoodCleaningRule
)
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.utils.class_weight import compute_class_weight

from src.exception import CustomException, DataValidationError, FeatureEngineeringError
from src.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


@dataclass
class ImbalanceConfig:
    """Configuration for imbalance detection and handling."""
    
    # Detection thresholds - FIXED VALUES
    imbalance_ratio_threshold: float = 0.4  # If minority class < 40%, consider imbalanced
    severe_imbalance_threshold: float = 0.1  # If minority class < 10%, severely imbalanced
    
    # Alternative: You could also use ratio-based thresholds
    # imbalance_ratio_threshold_ratio: float = 0.8  # If minority/majority ratio < 0.8, imbalanced
    # severe_imbalance_threshold_ratio: float = 0.3  # If minority/majority ratio < 0.3, severe
    
    # Resampling strategy
    resampling_strategy: str = "auto"
    target_ratio: float = 0.5  # Target minority class ratio after resampling
    
    # Specific technique
    technique: str = "SMOTE"
    
    # Random state for reproducibility
    random_state: int = 42
    
    # Visualization
    create_plots: bool = True
    plot_save_path: str = "artifacts/plots"


class ImbalanceDetector:
    """Class for detecting and analyzing class imbalance."""
    
    def __init__(self, config: Optional[ImbalanceConfig] = None):
        """
        Initialize ImbalanceDetector.
        
        Args:
            config: ImbalanceConfig object
        """
        self.config = config or ImbalanceConfig()
        logger.info("ImbalanceDetector initialized")
    
    def analyze_class_distribution(self, y: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Analyze class distribution and detect imbalance.
        
        Args:
            y: Target variable
            
        Returns:
            Dictionary containing distribution analysis
        """
        try:
            # Convert to numpy array if pandas Series
            if isinstance(y, pd.Series):
                y = y.values
            
            # Count classes
            class_counts = Counter(y)
            total_samples = len(y)
            
            # Calculate ratios
            class_ratios = {cls: count/total_samples for cls, count in class_counts.items()}
            
            # Find majority and minority classes
            majority_class = max(class_counts, key=class_counts.get)
            minority_class = min(class_counts, key=class_counts.get)
            
            majority_count = class_counts[majority_class]
            minority_count = class_counts[minority_class]
            
            # Calculate imbalance ratio
            imbalance_ratio = minority_count / majority_count
            minority_percentage = (minority_count / total_samples) * 100
            
            # Determine imbalance level
            if minority_percentage < self.config.severe_imbalance_threshold * 100:
                imbalance_level = "severe"
            elif minority_percentage < self.config.imbalance_ratio_threshold * 100:
                imbalance_level = "moderate"
            else:
                imbalance_level = "balanced"
            
            analysis = {
                "total_samples": total_samples,
                "class_counts": dict(class_counts),
                "class_ratios": class_ratios,
                "majority_class": majority_class,
                "minority_class": minority_class,
                "majority_count": majority_count,
                "minority_count": minority_count,
                "imbalance_ratio": imbalance_ratio,
                "minority_percentage": minority_percentage,
                "imbalance_level": imbalance_level,
                "needs_resampling": imbalance_level != "balanced"
            }
            
            # Log the analysis
            logger.info(f"Class distribution analysis completed:")
            logger.info(f"  - Total samples: {total_samples}")
            logger.info(f"  - Class counts: {class_counts}")
            logger.info(f"  - Minority class percentage: {minority_percentage:.2f}%")
            logger.info(f"  - Imbalance level: {imbalance_level}")
            logger.info(f"  - Needs resampling: {analysis['needs_resampling']}")
            
            return analysis
            
        except Exception as e:
            raise DataValidationError(
                "Error during class distribution analysis",
                error_detail=e
            )
    
    def visualize_distribution(
        self, 
        y: Union[np.ndarray, pd.Series], 
        title: str = "Class Distribution",
        save_path: Optional[str] = None
    ) -> None:
        """
        Create visualization of class distribution.
        
        Args:
            y: Target variable
            title: Plot title
            save_path: Path to save the plot
        """
        try:
            if not self.config.create_plots:
                return
            
            # Convert to pandas Series for easier plotting
            if isinstance(y, np.ndarray):
                y = pd.Series(y)
            
            # Create subplots
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Count plot
            y.value_counts().plot(kind='bar', ax=axes[0], color=['skyblue', 'lightcoral'])
            axes[0].set_title(f'{title} - Counts')
            axes[0].set_xlabel('Class')
            axes[0].set_ylabel('Count')
            axes[0].tick_params(axis='x', rotation=0)
            
            # Add count annotations
            for i, v in enumerate(y.value_counts()):
                axes[0].text(i, v + 0.01 * max(y.value_counts()), str(v), ha='center', va='bottom')
            
            # Pie chart
            y.value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', 
                                 colors=['skyblue', 'lightcoral'])
            axes[1].set_title(f'{title} - Proportions')
            axes[1].set_ylabel('')
            
            plt.tight_layout()
            
            # Save plot if path provided
            if save_path:
                import os
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Class distribution plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.warning(f"Could not create visualization: {e}")


class ImbalanceHandler:
    """Class for handling class imbalance using various resampling techniques."""
    
    def __init__(self, config: Optional[ImbalanceConfig] = None):
        """
        Initialize ImbalanceHandler.
        
        Args:
            config: ImbalanceConfig object
        """
        self.config = config or ImbalanceConfig()
        logger.info("ImbalanceHandler initialized")
    
    def get_resampling_strategy(self, analysis: Dict[str, Any]) -> str:
        """
        Determine the best resampling strategy based on data analysis.
        
        Args:
            analysis: Class distribution analysis
            
        Returns:
            Recommended resampling strategy
        """
        if not analysis["needs_resampling"]:
            return "none"
        
        if self.config.resampling_strategy != "auto":
            return self.config.resampling_strategy
        
        # Auto strategy selection
        minority_percentage = analysis["minority_percentage"]
        total_samples = analysis["total_samples"]
        
        if minority_percentage < 1:  # Less than 1%
            return "combine"  # Use combined methods
        elif minority_percentage < 5:  # 1-5%
            return "oversample"  # Focus on oversampling
        elif total_samples > 100000:  # Large dataset
            return "undersample"  # Undersampling for efficiency
        else:
            return "oversample"  # Default to oversampling
    
    def get_resampler(self, strategy: str, technique: str) -> Any:
        """
        Get the appropriate resampler based on strategy and technique.
        
        Args:
            strategy: Resampling strategy (oversample, undersample, combine)
            technique: Specific technique name
            
        Returns:
            Resampler object
        """
        random_state = self.config.random_state
        
        # Oversampling techniques
        if strategy == "oversample":
            oversample_techniques = {
                "RandomOverSampler": RandomOverSampler(random_state=random_state),
                "SMOTE": SMOTE(random_state=random_state),
                "ADASYN": ADASYN(random_state=random_state),
                "BorderlineSMOTE": BorderlineSMOTE(random_state=random_state),
                "SVMSMOTE": SVMSMOTE(random_state=random_state)
            }
            return oversample_techniques.get(technique, SMOTE(random_state=random_state))
        
        # Undersampling techniques
        elif strategy == "undersample":
            undersample_techniques = {
                "RandomUnderSampler": RandomUnderSampler(random_state=random_state),
                "EditedNearestNeighbours": EditedNearestNeighbours(),
                "RepeatedEditedNearestNeighbours": RepeatedEditedNearestNeighbours(),
                "AllKNN": AllKNN(),
                "CondensedNearestNeighbour": CondensedNearestNeighbour(random_state=random_state),
                "OneSidedSelection": OneSidedSelection(random_state=random_state),
                "NeighbourhoodCleaningRule": NeighbourhoodCleaningRule()
            }
            return undersample_techniques.get(technique, RandomUnderSampler(random_state=random_state))
        
        # Combined techniques
        elif strategy == "combine":
            combine_techniques = {
                "SMOTEENN": SMOTEENN(random_state=random_state),
                "SMOTETomek": SMOTETomek(random_state=random_state)
            }
            return combine_techniques.get(technique, SMOTEENN(random_state=random_state))
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def apply_resampling(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.Series], Dict[str, Any]]:
        """
        Apply resampling to address class imbalance.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Tuple of (resampled_X, resampled_y, resampling_info)
        """
        try:
            logger.info("Starting resampling process")
            
            # Analyze current distribution
            detector = ImbalanceDetector(self.config)
            analysis = detector.analyze_class_distribution(y)
            
            # Visualize original distribution
            if self.config.create_plots:
                detector.visualize_distribution(
                    y, 
                    "Original Class Distribution",
                    f"{self.config.plot_save_path}/original_distribution.png"
                )
            
            # Check if resampling is needed
            if not analysis["needs_resampling"]:
                logger.info("Data is already balanced, no resampling needed")
                return X, y, {"resampling_applied": False, "original_analysis": analysis}
            
            # Determine strategy
            strategy = self.get_resampling_strategy(analysis)
            logger.info(f"Selected resampling strategy: {strategy}")
            
            if strategy == "none":
                return X, y, {"resampling_applied": False, "original_analysis": analysis}
            
            # Get resampler
            resampler = self.get_resampler(strategy, self.config.technique)
            logger.info(f"Using technique: {self.config.technique}")
            
            # Apply resampling
            X_resampled, y_resampled = resampler.fit_resample(X, y)
            
            # Analyze new distribution
            new_analysis = detector.analyze_class_distribution(y_resampled)
            
            # Visualize new distribution
            if self.config.create_plots:
                detector.visualize_distribution(
                    y_resampled,
                    "Resampled Class Distribution",
                    f"{self.config.plot_save_path}/resampled_distribution.png"
                )
            
            # Create resampling info
            resampling_info = {
                "resampling_applied": True,
                "strategy": strategy,
                "technique": self.config.technique,
                "original_analysis": analysis,
                "new_analysis": new_analysis,
                "original_shape": X.shape,
                "new_shape": X_resampled.shape,
                "samples_added": X_resampled.shape[0] - X.shape[0]
            }
            
            logger.info(f"Resampling completed:")
            logger.info(f"  - Original shape: {X.shape}")
            logger.info(f"  - New shape: {X_resampled.shape}")
            logger.info(f"  - Samples added: {resampling_info['samples_added']}")
            logger.info(f"  - New minority percentage: {new_analysis['minority_percentage']:.2f}%")
            
            return X_resampled, y_resampled, resampling_info
            
        except Exception as e:
            raise FeatureEngineeringError(
                "Error during resampling process",
                error_detail=e,
                context={
                    "strategy": strategy if 'strategy' in locals() else "unknown",
                    "technique": self.config.technique
                }
            )
    
    def get_class_weights(self, y: Union[np.ndarray, pd.Series]) -> Dict[Any, float]:
        """
        Calculate class weights for algorithms that support class weighting.
        
        Args:
            y: Target variable
            
        Returns:
            Dictionary mapping classes to their weights
        """
        try:
            if isinstance(y, pd.Series):
                y = y.values
            
            classes = np.unique(y)
            weights = compute_class_weight('balanced', classes=classes, y=y)
            class_weights = dict(zip(classes, weights))
            
            logger.info(f"Calculated class weights: {class_weights}")
            return class_weights
            
        except Exception as e:
            raise DataValidationError(
                "Error calculating class weights",
                error_detail=e
            )


def detect_and_handle_imbalance(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    config: Optional[ImbalanceConfig] = None
) -> Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.Series], Dict[str, Any]]:
    """
    Convenience function to detect and handle class imbalance in one step.
    
    Args:
        X: Feature matrix
        y: Target vector
        config: ImbalanceConfig object
        
    Returns:
        Tuple of (processed_X, processed_y, processing_info)
    """
    handler = ImbalanceHandler(config)
    return handler.apply_resampling(X, y)


    if __name__ == "__main__":
        logger.info("Imbalance handler module loaded successfully")