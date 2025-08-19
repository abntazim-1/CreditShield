import os
import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models, generate_model_report
import numpy as np

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, column_names):
        """
        Convert arrays to DataFrames to locate the target column by name.
        """
        try:
            logging.info("Converting arrays to DataFrames to locate target column")
            target_column_name="loan_status"
            # Convert arrays to DataFrames
            train_df = pd.DataFrame(train_array, columns=column_names)
            test_df = pd.DataFrame(test_array, columns=column_names)

            # Check target column
            if target_column_name not in train_df.columns:
                raise CustomException(f"Target column '{target_column_name}' not found in train data")

            # Split features and target
            X_train = train_df.drop(columns=[target_column_name]).values
            y_train = train_df[target_column_name].values

            X_test = test_df.drop(columns=[target_column_name]).values
            y_test = test_df[target_column_name].values

            logging.info(f"Target column '{target_column_name}' successfully found")


            # Define candidate models
            models = {
                "LogisticRegression": LogisticRegression(max_iter=200),
                "RandomForest": RandomForestClassifier(),
                "DecisionTree": DecisionTreeClassifier(),
                "GradientBoosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "KNN": KNeighborsClassifier(),
                "SVM": SVC(probability=True),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
                "LightGBM": LGBMClassifier(),
                "CatBoost": CatBoostClassifier(verbose=0),
            }

            # Hyperparameters
            params = {
                "DecisionTree": {"criterion": ["gini", "entropy"]},
                "RandomForest": {"n_estimators": [50, 100, 200]},
                "GradientBoosting": {"learning_rate": [0.1, 0.05], "n_estimators": [50, 100]},
                "LogisticRegression": {"C": [0.1, 1, 10]},
                "XGBoost": {"learning_rate": [0.1, 0.05], "n_estimators": [50, 100]},
                "LightGBM": {"num_leaves": [31, 50], "n_estimators": [100,200]},
                "CatBoost": {"depth": [6, 8], "iterations": [50, 100]},
                "AdaBoost": {"n_estimators": [50, 100]},
                "KNN": {"n_neighbors": [3, 5, 7]},
                "SVM": {"C": [0.1, 1, 10]},
            }

            # Evaluate models
            logging.info("Evaluating all models...")
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param_grids=params,
                scoring="accuracy"
            )

            # Persist a human-readable model comparison report
            report_df = generate_model_report(
                model_reports=model_report,
                sort_by="accuracy",
                ascending=False,
                save_path=os.path.join("artifacts", "model_report.csv")
            )
            best_model_name = report_df.index[0]
            best_model_metrics = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(
                f"Best model: {best_model_name} with accuracy: {best_model_metrics.get('accuracy', 0.0)}"
            )

            if best_model_metrics.get("accuracy", 0.0) < 0.6:
                raise CustomException("No suitable model found with sufficient accuracy")

            # Build ensemble of top 3 models
            top3 = [(name, models[name]) for name in report_df.index[:3]]
            ensemble_model = VotingClassifier(estimators=top3, voting="soft")
            logging.info(f"Ensemble created with top 3 models: {[n for n, _ in top3]}")

            # Train ensemble
            ensemble_model.fit(X_train, y_train)

            # Save ensemble model
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=ensemble_model)

            # Final evaluation
            predictions = ensemble_model.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            auc = roc_auc_score(y_test, predictions)

            logging.info("Final Ensemble Model Metrics:")
            logging.info(f"Accuracy: {acc}")
            logging.info(f"F1 Score: {f1}")
            logging.info(f"ROC-AUC: {auc}")
            logging.info(f"Model saved at: {self.model_trainer_config.trained_model_file_path}")

            return acc

        except Exception as e:
            raise CustomException(e, sys)
