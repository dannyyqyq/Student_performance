import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
from sklearn.metrics import r2_score


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # all rows, ex-last column
                train_array[:, -1],  # all rows, last column
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "K-Nearest Neighbors": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
            }

            params = {
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                },
                "Random Forest": {"n_estimators": [8, 16, 32, 64, 128, 256]},
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},  # No hyperparameters for Linear Regression
                "XGBoost": {  # Match the key name from the models dictionary
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "CatBoost": {  # Match the key name from the models dictionary
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "AdaBoost": {  # Match the key name from the models dictionary
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "K-Nearest Neighbors": {},  # No hyperparameters for KNN
            }

            # Running through a list of models and appending the scores
            model_report: dict = evaluate_model(
                X_train, y_train, X_test, y_test, models=models, param=params
            )

            best_model = None
            # To make sure score is always sorted by the smallest value
            best_model_score = float("-inf")
            best_model_params = None
            # To obtain best score from list of scores from model
            for model_name, scores in model_report.items():
                test_scores = scores["Test r2 score"]
                # Sort model k,v by the score
                if test_scores > best_model_score:
                    best_model_name = model_name
                    best_model_score = test_scores
                    best_model_params = scores["Best params"]
            # Get the best model object from the models dictionary
            best_model = models[best_model_name]

            # Log and return model name and test score
            logging.info(
                f"\nBest model name: {best_model_name},  \nBest test r2 score : {best_model_score}, \nBest params: {best_model_params}"
            )

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("Best model found on training and test dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            y_pred = best_model.predict(X_test)
            r2_square = r2_score(y_test, y_pred)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
