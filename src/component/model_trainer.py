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

            # Running through a list of models and appending the scores
            model_report: dict = evaluate_model(
                X_train, y_train, X_test, y_test, models=models
            )

            best_model = None
            best_model_score = float("-inf")
            # To obtain best score from list of scores from model
            for model_name, scores in model_report.items():
                test_scores = scores["Test r2 score"]
                # Sort model k,v by the score
                if test_scores > best_model_score:
                    best_model_name = model_name
                    best_model_score = test_scores

            # Get the best model object from the models dictionary
            best_model = models[best_model_name]

            # Log and return model name and test score
            logging.info(
                f"Best model name: {best_model_name},  Best test r2 score : {best_model_score}"
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
