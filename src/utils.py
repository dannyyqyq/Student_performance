import os
import sys
from sklearn.metrics import r2_score
import dill
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj) -> object:
    """
    Save a Python object to a file using dill.
    Arg:
        file_path (str) : The path to the file where the object will be saved.
        obj (object): The Python object to be saved.

    Returns:
        object: The saved object.
    """

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path) -> str:
    """
    Load a Python object from a file using dill.

    Arg:
        file_path (str): The path to the file from which the object will be loaded.
        obj (object): The Python object to be loaded.

    Returns:
        object: The loaded object.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models, param) -> dict:
    """
    Evaluate multiple models on a given dataset and return a dict {model : score}.
    Arg:
        X_train (array): The features for the training set.
        y_train (array): The target variable for the training set.
        X_test (array): The features for the testing set.
        y_test (array): The target variable for the testing set.
        models (dict): A dictionary of models to evaluate, where the key is the model model_name and the value is the model object.

    Returns:
        dict: A dictionary containing the model model_name as the key and a dictionary with keys "Train r2 score" and "Test r2 score" as the value.
    """
    try:
        model_scores = {}
        for model_name, model in models.items():
            best_model = None
            best_params = None
            try:
                # Check if there are parameters assigned to model
                if model_name in param and param[model_name]:
                    logging.info(f"Tuning hyperparameters for {model_name}")
                    grid_search = GridSearchCV(
                        estimator=model, param_grid=param[model_name], cv=3
                    )
                    grid_search.fit(X_train, y_train)

                    best_model = grid_search.best_estimator_  # Get the best model
                    best_params = (
                        grid_search.best_params_
                    )  # Get the best hyperparameters
                else:
                    logging.info(
                        f"No hyperparameters specified for {model_name}. Using default model."
                    )
                    best_model = model.fit(X_train, y_train)

                # model.fit(X_train, y_train)

                # y_train_pred = model.predict(X_train)
                y_train_pred = best_model.predict(X_train)

                # y_test_pred = model.predict(X_test)
                y_test_pred = best_model.predict(X_test)

                train_score = r2_score(y_train, y_train_pred)

                test_score = r2_score(y_test, y_test_pred)

                # concentate the rest of the model score
                model_scores[model_name] = {
                    "Best params": best_params,
                    "Train r2 score": train_score,
                    "Test r2 score": test_score,
                }
            except Exception as e:
                logging.error(
                    f"Error occurred while evaluating model {model_name}: {e}"
                )
                raise CustomException(e, sys)

        return model_scores

    except Exception as e:
        raise CustomException(e, sys)
