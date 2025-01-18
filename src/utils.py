import os
import sys
from sklearn.metrics import r2_score
import dill
from src.exception import CustomException


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


def evaluate_model(X_train, y_train, X_test, y_test, models) -> dict:
    """
    Evaluate multiple models on a given dataset and return a dict {model : score}.
    Arg:
        X_train (array): The features for the training set.
        y_train (array): The target variable for the training set.
        X_test (array): The features for the testing set.
        y_test (array): The target variable for the testing set.
        models (dict): A dictionary of models to evaluate, where the key is the model name and the value is the model object.

    Returns:
        dict: A dictionary containing the model name as the key and a dictionary with keys "Train r2 score" and "Test r2 score" as the value.
    """
    try:
        model_scores = {}
        for name, model in models.items():
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)

            test_score = r2_score(y_test, y_test_pred)

            # concentate the rest of the model score
            model_scores[name] = {
                "Train r2 score": train_score,
                "Test r2 score": test_score,
            }

        return model_scores
    except Exception as e:
        raise CustomException(e, sys)
