import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_file_path: str = os.path.join("artifacts", "Preprocessing.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer(self):
        """
        This function generates a preprocessor object which will be used to transform the dataset
        Returns:
            preprocessor: ColumnTransformer object that contains the numerical and categorical pipeline
        """
        try:
            numerical_features = ["writing_score", "reading_score"]
            categorical_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  # Due to outliers
                    ("scaler", StandardScaler()),
                ]
            )
            logging.info("Numerical columns transformed.")

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(drop="first")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )
            logging.info("Categorical columns transformed.")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("numerical_pipeline", numerical_pipeline, numerical_features),
                    (
                        "categorical_pipeline",
                        categorical_pipeline,
                        categorical_features,
                    ),
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Load train and test data completed")
            logging.info("Obtaining preprocessing object")
            # start preprocessor object
            preprocessor_object = self.get_data_transformer()

            target_column = "math_score"
            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info(f"Training columns: {input_feature_train_df.columns.tolist()}")
            logging.info(f"Target columns: {[target_column]}")

            input_feature_train_array = preprocessor_object.fit_transform(
                input_feature_train_df
            )
            input_feature_test_array = preprocessor_object.transform(
                input_feature_test_df
            )

            # np.c_ used to concentate arrays horizontally -side by side
            train_array = np.c_[
                input_feature_train_array, np.array(target_feature_train_df)
            ]
            test_array = np.c_[
                input_feature_test_array, np.array(target_feature_test_df)
            ]

            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                obj=preprocessor_object,
            )

            logging.info("Preprocessing object saved")

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
