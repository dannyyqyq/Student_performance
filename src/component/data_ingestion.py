import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    # by default os.path.join == project root
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        """
        Initializes the DataIngestion instance with a default configuration.

        Attributes:
            ingestion_config (DataIngestionConfig): Configuration object containing file paths for train, test, and raw data.
        """

        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Initiates the data ingestion process.

        This method reads the dataset from notebooks/data/, creates a directory for storing the train,
        test, and raw data, performs a train-test split, and saves the data to the respective files.

        Return:
            A tuple containing the file paths for the train and test data
        """
        logging.info("Enter data ingestion component")
        try:
            df = pd.read_csv("notebooks/data/stud.csv")
            logging.info("Read the dataset as dataframe")
            directory = os.path.dirname(
                self.ingestion_config.train_data_path
            )  # Create artifacts folder
            os.makedirs(directory, exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(
                f"Dataframe shape: {df.shape}, \nDataframe columns: {list(df.columns)}"
            )

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            logging.info(f"Train dataframe shape: {train_set.shape}")
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )
            logging.info(f"Test dataframe shape: {test_set.shape}.")
            logging.info("Data ingestion process completed successfully.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_array, test_array, _ = data_transformation.initiate_data_transformation(
        train_data, test_data
    )

    model_trainer = ModelTrainer()

    model_trainer.initiate_model_trainer(train_array, test_array)
