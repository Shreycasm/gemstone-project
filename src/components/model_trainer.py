import pandas as pd
from pathlib import Path
import sys
from sklearn.ensemble import RandomForestRegressor
from src.utils.utility import load_joblib_object, save_joblib_object, load_data
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ModelTrainerArtifacts, DataIngestionArtifacts, DataTransformationArtifacts
from typing import Tuple
from src.logger import get_logger
from src.exception import SpaceshipTitanicException

logger = get_logger(__name__)

class ModelTrainer:

    def __init__(
            self, config: ModelTrainerConfig,
            data_ingestion_artifact: DataIngestionArtifacts,
            data_transformation_artifact: DataTransformationArtifacts 
            ) -> None:
        
        self.config = config
        self.scheme_config = self.config.scheme_config 
        self.model_config = self.config.model_config
        self.data_ingested_artifact = data_ingestion_artifact
        self.data_transformated_artifact = data_transformation_artifact

    def separate_target_feature(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        try:
            logger.info("Splitting Target column from traning data.")

            target_feature=self.scheme_config.get("target_feature")
            X = df.drop(columns=target_feature, errors="ignore")

            y = df[target_feature].values.ravel() if isinstance(target_feature, list) and len(target_feature) == 1 else df[target_feature].values
            logger.info("Target Variable seperated from training data.")
            return X, y
        
        except Exception as e:
            raise SpaceshipTitanicException(e,sys)

    def train_model(self, X, y, model=None):
        try:
            logger.info("Traning Model.")
            if model is None:
                model = RandomForestRegressor(random_state=42, n_jobs=-1)
            model.fit(X, y)
            logger.info("Model traning Completed.")
            return model
        
        except Exception as e:
            raise SpaceshipTitanicException(e,sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifacts:
        try:

            preprocessor = load_joblib_object(self.data_transformated_artifact.preprocessor_path)
            df = load_data(self.data_ingested_artifact.train_data_path)

            X, y = self.separate_target_feature(df)
            X_transformed = preprocessor.transform(X)

            model = self.train_model(X_transformed, y)
            save_joblib_object(self.config.model_path, model)

            model_artifact =  ModelTrainerArtifacts(model_path=self.config.model_path)
            logger.info(f"Model saved at {model_artifact}")


            return model_artifact
            
        except Exception as e:
            raise SpaceshipTitanicException(e,sys)
        
if __name__ == "__main__":
    from src.config.configuration import ConfigurationManager

    logger.info(">>>>>>>>>>>>>>>>> Model Trainer Component Started. <<<<<<<<<<<<<<<<<")

    config_manager = ConfigurationManager()
    model_trainer_config = config_manager.get_model_trainer_config()

    data_ingested_config = config_manager.get_data_ingestion_config()
    data_transformed_config = config_manager.get_data_transformation_config()

    model_trainer_config = config_manager.get_model_trainer_config()

    data_ingestion_artifact = DataIngestionArtifacts(
        train_data_path=data_ingested_config.train_data_file_path,
        test_data_path=data_ingested_config.test_data_file_path
    )

    data_transformation_artifact = DataTransformationArtifacts(
        preprocessor_path=data_transformed_config.preprocessor_file_path
    )

    obj = ModelTrainer(
        config=model_trainer_config,
        data_ingestion_artifact=data_ingestion_artifact,
        data_transformation_artifact=data_transformation_artifact
    )
    obj.initiate_model_trainer()

    logger.info(">>>>>>>>>>>>>>>>> Model Trainer Component Completed. <<<<<<<<<<<<<<<<<")
