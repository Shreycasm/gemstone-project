import pandas as pd
from pathlib import Path
import sys
from sklearn.ensemble import RandomForestRegressor
from src.utils.utility import load_joblib_object, save_joblib_object, load_data, load_yaml_file
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ModelTrainerArtifacts
from typing import Tuple
from src.logger import get_logger
from src.exception import SpaceshipTitanicException

logger = get_logger(__name__)

class ModelTrainer:

    def __init__(self, config: ModelTrainerConfig = ModelTrainerConfig()) -> None:
        self.config = config
        self.load_file = load_yaml_file(self.config.model_trainer_scheme_file_path)
        self.target = self.load_file.get("target_feature", [])

    def separate_target_feature(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        try:
            logger.info("Splitting Target column.")
            X = df.drop(columns=self.target, errors="ignore")
            y = df[self.target].values.ravel() if isinstance(self.target, list) and len(self.target) == 1 else df[self.target].values
            logger.info("Target Variable seperated,")
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

    def initiate_model_trainer(self, train_data_path: Path, preprocessor_path: Path) -> ModelTrainerArtifacts:
        try:
            logger.info("Model Trainer Component Started...")
            preprocessor = load_joblib_object(preprocessor_path)
            df = load_data(train_data_path)
            X, y = self.separate_target_feature(df)
            X_transformed = preprocessor.transform(X)

            model = self.train_model(X_transformed, y)
            save_joblib_object(self.config.model_path, model)

            model_artifact =  ModelTrainerArtifacts(model_path=self.config.model_path)
            logger.info(f"Model saved at {model_artifact}")

            logger.info("Model Traning Completed.")
            return model_artifact
        except Exception as e:
            raise SpaceshipTitanicException(e,sys)
