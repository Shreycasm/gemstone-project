import os, sys
from pathlib import Path
from src.entity.config_entity import  (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvalConfig,
    ConfigurationManagerConfig
)

from src.logger import get_logger
from src.exception import SpaceshipTitanicException
from src.utils.utility import load_yaml_file


logger = get_logger(__name__)

class ConfigurationManager:

    def __init__(
            self, config: ConfigurationManagerConfig = ConfigurationManagerConfig()
            ) -> None:
        try:
            self.config = config
            scheme_config_file_path = self.config.scheme_config_file_path
            model_config_file_path = self.config.model_config_file_path
            self.scheme_config = load_yaml_file(scheme_config_file_path)
            self.model_config = load_yaml_file(model_config_file_path)

            logger.info("Configuration manager initialized  Successfully.")
        
        except Exception as e:
            raise SpaceshipTitanicException(e, sys)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            data_ingestion_config = DataIngestionConfig()
            logger.info("Data Ingestion Config called")

            return data_ingestion_config

        except Exception as e:
            raise SpaceshipTitanicException(e, sys)


    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            data_transformation_config = DataTransformationConfig(scheme_file=self.scheme_config)
            logger.info("Data Transformation Config called")

            return data_transformation_config

        except Exception as e:
            raise SpaceshipTitanicException(e, sys)
        

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        try:
            model_trainer_config = ModelTrainerConfig(
                model_config=self.model_config,
                scheme_config=self.scheme_config
                )
            
            logger.info("Model Trainer Config called")

            return model_trainer_config

        except Exception as e:
            raise SpaceshipTitanicException(e, sys)
        

    def get_model_eval_config(self) -> ModelEvalConfig:
        try:
            model_eval_config = ModelEvalConfig(
                model_config=self.model_config,
                scheme_config=self.scheme_config
            )
            logger.info("Model Evaluation Config created")

            return model_eval_config

        except Exception as e:
            raise SpaceshipTitanicException(e, sys)


        





