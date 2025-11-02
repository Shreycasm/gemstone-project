from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_eval import ModelEval
from src.logger import get_logger
from src.exception import SpaceshipTitanicException
import sys
from src.config.configuration import ConfigurationManager
from src.entity.artifact_entity import (
    DataIngestionArtifacts,
    DataTransformationArtifacts,
    ModelTrainerArtifacts,
    ModelEvalArtifacts
)

logger = get_logger(__name__)


class TrainingPipeline:

    def initiate_training_pipeline(self):
        try :

            logger.info(">>>>>>>>>>>>>>>>> Traning Pipeline Started... <<<<<<<<<<<<<<<<<")

            logger.info(">>>>>>>>>>>>>>>>> Data Ingestion Component Started. <<<<<<<<<<<<<<<<<")

            config_manager = ConfigurationManager()
            data_ingestion_config = config_manager.get_data_ingestion_config()

            obj = DataIngestion(config = data_ingestion_config)
            artifacts = obj.initiate_data_ingestion()

            logger.info(">>>>>>>>>>>>>>>>> Data Ingestion Component Completed. <<<<<<<<<<<<<<<<<")

            logger.info(">>>>>>>>>>>>>>>>> Data Transformation Component Started. <<<<<<<<<<<<<<<<<")

            data_transformation_config = config_manager.get_data_transformation_config()

            data_ingested_config = config_manager.get_data_ingestion_config()
            data_ingestion_artifacts = DataIngestionArtifacts(
                train_data_path=data_ingested_config.train_data_file_path,
                test_data_path=data_ingested_config.test_data_file_path
            )

            obj = DataTransformation(
                config = data_transformation_config,
                data_ingestion_artifact = data_ingestion_artifacts
            )
            data_transformed_artifacts = obj.initiate_data_transformation()

            logger.info(">>>>>>>>>>>>>>>>> Data Transformation Component Completed  . <<<<<<<<<<<<<<<<<")

            logger.info(">>>>>>>>>>>>>>>>> Model Trainer Component Started. <<<<<<<<<<<<<<<<<")

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

            logger.info(">>>>>>>>>>>>>>>>> Model Evaluation Component Started. <<<<<<<<<<<<<<<<<")

            model_eval_config  = config_manager.get_model_eval_config()
            data_ingested_config = config_manager.get_data_ingestion_config()
            data_transformed_config = config_manager.get_data_transformation_config()
            model_trained_config = config_manager.get_model_trainer_config()

            data_ingested_artifact = DataIngestionArtifacts(
                train_data_path=data_ingested_config.train_data_file_path,
                test_data_path=data_ingested_config.test_data_file_path
            )

            data_transformed_artifact = DataTransformationArtifacts(
                preprocessor_path=data_transformed_config.preprocessor_file_path
            )

            model_trained_artifact = ModelTrainerArtifacts(
                model_path=model_trained_config.model_path
            )

            obj = ModelEval(
                config=model_eval_config,
                data_ingested_artifact=data_ingested_artifact,
                data_transformed_artifact=data_transformed_artifact,
                model_trained_artifact=model_trained_artifact
            )
            obj.initiate_model_eval()

            logger.info(">>>>>>>>>>>>>>>>> Model Evaluation Component Completed. <<<<<<<<<<<<<<<<<")
            logger.info(">>>>>>>>>>>>>>>>> Traning Pipeline Completed. <<<<<<<<<<<<<<<<<")

        except Exception as e:
            raise SpaceshipTitanicException(e,sys)


if __name__ == "__main__":
    training_pipeline = TrainingPipeline()
    training_pipeline.initiate_training_pipeline()
