import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.utility import load_data, load_joblib_object, save_json_file, load_yaml_file
from src.entity.config_entity import ModelEvalConfig
from src.entity.artifact_entity import ModelEvalArtifacts, DataIngestionArtifacts, DataTransformationArtifacts, ModelTrainerArtifacts
from typing import Tuple
import sys
from src.logger import get_logger
from src.exception import SpaceshipTitanicException

logger = get_logger(__name__) 
class ModelEval:

    def __init__(
            self, config: ModelEvalConfig,
            data_ingested_artifact:DataIngestionArtifacts,
            data_transformed_artifact:DataTransformationArtifacts,
            model_trained_artifact:ModelTrainerArtifacts
            ) ->  None:
        
        self.config = config
        self.scheme_file = self.config.scheme_config
        self.model_config = self.config.model_config
        self.target_feature=self.scheme_file.get("target_feature")

        self.data_ingested_artifact=data_ingested_artifact
        self.data_transformed_artifact=data_transformed_artifact
        self.model_trained_artifact=model_trained_artifact

    def separate_target_feature(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        try:
            logger.info("Splitting Target column from testing data.")
            X = df.drop(columns=self.target_feature, errors="ignore")
            y = df[self.target_feature].values.ravel() if isinstance(self.target_feature, list) and len(self.target_feature) == 1 else df[self.target_feature].values
            logger.info("Target Variable seperated from training data.")
            return X, y

        except Exception as e:
            raise SpaceshipTitanicException(e,sys)

    def eval_model(self, model: object, X: pd.DataFrame, y: pd.Series) -> dict:
        try:
            logger.info("Evaluating Model.")
            predictions = model.predict(X)

            mse = mean_squared_error(y, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, predictions)
            r2 = r2_score(y, predictions)
            logger.info("Model evaluation Completed.")

            return {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "r2": float(r2)
            }
        
        except Exception as e:
            raise SpaceshipTitanicException(e,sys)

    def initiate_model_eval(self) -> ModelEvalArtifacts:
        try:

            df = load_data(self.data_ingested_artifact.test_data_path)
            preprocessor = load_joblib_object(self.data_transformed_artifact.preprocessor_path)
            model = load_joblib_object(self.model_trained_artifact.model_path)

            X, y = self.separate_target_feature(df)
            X_transformed = preprocessor.transform(X)
            report = self.eval_model(model, X_transformed, y)
            save_json_file(self.config.eval_file_path, report)



            model_eval_artifacts =  ModelEvalArtifacts(eval_file_path=self.config.eval_file_path)
            logger.info(f"Model evaluation report saved at: {model_eval_artifacts}")

            return model_eval_artifacts

        except Exception as e:
            raise SpaceshipTitanicException(e,sys)
        
if __name__ == "__main__":
    from src.config.configuration import ConfigurationManager

    logger.info(">>>>>>>>>>>>>>>>> Model Evaluation Component Started. <<<<<<<<<<<<<<<<<")

    config_manager = ConfigurationManager()

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

