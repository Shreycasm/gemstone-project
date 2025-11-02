import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from src.utils.utility import load_data, save_joblib_object
from src.utils.preprocessing_utils import clip_outliers_array
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifacts, DataIngestionArtifacts
from src.logger import get_logger
from src.exception import SpaceshipTitanicException

logger = get_logger(__name__)

class DataTransformation:

    def __init__(
            self, config: DataTransformationConfig,
            data_ingestion_artifact: DataIngestionArtifacts
            ):
        self.config = config
        self.scheme_config = self.config.scheme_file 
        self.data_ingested_artifact = data_ingestion_artifact

    def change_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Dropping unwanted columns.")
            drop_columns = self.scheme_config.get("drop_columns")

            if drop_columns:
                df = df.drop(columns=drop_columns, errors="ignore")
            logger.info("Unwanted columns dropped.")

            logger.info("Changing The dtypes of columns to save memoory.")
            dtypes_map = self.scheme_config.get("change_dtypes", {})
            for col, dtype in dtypes_map.items():
                if col in df.columns:
                    df[col] = df[col].astype(dtype)
            logger.info("Saved memory and space by changing the dtypes of columns.")

            return df
        
        except Exception as e:
            raise SpaceshipTitanicException(e,sys)

    # def _clip_outliers_array(self, X: np.ndarray) -> np.ndarray:
    #     try:
    #         logger.info("Clipping Outliers to 1% and 99%")
    #         lower = np.quantile(X, 0.01, axis=0)
    #         upper = np.quantile(X, 0.99, axis=0)
    #         logger.info("Outliers clipped successfully.")
    #         return np.clip(X, lower, upper)
        
    #     except Exception as e:
    #         raise SpaceshipTitanicException(e,sys)

    def preprocessor(self) -> ColumnTransformer:
        try:
            logger.info("Preprocessing the data.")
            numeric_transform = Pipeline([
                    ("clipper", FunctionTransformer(clip_outliers_array, validate=False))
                ])

            logger.info("OneHot Encoding the categorical columns.")
            categorical_transform = OneHotEncoder(handle_unknown="ignore")
            logger.info("OneHot Encoded the categorical columns.")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transform, self.config.scheme_file.get("numeric_features")),
                    ("cat", categorical_transform, self.config.scheme_file.get("categorical_features"))
                ],
                remainder="drop",
                verbose_feature_names_out=False
            )
            
            logger.info("Data preprocessing Completed.")

            return preprocessor
        
        except Exception as e:
            raise SpaceshipTitanicException(e,sys)

    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        try:

            df = load_data(self.data_ingested_artifact.train_data_path)
            df = self.change_dtypes(df)
            features = list(self.config.scheme_file.get("numeric_features")) + list(self.config.scheme_file.get("categorical_features"))
            features = [c for c in features if c in df.columns]

            if not features:
                raise ValueError("No feature columns found for preprocessor fit.")

            X_for_fit = df[features]

            preprocessor_obj = self.preprocessor()
            preprocessor_obj.fit(X_for_fit)

            save_joblib_object(self.config.preprocessor_file_path, preprocessor_obj)
            preprocessor_artifact = DataTransformationArtifacts(preprocessor_path=self.config.preprocessor_file_path)
            logger.info(f"Preprocessor Artifact saved at {preprocessor_artifact}")


            return preprocessor_artifact


        except Exception as e:
            raise SpaceshipTitanicException(e,sys)

 
        
if __name__ =="__main__":
    from src.config.configuration import ConfigurationManager

    logger.info(">>>>>>>>>>>>>>>>> Data Transformation Component Started. <<<<<<<<<<<<<<<<<")

    config_manager = ConfigurationManager()
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