import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from src.utils.utility import load_data, save_joblib_object, load_yaml_file
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifacts
from src.logger import get_logger
from src.exception import SpaceshipTitanicException

logger = get_logger(__name__)

class DataTransformation:

    def __init__(self, config: DataTransformationConfig = DataTransformationConfig()):
        self.config = config
        self.yaml_file = load_yaml_file(self.config.data_transformation_scheme_file_path)
        self.drop_columns = self.yaml_file.get("drop_columns", [])
        self.numeric_columns = self.yaml_file.get("numeric_features", [])
        self.categorical_columns = self.yaml_file.get("categorical_features", [])
        self.target = self.yaml_file.get("target_feature", [])

    def change_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Dropping unwanted columns.")
            if self.drop_columns:
                df = df.drop(columns=self.drop_columns, errors="ignore")
            logger.info("Unwanted columns dropped.")

            logger.info("Changing The dtypes of columns to save memoory.")
            dtypes_map = self.yaml_file.get("change_dtypes", {})
            for col, dtype in dtypes_map.items():
                if col in df.columns:
                    df[col] = df[col].astype(dtype)
            logger.info("Saved memory and space by changing the dtypes of columns.")

            return df
        
        except Exception as e:
            raise SpaceshipTitanicException(e,sys)

    def _clip_outliers_array(self, X: np.ndarray) -> np.ndarray:
        try:
            logger.info("Clipping Outliers to 1% and 99%")
            lower = np.quantile(X, 0.01, axis=0)
            upper = np.quantile(X, 0.99, axis=0)
            logger.info("Outliers clipped successfully.")
            return np.clip(X, lower, upper)
        
        except Exception as e:
            raise SpaceshipTitanicException(e,sys)

    def preprocessor(self) -> ColumnTransformer:
        try:
            logger.info("Preprocessing the data.")
            numeric_transform = Pipeline([
                ("clipper", FunctionTransformer(self._clip_outliers_array, validate=False))
            ])

            logger.info("OneHot Encoding the categorical columns.")
            categorical_transform = OneHotEncoder(handle_unknown="ignore")
            logger.info("OneHot Encoded the categorical columns.")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transform, self.numeric_columns),
                    ("cat", categorical_transform, self.categorical_columns)
                ],
                remainder="drop",
                verbose_feature_names_out=False
            )
            
            logger.info("Data preprocessing Completed.")

            return preprocessor
        
        except Exception as e:
            raise SpaceshipTitanicException(e,sys)

    def initiate_data_transformation(self, train_data_path: Path) -> DataTransformationArtifacts:
        try:
            logger.info("Data Transformation Component Started...")
            df = load_data(train_data_path)
            df = self.change_dtypes(df)
            features = list(self.numeric_columns) + list(self.categorical_columns)
            features = [c for c in features if c in df.columns]

            if not features:
                raise ValueError("No feature columns found for preprocessor fit.")

            X_for_fit = df[features]

            preprocessor_obj = self.preprocessor()
            preprocessor_obj.fit(X_for_fit)

            save_joblib_object(self.config.preprocessor_file_path, preprocessor_obj)
            preprocessor_artifact = DataTransformationArtifacts(preprocessor_path=self.config.preprocessor_file_path)
            logger.info(f"Preprocessor Artifact saved at {preprocessor_artifact}")

            logger.info("Data Transformation Component Completed Successfully.")

            return preprocessor_artifact

        except Exception as e:
            raise SpaceshipTitanicException(e,sys)
        
