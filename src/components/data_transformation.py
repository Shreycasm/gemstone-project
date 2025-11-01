import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from src.utils.utility import load_data, save_joblib_object, load_yaml_file
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifacts


class DataTransformation:

    def __init__(self, config: DataTransformationConfig = DataTransformationConfig()):
        self.config = config
        self.yaml_file = load_yaml_file(self.config.data_transformation_scheme_file_path)
        self.drop_columns = self.yaml_file.get("drop_columns", [])
        self.numeric_columns = self.yaml_file.get("numeric_features", [])
        self.categorical_columns = self.yaml_file.get("categorical_features", [])
        self.target = self.yaml_file.get("target_feature", [])

    def change_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.drop_columns:
            df = df.drop(columns=self.drop_columns, errors="ignore")
        dtypes_map = self.yaml_file.get("change_dtypes", {})
        for col, dtype in dtypes_map.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        return df

    def _clip_outliers_array(self, X: np.ndarray) -> np.ndarray:
        lower = np.quantile(X, 0.01, axis=0)
        upper = np.quantile(X, 0.99, axis=0)
        return np.clip(X, lower, upper)

    def preprocessor(self) -> ColumnTransformer:
        numeric_transform = Pipeline([
            ("clipper", FunctionTransformer(self._clip_outliers_array, validate=False))
        ])

        categorical_transform = OneHotEncoder(handle_unknown="ignore")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transform, self.numeric_columns),
                ("cat", categorical_transform, self.categorical_columns)
            ],
            remainder="drop",
            verbose_feature_names_out=False
        )

        return preprocessor

    def initiate_data_transformation(self, train_data_path: Path) -> DataTransformationArtifacts:
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

        return DataTransformationArtifacts(preprocessor_path=self.config.preprocessor_file_path)
