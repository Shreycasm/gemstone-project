import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from src.utils.utility import load_data_parquet, save_joblib_object

@dataclass
class DataTransformationConfig:
    data_transformation_dir: Path = Path("artifacts/data_transformation")
    preprocessor_file_name: str = "preprocessor.pkl"
    preprocessor_file_path: Path = data_transformation_dir / preprocessor_file_name

    numeric_features = ["carat", "depth", "table", "x", "y", "z"]
    categorical_features = ["cut", "color", "clarity"]

@dataclass
class DataTransformationArtifacts:
    preprocessor_path: Path

class DataTransformation:

    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def change_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df
            .drop(columns="id")
            .assign(
                carat=lambda x: x.carat.astype(np.float32),
                depth=lambda x: x.depth.astype(np.float32),
                table=lambda x: x.table.astype(np.float32),
                x=lambda x: x.x.astype(np.float32),
                y=lambda x: x.y.astype(np.float32),
                z=lambda x: x.z.astype(np.float32),
                cut=lambda x: x.cut.astype("category"),
                color=lambda x: x.color.astype("category"),
                clarity=lambda x: x.clarity.astype("category"),
                price=lambda x: x.price.astype(np.int32)
            )
        )

    def clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.config.numeric_features:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower, upper)
        return df

    def preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        numeric_transform = FunctionTransformer(self.clip_outliers)
        categorical_transform = OneHotEncoder(handle_unknown="ignore")
        return ColumnTransformer(
            transformers=[
                ("num", numeric_transform, self.config.numeric_features),
                ("cat", categorical_transform, self.config.categorical_features)
            ]
        )

    def initiate_data_transformation(self, train_data_path:Path) -> DataTransformationArtifacts:
        df = load_data_parquet(train_data_path)
        df = self.change_dtypes(df)

        preprocessor_obj = self.preprocessor(df)
        preprocessor_obj.fit(df)

        save_joblib_object(self.config.preprocessor_file_path, preprocessor_obj)
        return DataTransformationArtifacts(preprocessor_path=self.config.preprocessor_file_path)
        
