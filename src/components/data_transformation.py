import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from pathlib import Path
import pandas as pd

def load_data_parquet(file_path: Path)-> pd.DataFrame:
    df = pd.read_parquet(file_path)
    return df

def change_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    return (
         df
        .drop(columns = "id")
        .assign(
            carat = lambda x : x.carat.astype(np.float32),
            depth = lambda x : x.depth.astype(np.float32),
            table = lambda x : x.table.astype(np.float32),
            x = lambda x : x.x.astype(np.float32),
            y = lambda x : x.y.astype(np.float32),
            z = lambda x : x.z.astype(np.float32),
            cut = lambda x : x.cut.astype("category"),
            color = lambda x : x.color.astype("category"),
            clarity = lambda x : x.clarity.astype("category"),
            price = lambda x : x.price.astype("int32")
        )
    )



numeric_features = ["carat", "depth", "table", "x", "y", "z"]
categorical_features = ["cut", "color", "clarity"]

def clip_outliers(df):
    df = df.copy()
    for col in numeric_features:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower, upper)
    return df


def preprocessor(df: pd.DataFrame) -> ColumnTransformer:

    numeric_transform = FunctionTransformer(clip_outliers)
    categorical_transform = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transform, numeric_features),
            ("cat", categorical_transform, categorical_features),
        ]
    )

    return preprocessor

def save_object(file_path: Path, obj: object) -> None:
    file_dir = file_path.parent
    file_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, file_path)


if __name__ =="__main__":
    df = load_data_parquet(Path("artifacts/data_ingestion/ingested_data/train.parquet"))
    df = change_dtypes(df)

    preprocessor_obj = preprocessor(df)
    traindf = preprocessor_obj.fit_transform(df)
    save_object(Path("artifacts/data_transformation/preprocessor.pkl"), preprocessor_obj)