import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import numpy as np
from src.components.data_transformation import clip_outliers


def load_data_parquet(file_path: str) -> pd.DataFrame:
    df = pd.read_parquet(file_path)
    return df

def load_preprocessor(file_path: Path) -> object:

    preprocessor = joblib.load(file_path)

    return preprocessor

def load_model(file_path: Path) -> object:

    model = joblib.load(file_path)

    return model

def seperate_target_feature(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:

    X = df.drop(columns="price")
    y = df["price"]

    return X, y

def eval_model(model: object, X: pd.DataFrame, y: pd.Series) -> dict:

    predictions = model.predict(X)

    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }

def save_report(file_path: Path, report: dict) -> None:
    file_dir = file_path.parent
    file_dir.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as f:
        for key, value in report.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    df = load_data_parquet(file_path=Path("artifacts/data_ingestion/ingested_data/test.parquet"))
    preprocessor = load_preprocessor(file_path=Path("artifacts/data_transformation/preprocessor.pkl"))
    X, y = seperate_target_feature(df)
    X_processed = preprocessor.transform(X)
    model = load_model(Path("artifacts/models/random_forest_model.joblib"))
    report = eval_model(model, X_processed, y)
    save_report(Path("artifacts/model_evaluation/evaluation_report.json"), report)