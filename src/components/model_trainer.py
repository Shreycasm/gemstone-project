import pandas as pd
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor
from src.components.data_transformation import clip_outliers

def load_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(file_path)

    return df

def load_preprocessor(file_path: Path) -> object:

    preprocessor = joblib.load(file_path)

    return preprocessor

def seperate_target_feature(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:

    X = df.drop(columns="price")
    y = df["price"]

    return X, y

def model_trainer(X: pd.DataFrame, y: pd.Series, algo):
    model = algo.fit(X, y)

    return model

def save_object(file_path: Path, model: object) -> None:
    file_dir = file_path.parent
    file_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, file_path)

if __name__ == "__main__":
    df = load_data(Path("artifacts/data_ingestion/ingested_data/train.parquet"))
    preprocessor = load_preprocessor(Path("artifacts/data_transformation/preprocessor.pkl"))
    X, y = seperate_target_feature(df)
    X_processed = preprocessor.transform(X)
    rfr = RandomForestRegressor()
    model = model_trainer(X_processed, y, rfr)
    save_object(Path("artifacts/models/random_forest_model.joblib"), model)