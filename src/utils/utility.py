import pandas as pd
from pathlib import Path
import joblib
import json


def load_csv_data(file_path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

def save_data_to_parquet(df: pd.DataFrame, file_path: Path)-> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_path, index= False, engine="pyarrow", compression="snappy")

def load_data_parquet(file_path: Path)-> pd.DataFrame:
    df = pd.read_parquet(file_path)
    return df

def save_joblib_object(file_path: Path, obj: object) -> None:
    file_dir = file_path.parent
    file_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, file_path)

def load_joblib_object(file_path: Path) -> object:
    with open(file_path, "rb") as obj:
        return joblib.load(obj)
    
def save_json_file(file_path:str, data:dict) -> None:
    file_dir = Path(file_path).parent
    file_dir.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)