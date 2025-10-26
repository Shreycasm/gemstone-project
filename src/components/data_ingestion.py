import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_data(file_path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

def save_data(df: pd.DataFrame, file_path: Path)-> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_path, index= False, engine="pyarrow", compression="snappy")

def train_test_split_data(df:pd.DataFrame, split_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(df, test_size= split_size, random_state= 42)
 

if __name__ == "__main__":
    df  = load_data("https://raw.githubusercontent.com/Shreycasm/Datasets/refs/heads/main/gemstone/train.csv")
    save_data(df, Path("artifacts/data_ingestion/feature_store/raw_data.parquet"))
    train_df, test_df = train_test_split_data(df, split_size= 0.3)
    save_data(train_df, Path("artifacts/data_ingestion/ingested_data/train.parquet"))
    save_data(test_df, Path("artifacts/data_ingestion/ingested_data/test.parquet"))
