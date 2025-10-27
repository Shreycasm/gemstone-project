import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.utils.utility import load_csv_data, save_data_to_parquet

@dataclass
class DataIngestionConfig:
    data_url: str = "https://raw.githubusercontent.com/Shreycasm/Datasets/refs/heads/main/gemstone/train.csv"
    data_ingestion_dir: Path = Path("artifacts/data_ingestion")

    feature_store_dir: Path = data_ingestion_dir / "feature_store"
    raw_data_file_name: str = "raw_data.parquet"
    raw_data_file_path: Path = feature_store_dir / raw_data_file_name

    ingested_data_dir: Path = data_ingestion_dir / "ingested_data"
    train_data_file_name: str = "train.parquet"
    test_data_file_name: str = "test.parquet"
    train_data_file_path: Path = ingested_data_dir / train_data_file_name
    test_data_file_path: Path = ingested_data_dir / test_data_file_name

    test_size: float = 0.3

@dataclass
class DataIngestionArtifacts:
    train_data_path: Path
    test_data_path: Path


class DataIngestion:

    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config


    def train_test_split_data(self,df:pd.DataFrame, split_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        return train_test_split(df, test_size= split_size, random_state= 42)
    
    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        df  = load_csv_data(self.config.data_url)
        save_data_to_parquet(df, self.config.raw_data_file_path)
        train_df, test_df = self.train_test_split_data(df, split_size= self.config.test_size)
        save_data_to_parquet(train_df, self.config.train_data_file_path)
        save_data_to_parquet(test_df, self.config.test_data_file_path)

        return DataIngestionArtifacts(
                    train_data_path=self.config.train_data_file_path,
                    test_data_path=self.config.test_data_file_path
                )   
    
        
 

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
    
