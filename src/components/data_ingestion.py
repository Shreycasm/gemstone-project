from pathlib import Path
from typing import Union, Optional
import os
from src.utils.utility import load_data, save_data
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifacts
from sklearn.model_selection import train_test_split


class DataIngestion:

    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()) -> None:
        self.config = config

    def save_data_to_feature_store(
            self, file_path: Union[str, Path], 
            save_path: Path,
            frac:float = 1.0,
            mode: str="development",
            sample_file_path: Optional[Path]= None
            ) -> None:

        df = load_data(file_path)
        if mode == "development":
            df_sample = df.sample(frac=frac, random_state=42)
            save_data(df_sample, sample_file_path)
        save_data(df, save_path)

    def train_test_split_data(self, file_path: Union[str, Path], train_path: Path, test_path: Path,
                              split_size: float) -> None:
        df = load_data(file_path)
        train_df, test_df = train_test_split(df, test_size=split_size, random_state=42)
        save_data(train_df, train_path)
        save_data(test_df, test_path)

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:

        mode = os.getenv("MODE" , "development")
        if mode == "development":
            file_path = self.config.data_url
            sample_data_file_path = self.config.sample_data_file_path
        elif mode == "ci":
            file_path = self.config.sample_data_file_path
            sample_data_file_path = None

        self.save_data_to_feature_store(
            file_path=file_path,
            save_path=self.config.raw_data_file_path,
            frac=self.config.sample_data_size, 
            mode=mode,
            sample_file_path= sample_data_file_path
        )

        self.train_test_split_data(
            file_path=self.config.raw_data_file_path,
            train_path=self.config.train_data_file_path,
            test_path=self.config.test_data_file_path,
            split_size=self.config.test_size
        )

        return DataIngestionArtifacts(
            train_data_path=self.config.train_data_file_path,
            test_data_path=self.config.test_data_file_path
        )


if __name__ == "__main__":
    obj = DataIngestion()
    artifacts = obj.initiate_data_ingestion()
    print(artifacts)
