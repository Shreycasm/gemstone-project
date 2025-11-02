from pathlib import Path
from typing import Union, Optional
import os,sys
from src.utils.utility import load_data, save_data
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifacts
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.exception import SpaceshipTitanicException

logger  = get_logger(__name__)

class DataIngestion:

    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def save_data_to_feature_store(
            self, file_path: Union[str, Path], 
            save_path: Path,
            frac:float = 1.0,
            mode: str="development",
            sample_file_path: Optional[Path]= None
            ) -> None:
        
        try:
            logger.info("Fetching the data...")
            df = load_data(file_path)
            logger.info("Data Fetched Successfully.")

            if mode == "development":
                logger.info("Saving Sample 20% data for CI envoirnment.")
                df_sample = df.sample(frac=frac, random_state=42)
                save_data(df_sample, sample_file_path)
                logger.info(f"Saved Sample 20% data for CI envoirnment. Shape of data {df_sample.shape}")

            logger.info("Saving data in Feature Store.")
            save_data(df, save_path)
            logger.info(f"Data saved in Feature Store. Shape of data {df.shape}")

        except Exception as e:
            raise SpaceshipTitanicException(e,sys)

    def train_test_split_data(self, file_path: Union[str, Path], train_path: Path, test_path: Path,
                              split_size: float) -> None:
        try :
            logger.info("Splitting the raw data to train test splits.")

            df = load_data(file_path)
            train_df, test_df = train_test_split(df, test_size=split_size, random_state=42)
            
            save_data(train_df, train_path)
            logger.info(f"Training data saved. Shape of data {train_df.shape}")
            save_data(test_df, test_path)
            logger.info(f"Testing data saved. Shape of data {test_df.shape}")

            logger.info("Splitting the raw data to train test splits completed.")

        except Exception as e:
            raise SpaceshipTitanicException(e,sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        try:

            mode = os.getenv("MODE" , "development")
            logger.info(f"Working in {mode} environment.")

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

            data_ingestion_artifacts = DataIngestionArtifacts(
                train_data_path=self.config.train_data_file_path,
                test_data_path=self.config.test_data_file_path
            )


            logger.info(f"Data Ingestion Artifacts: {data_ingestion_artifacts}")

            return data_ingestion_artifacts

        
        except Exception as e:
            raise SpaceshipTitanicException(e,sys)




if __name__ == "__main__":
    from src.config.configuration import ConfigurationManager

    logger.info(">>>>>>>>>>>>>>>>> Data Ingestion Component Started. <<<<<<<<<<<<<<<<<")

    config_manager = ConfigurationManager()
    data_ingestion_config = config_manager.get_data_ingestion_config()

    obj = DataIngestion(config = data_ingestion_config)
    artifacts = obj.initiate_data_ingestion()

    logger.info(">>>>>>>>>>>>>>>>> Data Ingestion Component Completed. <<<<<<<<<<<<<<<<<")
