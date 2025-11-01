from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from src.constants import *


@dataclass
class TrainingPipeline:
    pipeline_name:str = PIPELINE_NAME
    artifact_dir:str = Path(ARTIFACT_DIR)

@dataclass
class DataIngestionConfig:
    data_url: str = DATA_URL
    data_ingestion_dir: Path = TrainingPipeline.artifact_dir / DATA_INGESTION_DIR

    feature_store_dir: Path = data_ingestion_dir / FEATURE_STORE_DIR
    raw_data_file_name: str = RAW_DATA_FILE_NAME
    raw_data_file_path: Path = feature_store_dir / raw_data_file_name

    ingested_data_dir: Path = data_ingestion_dir / INGESTED_DATA_FILE_NAME
    train_data_file_name: str = TRAIN_DATA_FILE_NAME
    test_data_file_name: str = TEST_DATA_FILE_NAME
    train_data_file_path: Path = ingested_data_dir / train_data_file_name
    test_data_file_path: Path = ingested_data_dir / test_data_file_name
    test_size: float = TEST_SIZE

    sample_data_dir: Path = Path(SAMPLE_DATA_DIR)
    sample_data_file_path: Path = sample_data_dir / SAMPLE_DATA_FILE_NAME
    sample_data_size: float = SAMPLE_DATA_SIZE
    

@dataclass
class DataTransformationConfig:
    data_transformation_dir: Path = TrainingPipeline.artifact_dir / DATA_TRANSFORMATION_DIR
    preprocessor_file_name: str = PREPROCESSOR_FILE_NAME
    preprocessor_file_path: Path = data_transformation_dir / preprocessor_file_name
    data_transformation_scheme_file_path : Path = SCHEME_FILE_PATH

@dataclass
class ModelTrainerConfig:
    model_trainer_dir: Path = TrainingPipeline.artifact_dir / MODEL_TRAINER_DIR
    model_name:str = MODEL_NAME
    model_path: Path = model_trainer_dir / model_name
    model_trainer_scheme_file_path: Path = SCHEME_FILE_PATH

@dataclass
class ModelEvalConfig:
    model_eval_dir: Path = TrainingPipeline.artifact_dir / MODEL_EVALUATION
    eval_file_name: str = MODEL_EVALUATION_REPORT_NAME 
    eval_file_path: Path = model_eval_dir / eval_file_name
    model_eval_scheme_file_path: Path = SCHEME_FILE_PATH 

@dataclass
class PredictionPipelineConfig:
    test_data_file_path: Path = DataIngestionConfig.test_data_file_path
    preprocessor_file_path: Path = DataTransformationConfig.preprocessor_file_path
    model_file_path: Path = ModelTrainerConfig.model_path

