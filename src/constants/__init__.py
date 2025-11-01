from pathlib import Path

PIPELINE_NAME: str = "GemStone"
ARTIFACT_DIR: str = "artifacts"


# <==================== Data Ingestion Constants ====================>
DATA_URL: str = "https://raw.githubusercontent.com/Shreycasm/Datasets/refs/heads/main/gemstone/train.csv"
DATA_INGESTION_DIR:str = "data_ingestion"
FEATURE_STORE_DIR: str = "feature_store"
RAW_DATA_FILE_NAME: str = "raw_data.parquet"
INGESTED_DATA_FILE_NAME:str ="ingested_data"
TRAIN_DATA_FILE_NAME:str = "train.parquet"
TEST_DATA_FILE_NAME:str = "test.parquet"
TEST_SIZE:float = 0.3
SAMPLE_DATA_DIR: str = "sample_data"
SAMPLE_DATA_FILE_NAME: str = "data.csv"
SAMPLE_DATA_SIZE: float = 0.2


# <==================== Data Transformation Constants ====================>
DATA_TRANSFORMATION_DIR: str = "data_transformation"
PREPROCESSOR_FILE_NAME:str = "preprocessor.pkl"
SCHEME_FILE_PATH: str = Path("config/scheme.yaml")


# <==================== Model Trainer Constants ====================>
MODEL_TRAINER_DIR: str = "model_trainer"
MODEL_NAME: str =  "random_forest_model.joblib"


# <==================== Model Evaluation Constants ====================>
MODEL_EVALUATION: str = "model_evaluation"
MODEL_EVALUATION_REPORT_NAME: str = "evaluation_report.json"
