from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionArtifacts:
    train_data_path: Path
    test_data_path: Path

@dataclass
class DataTransformationArtifacts:
    preprocessor_path: Path

@dataclass
class ModelTrainerArtifacts:
    model_path: Path

@dataclass
class ModelEvalArtifacts:
    eval_file_path: Path
