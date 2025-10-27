import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import numpy as np
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.data_ingestion import DataIngestionArtifacts  
from dataclasses import dataclass 
from src.utils.utility import load_data_parquet, load_joblib_object, save_json_file


@dataclass
class ModelEvalConfig:
    model_eval_dir:Path = Path("artifacts/model_evaluation")
    eval_file_name:str = "evaluation_report.json" 
    eval_file_path:Path = model_eval_dir / eval_file_name

@dataclass
class ModelEvalArtifacts:
    evel_file_path: Path


class ModelEval:
    
    def __init__(self, config: ModelEvalConfig):
        self.config = config

    def seperate_target_feature(self,df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:

        X = df.drop(columns="price")
        y = df["price"]

        return X, y

    def eval_model(self,model: object, X: pd.DataFrame, y: pd.Series) -> dict:

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

    def initiate_model_eval(self, test_data_path:Path, preprocessor_path:Path,
                            model_path:Path) -> ModelEvalArtifacts:
        df = load_data_parquet(test_data_path)
        preprocessor = load_joblib_object(preprocessor_path)
        model = load_joblib_object(model_path)

        X, y = self.seperate_target_feature(df)
        X = preprocessor.transform(X)
        report = self.eval_model(model, X, y)

        save_json_file(self.config.eval_file_path, report)

        return ModelEvalArtifacts(evel_file_path=self.config.eval_file_path)
