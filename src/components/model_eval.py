import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.utility import load_data, load_joblib_object, save_json_file, load_yaml_file
from src.entity.config_entity import ModelEvalConfig
from src.entity.artifact_entity import ModelEvalArtifacts
from typing import Tuple


class ModelEval:

    def __init__(self, config: ModelEvalConfig = ModelEvalConfig()):
        self.config = config
        self.yaml_file = load_yaml_file(self.config.model_eval_scheme_file_path)
        self.target = self.yaml_file.get("target_feature", [])

    def separate_target_feature(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        X = df.drop(columns=self.target, errors="ignore")
        y = df[self.target].values.ravel() if isinstance(self.target, list) and len(self.target) == 1 else df[self.target].values
        return X, y

    def eval_model(self, model: object, X: pd.DataFrame, y: pd.Series) -> dict:
        predictions = model.predict(X)

        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)

        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2)
        }

    def initiate_model_eval(self, test_data_path: Path, preprocessor_path: Path,
                            model_path: Path) -> ModelEvalArtifacts:
        
        df = load_data(test_data_path)
        preprocessor = load_joblib_object(preprocessor_path)
        model = load_joblib_object(model_path)
        X, y = self.separate_target_feature(df)
        X_transformed = preprocessor.transform(X)
        report = self.eval_model(model, X_transformed, y)
        save_json_file(self.config.eval_file_path, report)

        return ModelEvalArtifacts(eval_file_path=self.config.eval_file_path)
