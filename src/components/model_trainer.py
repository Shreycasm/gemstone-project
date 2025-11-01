import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from src.utils.utility import load_joblib_object, save_joblib_object, load_data, load_yaml_file
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ModelTrainerArtifacts
from typing import Tuple


class ModelTrainer:

    def __init__(self, config: ModelTrainerConfig = ModelTrainerConfig()) -> None:
        self.config = config
        self.load_file = load_yaml_file(self.config.model_trainer_scheme_file_path)
        self.target = self.load_file.get("target_feature", [])

    def separate_target_feature(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        X = df.drop(columns=self.target, errors="ignore")
        y = df[self.target].values.ravel() if isinstance(self.target, list) and len(self.target) == 1 else df[self.target].values
        return X, y

    def train_model(self, X, y, model=None):
        if model is None:
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
        model.fit(X, y)
        return model

    def initiate_model_trainer(self, train_data_path: Path, preprocessor_path: Path) -> ModelTrainerArtifacts:
        preprocessor = load_joblib_object(preprocessor_path)
        df = load_data(train_data_path)
        X, y = self.separate_target_feature(df)
        X_transformed = preprocessor.transform(X)

        model = self.train_model(X_transformed, y)
        save_joblib_object(self.config.model_path, model)

        return ModelTrainerArtifacts(model_path=self.config.model_path)
