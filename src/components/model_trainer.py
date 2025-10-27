import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from dataclasses import dataclass
from src.utils.utility import load_joblib_object, save_joblib_object, load_data_parquet

@dataclass
class ModelTrainerConfig:
    model_trainer_dir: Path = Path("artifacts/models")
    model_name:str = "random_forest_model.joblib"
    model_path: Path = model_trainer_dir / model_name

@dataclass
class ModelTrainerArtifacts:
    model_path: Path

class ModelTrainer:

    def __init__(self, config: ModelTrainerConfig) -> None:
        self.config = config
         
    def separate_target_feature(self,df: pd.DataFrame):
        X = df.drop(columns="price")
        y = df["price"]
        return X, y

    def train_model(self,X, y, model=None):
        if model is None:
            model = RandomForestRegressor(random_state=42,n_jobs=-1)
        model.fit(X, y)
        return model
    
    def initiate_model_trainer(self, train_data_path: Path, preprocessor_path: Path)-> ModelTrainerArtifacts:
        preprocessor = load_joblib_object(preprocessor_path)
        df = load_data_parquet(train_data_path)
        X, y = self.separate_target_feature(df)
        X = preprocessor.transform(X)
        model = self.train_model(X,y)  
        save_joblib_object(self.config.model_path, model) 

        return ModelTrainerArtifacts(model_path = self.config.model_path) 

