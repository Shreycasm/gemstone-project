from src.components.data_ingestion import load_data, save_data, train_test_split_data 
from src.components.data_transformation import clip_outliers, change_dtypes, preprocessor, save_object, load_data_parquet
from src.components.model_trainer import model_trainer, load_preprocessor, seperate_target_feature, load_data_parquet
from src.components.moel_eval import load_model, eval_model, save_report, load_data_parquet

from pathlib import Path
import pandas as pd


if __name__ == "__main__":
    new_data = pd.DataFrame([{
    "carat": 0.8,
    "depth": 61.5,
    "table": 55.0,
    "x": 5.9,
    "y": 5.8,
    "z": 3.6,
    "cut": "Ideal",
    "color": "G",
    "clarity": "VS2"
}])
    preprocessor = load_preprocessor(file_path=Path("artifacts/data_transformation/preprocessor.pkl"))
    
    X_processed = preprocessor.transform(new_data)
    model = load_model(Path("artifacts/models/random_forest_model.joblib"))
    predictions = model.predict(X_processed)
    print(predictions)