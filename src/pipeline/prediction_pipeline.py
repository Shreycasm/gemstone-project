from src.utils.utility import load_joblib_object
from pathlib import Path
import pandas as pd


class PredictionPipeline:

    def initiate_predction_pipeline(self, new_data: pd.DataFrame) -> pd.Series:

        preprocessor = load_joblib_object(file_path=Path("artifacts/data_transformation/preprocessor.pkl"))
        
        X_processed = preprocessor.transform(new_data)
        model = load_joblib_object(Path("artifacts/models/random_forest_model.joblib"))
        predictions = model.predict(X_processed)
        print(predictions)


if __name__ == "__main__":
    new_data = pd.DataFrame({
        "carat": [0.8, 1.0, 1.5],
        "depth": [61.5, 62.0, 63.0],
        "table": [55.0, 57.0, 59.0],
        "x": [5.9, 6.5, 7.4],
        "y": [5.8, 6.4, 7.3],
        "z": [3.6, 4.0, 4.7],
        "cut": ["Ideal", "Premium", "Good"],
        "color": ["G", "E", "D"],
        "clarity": ["VS2", "VVS1", "SI1"]
    })
    obj = PredictionPipeline()
    obj.initiate_predction_pipeline(new_data)   
 