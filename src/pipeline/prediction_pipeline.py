from src.utils.utility import load_joblib_object
from pathlib import Path
import pandas as pd
from src.entity.config_entity import PredictionPipelineConfig
from typing import Any
from src.logger import get_logger
from src.exception import SpaceshipTitanicException
import sys

logger = get_logger(__name__)


class PredictionPipeline:

    def __init__(self, config: PredictionPipelineConfig = PredictionPipelineConfig()):
        self.config = config

    def initiate_prediction_pipeline(self, new_data: pd.DataFrame) -> Any:
        try:
            logger.info("Prediction Pipeline started...")
            preprocessor = load_joblib_object(file_path=self.config.preprocessor_file_path)
            X_processed = preprocessor.transform(new_data)
            model = load_joblib_object(file_path=self.config.model_file_path)
            predictions = model.predict(X_processed)
            logger.info(f"The Predictions are: {predictions}")
            logger.info("prediction Pipeline Completed."
                        )        
        except Exception as e:
            raise SpaceshipTitanicException(e,sys)


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
    preds = obj.initiate_prediction_pipeline(new_data)
    print(preds)
