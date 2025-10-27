from src.components.data_ingestion import DataIngestion, DataIngestionConfig, DataIngestionArtifacts
from src.components.data_transformation import DataTransformation, DataTransformationConfig, DataTransformationArtifacts
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig, ModelTrainerArtifacts
from src.components.moel_eval import ModelEvalConfig, ModelEval, ModelEvalArtifacts
from pathlib import Path
from dataclasses import dataclass
from typing import List

@dataclass
class TrainingPipelineArtifacts:
    data_ingestion_artifacts: DataIngestionArtifacts
    data_transformation_artifacts: DataTransformationArtifacts 
    model_trainer_artifacts: ModelTrainerArtifacts
    model_eval_artifacts: ModelEvalArtifacts

class TrainingPipeline:


    def initiate_traning_pipeline(self):

        data_ingestion_obj = DataIngestion(DataIngestionConfig)
        data_ingestion_artifacts = data_ingestion_obj.initiate_data_ingestion()
        
        data_transformation_obj = DataTransformation(DataTransformationConfig)
        data_transformation_artifacts = data_transformation_obj.initiate_data_transformation(
            data_ingestion_artifacts.train_data_path
        )
        
        model_trainer_obj = ModelTrainer(ModelTrainerConfig)
        model_trainer_artifacts = model_trainer_obj.initiate_model_trainer(
            data_ingestion_artifacts.train_data_path,data_transformation_artifacts.preprocessor_path
        )

        model_eval_obj = ModelEval(ModelEvalConfig)
        model_eval_artifacts = model_eval_obj.initiate_model_eval(data_ingestion_artifacts.test_data_path,
                                                                data_transformation_artifacts.preprocessor_path,
                                                                model_trainer_artifacts.model_path)
        
        return (
            TrainingPipelineArtifacts
        )
        
if __name__ == "__main__":
    training_pipeline = TrainingPipeline()
    training_pipeline.initiate_traning_pipeline()