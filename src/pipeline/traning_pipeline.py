from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_eval import ModelEval


class TrainingPipeline:

    def initiate_training_pipeline(self):

        data_ingestion_obj = DataIngestion()
        data_ingestion_artifacts = data_ingestion_obj.initiate_data_ingestion()

        data_transformation_obj = DataTransformation()
        data_transformation_artifacts = data_transformation_obj.initiate_data_transformation(
            data_ingestion_artifacts.train_data_path
        )

        model_trainer_obj = ModelTrainer()
        model_trainer_artifacts = model_trainer_obj.initiate_model_trainer(
            data_ingestion_artifacts.train_data_path, data_transformation_artifacts.preprocessor_path
        )

        model_eval_obj = ModelEval()
        model_eval_artifacts = model_eval_obj.initiate_model_eval(
            data_ingestion_artifacts.test_data_path,
            data_transformation_artifacts.preprocessor_path,
            model_trainer_artifacts.model_path
        )


if __name__ == "__main__":
    training_pipeline = TrainingPipeline()
    training_pipeline.initiate_training_pipeline()
