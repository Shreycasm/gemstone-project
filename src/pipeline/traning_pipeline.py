from src.components.data_ingestion import load_data, save_data, train_test_split_data 
from src.components.data_transformation import clip_outliers, change_dtypes, preprocessor, save_object, load_data_parquet
from src.components.model_trainer import model_trainer, load_preprocessor, seperate_target_feature, load_data_parquet
from src.components.moel_eval import load_model, eval_model, save_report, load_data_parquet

from sklearn.ensemble import RandomForestRegressor

from pathlib import Path


if __name__ == "__main__":

    df  = load_data("https://raw.githubusercontent.com/Shreycasm/Datasets/refs/heads/main/gemstone/train.csv")
    save_data(df, Path("artifacts/data_ingestion/feature_store/raw_data.parquet"))
    train_df, test_df = train_test_split_data(df, split_size= 0.3)
    save_data(train_df, Path("artifacts/data_ingestion/ingested_data/train.parquet"))
    save_data(test_df, Path("artifacts/data_ingestion/ingested_data/test.parquet"))

    df = load_data_parquet(Path("artifacts/data_ingestion/ingested_data/train.parquet"))
    df = change_dtypes(df)
    preprocessor_obj = preprocessor(df)
    traindf = preprocessor_obj.fit_transform(df)
    save_object(Path("artifacts/data_transformation/preprocessor.pkl"), preprocessor_obj)

    df = load_data_parquet(Path("artifacts/data_ingestion/ingested_data/train.parquet"))
    preprocessor = load_preprocessor(Path("artifacts/data_transformation/preprocessor.pkl"))
    X, y = seperate_target_feature(df)
    X_processed = preprocessor.transform(X)
    rfr = RandomForestRegressor()
    model = model_trainer(X_processed, y, rfr)
    save_object(Path("artifacts/models/random_forest_model.joblib"), model)

    df = load_data_parquet(file_path=Path("artifacts/data_ingestion/ingested_data/test.parquet"))
    preprocessor = load_preprocessor(file_path=Path("artifacts/data_transformation/preprocessor.pkl"))
    X, y = seperate_target_feature(df)
    X_processed = preprocessor.transform(X)
    model = load_model(Path("artifacts/models/random_forest_model.joblib"))
    report = eval_model(model, X_processed, y)
    save_report(Path("artifacts/model_evaluation/evaluation_report.json"), report)