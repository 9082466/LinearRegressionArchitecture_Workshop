from DataExtractionAnalysis.data_extraction import DataExtractionAnalysis
from DataPreparation.data_preparation import DataPreparation
from ModelSelection.model_selection import ModelSelection
from ModelTraining.model_training import ModelTraining
from ModelEvaluationValidation.model_ev_validation import ModelEvaluationValidation
from TrainedMLModel.trained_ml_model import TrainedMLModel

def main():
    # Step 1: Data extraction
    data_handler = DataExtractionAnalysis(data_path="./data/training_data.csv")
    df = data_handler.load_data()
    data_handler.basic_summary()

    # Step 2: Preparation
    prep = DataPreparation()
    df_clean = prep.clean(df)
    df_std, df_norm = prep.transform(df_clean, features=["axis1", "axis2", "axis3"])
    X_train, X_test, y_train, y_test = prep.split(df_std, target="axis1")

    # Step 3: Model selection
    model_selector = ModelSelection(model_type="linear")
    model = model_selector.get_model()

    # Step 4: Training
    trainer = ModelTraining(model)
    trainer.fit(X_train, y_train)

    # Step 5: Evaluation
    evaluator = ModelEvaluationValidation()
    y_pred = model.predict(X_test)
    metrics = evaluator.evaluate(y_test, y_pred)
    print("Metrics:", metrics)

    # Step 6: Thresholds
    residuals = trainer.compute_residuals(X_train, y_train)
    thresholds = evaluator.compute_thresholds(residuals)
    print("Thresholds:", thresholds)

    # Step 7: Save model
    saved_model = TrainedMLModel(model)
    saved_model.save("linear_model.pkl")

if __name__ == "__main__":
    main()
