import pandas as pd
import pytest

from src.training import ModelTrainer


def test_classification_returns_results(clf_df, tmp_path):
    trainer = ModelTrainer(tracking_uri=str(tmp_path / "mlruns"))
    results_df, model, preprocessor = trainer.train(clf_df, "bought")
    assert not results_df.empty
    assert model is not None
    assert preprocessor.task_type == "classification"


def test_regression_returns_results(reg_df, tmp_path):
    trainer = ModelTrainer(tracking_uri=str(tmp_path / "mlruns"))
    results_df, model, preprocessor = trainer.train(reg_df, "price")
    assert not results_df.empty
    assert preprocessor.task_type == "regression"


def test_best_model_set(clf_df, tmp_path):
    trainer = ModelTrainer(tracking_uri=str(tmp_path / "mlruns"))
    trainer.train(clf_df, "bought")
    assert trainer.best_model is not None
    assert trainer.best_model_name != ""


def test_classification_metrics_present(clf_df, tmp_path):
    trainer = ModelTrainer(tracking_uri=str(tmp_path / "mlruns"))
    results_df, _, _ = trainer.train(clf_df, "bought")
    assert "accuracy" in results_df.columns
    assert "f1_weighted" in results_df.columns


def test_regression_metrics_present(reg_df, tmp_path):
    trainer = ModelTrainer(tracking_uri=str(tmp_path / "mlruns"))
    results_df, _, _ = trainer.train(reg_df, "price")
    assert "r2" in results_df.columns
    assert "rmse" in results_df.columns


def test_cv_folds(clf_df, tmp_path):
    trainer = ModelTrainer(tracking_uri=str(tmp_path / "mlruns"))
    results_df, _, _ = trainer.train(clf_df, "bought", cv_folds=3)
    assert "cv_mean" in results_df.columns


def test_voting_ensemble(clf_df, tmp_path):
    trainer = ModelTrainer(tracking_uri=str(tmp_path / "mlruns"))
    results_df, _, _ = trainer.train(clf_df, "bought", use_ensemble=True)
    assert "VotingClassifier" in results_df["model"].values


def test_ordinal_encoder(clf_df, tmp_path):
    trainer = ModelTrainer(tracking_uri=str(tmp_path / "mlruns"))
    results_df, _, preprocessor = trainer.train(clf_df, "bought", cat_encoder="ordinal")
    assert not results_df.empty


def test_knn_imputer(df_with_missing, tmp_path):
    trainer = ModelTrainer(tracking_uri=str(tmp_path / "mlruns"))
    results_df, _, _ = trainer.train(df_with_missing, "target", num_imputer="knn")
    assert not results_df.empty


def test_calibration(clf_df, tmp_path):
    trainer = ModelTrainer(tracking_uri=str(tmp_path / "mlruns"))
    trainer.train(clf_df, "bought", use_calibration=True, calibration_method="sigmoid")
    assert trainer.calibrated_model is not None


def test_save(clf_df, tmp_path):
    trainer = ModelTrainer(tracking_uri=str(tmp_path / "mlruns"))
    trainer.train(clf_df, "bought")
    trainer.save(str(tmp_path))
    assert (tmp_path / "best_model.pkl").exists()
    assert (tmp_path / "preprocessor.pkl").exists()
    assert (tmp_path / "best_model_original.pkl").exists()


def test_evaluation_artifacts_classification(clf_df, tmp_path):
    trainer = ModelTrainer(tracking_uri=str(tmp_path / "mlruns"))
    trainer.train(clf_df, "bought")
    artifacts = trainer.get_evaluation_artifacts()
    assert "confusion_matrix" in artifacts
    assert "classes" in artifacts


def test_evaluation_artifacts_regression(reg_df, tmp_path):
    trainer = ModelTrainer(tracking_uri=str(tmp_path / "mlruns"))
    trainer.train(reg_df, "price")
    artifacts = trainer.get_evaluation_artifacts()
    assert "residuals" in artifacts
    assert "y_test" in artifacts
