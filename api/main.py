import io
import json
import logging
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.drift import DataDriftDetector
from src.prediction import ModelPredictor
from src.report import generate_pdf_report
from src.training import ModelTrainer
from src.utils import setup_logging

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Supervised ML API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_predictor: ModelPredictor | None = None


def _read_csv(file: UploadFile) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file.file.read()))


def _get_predictor() -> ModelPredictor:
    """Lazy-load the predictor to avoid reloading the model on every request."""
    global _predictor
    if _predictor is None:
        try:
            _predictor = ModelPredictor()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    return _predictor


# Status

@app.get("/", tags=["Status"])
def root():
    return {"message": "ML API is running", "docs": "/docs"}

@app.get("/health", tags=["Status"])
def health():
    return {"status": "ok"}


# Training

@app.post("/train", tags=["Training"])
async def train(
    file: UploadFile = File(...),
    target_col: str = Query(...),
    test_size: float = Query(0.2, ge=0.05, le=0.5),
    experiment_name: str = Query("ml_experiment"),
    cv_folds: int = Query(0, ge=0, le=10),
    use_tuning: bool = Query(False),
    search_strategy: str = Query("random"),
    n_iter: int = Query(20, ge=5, le=100),
    use_ensemble: bool = Query(False),
    use_feature_selection: bool = Query(False),
    use_calibration: bool = Query(False),
    calibration_method: str = Query("sigmoid"),
    num_imputer: str = Query("median"),
    cat_imputer: str = Query("most_frequent"),
    cat_encoder: str = Query("onehot"),
):
    """
    Train multiple ML models and select the best one.
    
    Handles both classification and regression tasks with configurable preprocessing,
    hyperparameter tuning, feature selection, and model calibration.
    Generates a PDF report and saves reference statistics for drift detection.
    """
    df = _read_csv(file)
    if target_col not in df.columns:
        raise HTTPException(status_code=422, detail=f"Column '{target_col}' not found. Available: {df.columns.tolist()}")

    try:
        trainer = ModelTrainer(experiment_name=experiment_name, tracking_uri=MLFLOW_TRACKING_URI)
        results_df, best_model, preprocessor = trainer.train(
            df, target_col,
            test_size=test_size, cv_folds=cv_folds,
            use_tuning=use_tuning, search_strategy=search_strategy, n_iter=n_iter,
            use_ensemble=use_ensemble,
            use_feature_selection=use_feature_selection,
            use_calibration=use_calibration, calibration_method=calibration_method,
            num_imputer=num_imputer, cat_imputer=cat_imputer, cat_encoder=cat_encoder,
        )
        trainer.save()

        # Save reference stats for drift detection
        detector = DataDriftDetector()
        stats_ref = detector.compute_reference_stats(df, target_col)
        detector.save_reference(stats_ref)
        # Keep full samples for drift detection (includes values_sample)
        with open("models/training_stats_full.json", "w") as f:
            clean = {"numerical": {}, "categorical": stats_ref.get("categorical", {})}
            for col, s in stats_ref.get("numerical", {}).items():
                clean["numerical"][col] = s
            json.dump(clean, f)

        global _predictor
        _predictor = ModelPredictor()
        evaluation = trainer.get_evaluation_artifacts()

        # Generate PDF report
        dataset_info = {
            "Rows": df.shape[0],
            "Columns": df.shape[1],
            "Missing values": int(df.isnull().sum().sum()),
            "Target column": target_col,
            "Feature selection": str(use_feature_selection),
            "Ensembles": str(use_ensemble),
        }
        generate_pdf_report(dataset_info, results_df, type(best_model).__name__, preprocessor.task_type, evaluation)

        return {
            "status": "success",
            "task_type": preprocessor.task_type,
            "best_model": type(best_model).__name__,
            "results": results_df.to_dict(orient="records"),
            "evaluation": evaluation,
        }
    except Exception as exc:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=str(exc))


# Adversarial Validation

@app.post("/adversarial_validation", tags=["Analysis"])
async def adversarial_validation(
    file: UploadFile = File(...),
    target_col: str = Query(...),
    test_size: float = Query(0.2, ge=0.05, le=0.5),
    random_state: int = Query(42),
):
    """Check whether the train/test split is well distributed.
    Trains RF and XGB to distinguish train vs test samples.
    AUC ~ 0.5 means a good split. 
    AUC >> 0.5 means distribution shift or data leakage.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    try:
        df = _read_csv(file)
        if target_col not in df.columns:
            raise HTTPException(status_code=422, detail=f"Column '{target_col}' not found.")

        from src.preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor()
        X, _ = preprocessor.fit_transform(df, target_col)

        # Create adversarial dataset: 0 = train, 1 = test
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
        y_adv = np.array([0] * len(X_train) + [1] * len(X_test))
        X_adv = np.vstack([X_train, X_test])

        X_tr, X_val, y_tr, y_val = train_test_split(X_adv, y_adv, test_size=0.3, random_state=random_state)

        results = {}
        feature_names = preprocessor.feature_names_out

        for name, model in [
            ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)),
            ("XGBoost", XGBClassifier(n_estimators=100, random_state=random_state, eval_metric="logloss", verbosity=0)),
        ]:
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_val)[:, 1]
            auc = round(float(roc_auc_score(y_val, proba)), 4)

            importances = model.feature_importances_.tolist()
            sorted_idx = np.argsort(importances)[::-1][:10]
            top_features = [
                {"feature": feature_names[i], "importance": round(importances[i], 4)}
                for i in sorted_idx
            ]

            status = "ok" if auc < 0.6 else ("warning" if auc < 0.75 else "alert")
            results[name] = {"auc": auc, "status": status, "top_features": top_features}

        overall_auc = round(float(np.mean([r["auc"] for r in results.values()])), 4)
        overall_status = "ok" if overall_auc < 0.6 else ("warning" if overall_auc < 0.75 else "alert")
        interpretation = (
            "The split looks good — models cannot distinguish train from test."
            if overall_status == "ok"
            else "Warning: train and test appear to come from different distributions. Check the split or look for data leakage."
            if overall_status == "alert"
            else "Slight difference between train and test. Worth monitoring."
        )

        return {
            "status": "success",
            "overall_auc": overall_auc,
            "overall_status": overall_status,
            "interpretation": interpretation,
            "models": results,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# Inference

@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    predictor = _get_predictor()
    try:
        df = _read_csv(file)
        predictions = predictor.predict(df).tolist()
        return {"status": "success", "task_type": predictor.task_type, "n_samples": len(predictions), "predictions": predictions}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict_proba", tags=["Inference"])
async def predict_proba(file: UploadFile = File(...)):
    predictor = _get_predictor()
    try:
        df = _read_csv(file)
        return {"status": "success", "classes": predictor.classes, "probabilities": predictor.predict_proba(df).tolist()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))



@app.get("/model_info", tags=["Inference"])
def model_info():
    predictor = _get_predictor()
    return {
        "model_type": type(predictor.model).__name__,
        "task_type": predictor.task_type,
        "n_features": len(predictor.preprocessor.feature_names_out),
        "feature_names": predictor.preprocessor.feature_names_out,
        "classes": predictor.classes,
    }


# Explainability

@app.get("/feature_importance", tags=["Explainability"])
def feature_importance():
    predictor = _get_predictor()
    model = predictor.model
    feature_names = predictor.preprocessor.feature_names_out

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_.tolist()
        importance_type = "impurity (Gini)"
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = np.abs(coef[0] if coef.ndim > 1 else coef).tolist()
        importance_type = "coefficient absolu"
    else:
        raise HTTPException(status_code=400, detail="This model does not expose native feature importances.")

    sorted_idx = np.argsort(importances)[::-1]
    return {
        "status": "success",
        "importance_type": importance_type,
        "model_type": type(model).__name__,
        "feature_names": [feature_names[i] for i in sorted_idx],
        "importances": [round(importances[i], 6) for i in sorted_idx],
    }


@app.post("/permutation_importance", tags=["Explainability"])
async def permutation_importance(
    file: UploadFile = File(...),
    target_col: str = Query(...),
    n_repeats: int = Query(10, ge=3, le=30),
):
    from sklearn.inspection import permutation_importance as sk_pi
    predictor = _get_predictor()
    try:
        df = _read_csv(file)
        if target_col not in df.columns:
            raise HTTPException(status_code=422, detail=f"Column '{target_col}' not found.")
        y = df[target_col].copy()
        X = predictor.preprocessor.transform(df)
        if predictor.preprocessor.label_encoder is not None and y.dtype == object:
            y = predictor.preprocessor.label_encoder.transform(y)
        result = sk_pi(predictor.model, X, np.array(y), n_repeats=n_repeats, random_state=42, n_jobs=-1)
        sorted_idx = np.argsort(result.importances_mean)[::-1]
        feature_names = predictor.preprocessor.feature_names_out
        return {
            "status": "success",
            "model_type": type(predictor.model).__name__,
            "n_repeats": n_repeats,
            "feature_names": [feature_names[i] for i in sorted_idx],
            "importances_mean": [round(float(result.importances_mean[i]), 6) for i in sorted_idx],
            "importances_std": [round(float(result.importances_std[i]), 6) for i in sorted_idx],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/shap", tags=["Explainability"])
async def compute_shap(
    file: UploadFile = File(...),
    target_col: str = Query(...),
    max_samples: int = Query(100, ge=10, le=500),
):
    from sklearn.inspection import permutation_importance as sk_pi

    predictor = _get_predictor()
    df = _read_csv(file)
    X = predictor.preprocessor.transform(df)[:max_samples]
    feature_names = predictor.preprocessor.feature_names_out
    model = predictor.model
    model_type = type(model).__name__

    _TREE_MODELS = ("RandomForestClassifier", "RandomForestRegressor", "XGBClassifier", "XGBRegressor", "LGBMClassifier", "LGBMRegressor")

    if model_type in _TREE_MODELS:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_array = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:
                shap_array = np.abs(shap_values)
            if shap_array.ndim == 3:
                shap_array = shap_array.mean(axis=2)
            mean_vals = shap_array.mean(axis=0).tolist()
            sorted_idx = np.argsort(mean_vals)[::-1]
            return {
                "status": "success",
                "method": "shap",
                "model_type": model_type,
                "feature_names": [feature_names[i] for i in sorted_idx],
                "importances": [round(mean_vals[i], 6) for i in sorted_idx],
                "n_samples": len(X),
            }
        except Exception:
            pass

    # Fallback: permutation importance
    y = df[target_col].copy() if target_col in df.columns else None
    if y is None:
        raise HTTPException(status_code=422, detail=f"Column '{target_col}' not found for permutation importance fallback.")
    if predictor.preprocessor.label_encoder is not None and y.dtype == object:
        y = predictor.preprocessor.label_encoder.transform(y)
    result = sk_pi(model, X, np.array(y)[:max_samples], n_repeats=5, random_state=42, n_jobs=-1)
    sorted_idx = np.argsort(result.importances_mean)[::-1]
    return {
        "status": "success",
        "method": "permutation_importance",
        "model_type": model_type,
        "feature_names": [feature_names[i] for i in sorted_idx],
        "importances": [round(float(result.importances_mean[i]), 6) for i in sorted_idx],
        "n_samples": len(X),
    }


# Analysis

@app.post("/learning_curves", tags=["Analysis"])
async def learning_curves(
    file: UploadFile = File(...),
    target_col: str = Query(...),
    cv: int = Query(5, ge=2, le=10),
    n_points: int = Query(5, ge=3, le=10),
):
    """Learning curves: score vs training set size."""
    from sklearn.model_selection import learning_curve
    predictor = _get_predictor()
    try:
        df = _read_csv(file)
        if target_col not in df.columns:
            raise HTTPException(status_code=422, detail=f"Column '{target_col}' not found.")
        y = df[target_col].copy()
        X = predictor.preprocessor.transform(df)
        if predictor.preprocessor.label_encoder is not None and y.dtype == object:
            y = predictor.preprocessor.label_encoder.transform(y)
        task_type = predictor.task_type
        scoring = "f1_weighted" if task_type == "classification" else "r2"
        train_sizes_rel = np.linspace(0.1, 1.0, n_points)
        train_sizes, train_scores, test_scores = learning_curve(
            predictor.model, X, np.array(y),
            train_sizes=train_sizes_rel, cv=cv, scoring=scoring, n_jobs=-1,
        )
        return {
            "status": "success",
            "scoring": scoring,
            "train_sizes": train_sizes.tolist(),
            "train_mean": np.mean(train_scores, axis=1).round(4).tolist(),
            "train_std": np.std(train_scores, axis=1).round(4).tolist(),
            "test_mean": np.mean(test_scores, axis=1).round(4).tolist(),
            "test_std": np.std(test_scores, axis=1).round(4).tolist(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/calibration", tags=["Analysis"])
async def calibration_curve(
    file: UploadFile = File(...),
    target_col: str = Query(...),
    n_bins: int = Query(10, ge=5, le=20),
):
    """
    Analyze probability calibration (binary classification only).
    
    Well-calibrated probabilities mean that if the model predicts 70% confidence,
    approximately 70% of those instances are actually positive.
    Returns curves before and after calibration if calibrated model exists.
    """
    from sklearn.calibration import calibration_curve as sk_cal
    predictor = _get_predictor()
    if predictor.task_type != "classification":
        raise HTTPException(status_code=400, detail="La calibration est disponible uniquement en classification.")
    try:
        df = _read_csv(file)
        y = df[target_col].copy()
        X = predictor.preprocessor.transform(df)
        if predictor.preprocessor.label_encoder is not None and y.dtype == object:
            y = predictor.preprocessor.label_encoder.transform(y)
        if not hasattr(predictor.model, "predict_proba"):
            raise HTTPException(status_code=400, detail="This model does not support predict_proba.")

        classes = predictor.classes or []
        if len(classes) != 2:
            raise HTTPException(status_code=400, detail="Calibration curve is only available for binary classification.")

        proba_before = predictor.model.predict_proba(X)[:, 1]
        frac_before, mean_before = sk_cal(y, proba_before, n_bins=n_bins, strategy="uniform")

        result = {
            "status": "success",
            "before": {
                "fraction_of_positives": frac_before.round(4).tolist(),
                "mean_predicted_value": mean_before.round(4).tolist(),
            },
            "after": None,
            "calibrated": False,
        }

        calib_path = Path("models/best_model_calibrated.pkl")
        if calib_path.exists():
            import joblib as jl
            calibrated_model = jl.load(calib_path)
            proba_after = calibrated_model.predict_proba(X)[:, 1]
            frac_after, mean_after = sk_cal(y, proba_after, n_bins=n_bins, strategy="uniform")
            result["after"] = {
                "fraction_of_positives": frac_after.round(4).tolist(),
                "mean_predicted_value": mean_after.round(4).tolist(),
            }
            result["calibrated"] = True

        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/pdp", tags=["Analysis"])
async def partial_dependence_plot(
    file: UploadFile = File(...),
    feature: str = Query(..., description="Nom de la feature originale"),
):
    """
    Compute Partial Dependence Plot showing marginal effect of a feature.
    
    Shows how predictions change as a feature varies while holding others constant.
    """
    from sklearn.inspection import partial_dependence
    predictor = _get_predictor()
    try:
        df = _read_csv(file)
        X = predictor.preprocessor.transform(df)
        feature_names = predictor.preprocessor.feature_names_out

        matching = [i for i, n in enumerate(feature_names) if feature in n]
        if not matching:
            raise HTTPException(status_code=404, detail=f"Feature '{feature}' not found. Available: {feature_names}")

        feat_idx = matching[0]
        result = partial_dependence(predictor.model, X, features=[feat_idx], kind="average")
        return {
            "status": "success",
            "feature": feature_names[feat_idx],
            "grid_values": result["grid_values"][0].round(4).tolist(),
            "pdp_values": result["average"][0].round(4).tolist(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# Drift

@app.post("/drift", tags=["Drift"])
async def detect_drift(
    file: UploadFile = File(...),
    target_col: str = Query(None),
):
    """
    Detect data drift between new data and training data.
    
    Uses statistical tests (KS-test for numerical, chi-square for categorical)
    to identify distribution shifts that may degrade model performance.
    """
    stats_path = "models/training_stats_full.json"
    if not Path(stats_path).exists():
        raise HTTPException(status_code=400, detail="No reference stats found. Train a model first.")
    try:
        with open(stats_path) as f:
            stats_ref = json.load(f)
        df_new = _read_csv(file)
        detector = DataDriftDetector()
        results = detector.detect(df_new, stats_ref, target_col)
        return {"status": "success", **results}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# Model switch

@app.post("/model/activate", tags=["Inference"])
def activate_model(use_calibrated: bool = Query(...)):
    """Switch between the original and calibrated model as the active model."""
    import shutil
    global _predictor
    src = Path("models/best_model_calibrated.pkl" if use_calibrated else "models/best_model_original.pkl")
    if not src.exists():
        label = "calibrated" if use_calibrated else "original"
        raise HTTPException(status_code=404, detail=f"{label.capitalize()} model not found. Train with calibration enabled first.")
    shutil.copy(src, "models/best_model.pkl")
    _predictor = ModelPredictor()
    return {"status": "success", "active_model": "calibrated" if use_calibrated else "original"}


# Download

@app.get("/download/model", tags=["Download"])
def download_model():
    path = Path("models/best_model.pkl")
    if not path.exists():
        raise HTTPException(status_code=404, detail="No trained model found.")
    return FileResponse(path, filename="best_model.pkl", media_type="application/octet-stream")


@app.get("/download/preprocessor", tags=["Download"])
def download_preprocessor():
    path = Path("models/preprocessor.pkl")
    if not path.exists():
        raise HTTPException(status_code=404, detail="No preprocessor found.")
    return FileResponse(path, filename="preprocessor.pkl", media_type="application/octet-stream")


@app.get("/download/report", tags=["Download"])
def download_report():
    path = Path("models/report.pdf")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Aucun rapport disponible. Lancez /train d'abord.")
    return FileResponse(path, filename="rapport_ml.pdf", media_type="application/pdf")


# MLflow

@app.get("/runs", tags=["MLflow"])
def list_runs(experiment_name: str = "ml_experiment"):
    """List all MLflow runs for an experiment."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return {"runs": []}
        runs = client.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"])
        return {
            "experiment": experiment_name,
            "runs": [{"run_id": r.info.run_id, "model": r.data.params.get("model"), "task_type": r.data.params.get("task_type"), **r.data.metrics} for r in runs],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/runs/compare", tags=["MLflow"])
def compare_runs(run_id_1: str = Query(...), run_id_2: str = Query(...)):
    """Compare two MLflow runs side by side."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        r1 = client.get_run(run_id_1)
        r2 = client.get_run(run_id_2)
        return {
            "run_1": {"run_id": run_id_1, "model": r1.data.params.get("model"), **r1.data.params, **r1.data.metrics},
            "run_2": {"run_id": run_id_2, "model": r2.data.params.get("model"), **r2.data.params, **r2.data.metrics},
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/recommend", tags=["MLflow"])
def recommend(experiment_name: str = "ml_experiment"):
    """
    Recommend the best model based on MLflow runs.
    
    Selects model with highest metric (F1 for classification, R² for regression)
    and checks for overfitting if CV data is available.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            raise HTTPException(status_code=404, detail="Experiment not found.")

        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        if not runs:
            raise HTTPException(status_code=404, detail="No runs found.")

        task_type = runs[0].data.params.get("task_type", "classification")
        score_key = "f1_weighted" if task_type == "classification" else "r2"

        scored = [(r, r.data.metrics.get(score_key, -1)) for r in runs if score_key in r.data.metrics]
        if not scored:
            raise HTTPException(status_code=404, detail=f"Metric '{score_key}' not found in runs.")

        best_run, best_score = max(scored, key=lambda x: x[1])
        model_name = best_run.data.params.get("model", "?")

        cv_mean = best_run.data.metrics.get("cv_mean")
        cv_train = best_run.data.metrics.get("cv_train_mean")
        overfitting = (cv_train - cv_mean) > 0.05 if (cv_mean and cv_train) else None

        reasons = [f"Highest {score_key}: {best_score:.4f}"]
        if cv_mean:
            reasons.append(f"CV mean: {cv_mean:.4f} ± {best_run.data.metrics.get('cv_std', 0):.4f}")
        if overfitting is not None:
            reasons.append("Warning: overfitting detected" if overfitting else "No overfitting detected")

        all_scores = {r.data.params.get("model", "?"): round(s, 4) for r, s in scored}

        return {
            "recommended_model": model_name,
            "score": round(best_score, 4),
            "metric": score_key,
            "reasons": reasons,
            "all_scores": all_scores,
            "overfitting_warning": overfitting,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
