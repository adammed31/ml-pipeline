import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    explained_variance_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate, train_test_split
from xgboost import XGBClassifier, XGBRegressor

from src.preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)

def _base_classifiers() -> dict:
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBClassifier": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0),
        "LGBMClassifier": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    }

def _base_regressors() -> dict:
    return {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBRegressor": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        "LGBMRegressor": LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
    }

def _ensemble_classifiers() -> dict:
    base = list(_base_classifiers().items())
    return {
        "VotingClassifier": VotingClassifier(estimators=base, voting="soft"),
    }

def _ensemble_regressors() -> dict:
    base = list(_base_regressors().items())
    return {
        "VotingRegressor": VotingRegressor(estimators=base),
    }

PARAM_GRIDS: dict = {
    "classification": {
        "LogisticRegression": {"C": [0.01, 0.1, 1, 10, 100], "solver": ["lbfgs", "liblinear"]},
        "RandomForestClassifier": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]},
        "XGBClassifier": {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1, 0.3], "subsample": [0.8, 1.0]},
        "LGBMClassifier": {"n_estimators": [50, 100, 200], "max_depth": [-1, 5, 10], "learning_rate": [0.01, 0.1, 0.3], "num_leaves": [31, 63, 127]},
    },
    "regression": {
        "LinearRegression": {},
        "RandomForestRegressor": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]},
        "XGBRegressor": {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1, 0.3], "subsample": [0.8, 1.0]},
        "LGBMRegressor": {"n_estimators": [50, 100, 200], "max_depth": [-1, 5, 10], "learning_rate": [0.01, 0.1, 0.3], "num_leaves": [31, 63, 127]},
    },
}


class ModelTrainer:
    """Train multiple models on any dataset and track everything with MLflow."""

    def __init__(self, experiment_name: str = "ml_experiment", tracking_uri: str = "mlruns"):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.preprocessor: DataPreprocessor | None = None
        self.best_model = None
        self.best_score: float = -np.inf
        self.best_model_name: str = ""
        self.feature_selector = None
        self._X_test = None
        self._y_test = None
        self._y_pred_best = None

    def train(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_size: float = 0.2,
        random_state: int = 42,
        cv_folds: int = 0,
        use_tuning: bool = False,
        search_strategy: str = "random",
        n_iter: int = 20,
        use_ensemble: bool = False,
        use_feature_selection: bool = False,
        use_calibration: bool = False,
        calibration_method: str = "sigmoid",
        num_imputer: str = "median",
        cat_imputer: str = "most_frequent",
        cat_encoder: str = "onehot",
    ) -> tuple[pd.DataFrame, object, DataPreprocessor]:
        self.preprocessor = DataPreprocessor(
            num_imputer=num_imputer,
            cat_imputer=cat_imputer,
            cat_encoder=cat_encoder,
        )
        X, y = self.preprocessor.fit_transform(df, target_col)

        task_type = self.preprocessor.task_type
        cv_scoring = "f1_weighted" if task_type == "classification" else "r2"

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self._X_train = X_train
        self._y_train = y_train

        if use_feature_selection:
            selector_model = (
                RandomForestClassifier(n_estimators=50, random_state=42)
                if task_type == "classification"
                else RandomForestRegressor(n_estimators=50, random_state=42)
            )
            self.feature_selector = SelectFromModel(selector_model, threshold="median")
            self.feature_selector.fit(X_train, y_train)
            X_train = self.feature_selector.transform(X_train)
            X_test = self.feature_selector.transform(X_test)
            selected = self.feature_selector.get_support()
            selected_names = [n for n, s in zip(self.preprocessor.feature_names_out, selected) if s]
            logger.info("Feature selection: %d/%d features kept: %s", len(selected_names), len(self.preprocessor.feature_names_out), selected_names)

        candidates = _base_classifiers() if task_type == "classification" else _base_regressors()
        if use_ensemble:
            ensembles = _ensemble_classifiers() if task_type == "classification" else _ensemble_regressors()
            candidates.update(ensembles)

        logger.info("Train/test split: %d / %d samples", len(X_train), len(X_test))

        rows = []
        for name, model in candidates.items():
            with mlflow.start_run(run_name=name):
                mlflow.log_params({
                    "model": name,
                    "task_type": task_type,
                    "test_size": test_size,
                    "n_features_in": X.shape[1],
                    "n_features_used": X_train.shape[1],
                    "n_train": X_train.shape[0],
                    "n_test": X_test.shape[0],
                    "use_feature_selection": use_feature_selection,
                    "cv_folds": cv_folds,
                    "use_tuning": use_tuning,
                })

                row = {"model": name}

                if use_tuning and name in PARAM_GRIDS.get(task_type, {}):
                    model, best_params = self._tune_model(
                        model, name, task_type, X_train, y_train,
                        cv_scoring, search_strategy, n_iter, cv_folds or 5,
                    )
                    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
                    row["best_params"] = str(best_params)

                if cv_folds >= 2:
                    cv_results = cross_validate(
                        model, X_train, y_train,
                        cv=cv_folds, scoring=cv_scoring,
                        return_train_score=True, n_jobs=-1,
                    )
                    cv_mean = round(float(np.mean(cv_results["test_score"])), 4)
                    cv_std = round(float(np.std(cv_results["test_score"])), 4)
                    train_mean = round(float(np.mean(cv_results["train_score"])), 4)
                    mlflow.log_metrics({"cv_mean": cv_mean, "cv_std": cv_std, "cv_train_mean": train_mean})
                    row.update({"cv_mean": cv_mean, "cv_std": cv_std, "cv_train_mean": train_mean})

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                metrics = self._compute_metrics(y_test, y_pred, task_type, model, X_test)
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, artifact_path="model")

                score = metrics.get("f1_weighted", metrics.get("r2", -np.inf))
                if score > self.best_score:
                    self.best_score = score
                    self.best_model = model
                    self.best_model_name = name
                    self._X_test = X_test
                    self._y_test = y_test
                    self._y_pred_best = y_pred

                row.update(metrics)
                rows.append(row)
                logger.info("%s → %s", name, metrics)

        logger.info("Best model: %s (score=%.4f)", self.best_model_name, self.best_score)

        self.calibrated_model = None
        if use_calibration and task_type == "classification" and hasattr(self.best_model, "predict_proba"):
            calibrated = CalibratedClassifierCV(clone(self.best_model), method=calibration_method, cv=5)
            calibrated.fit(self._X_train, self._y_train)
            self.calibrated_model = calibrated
            logger.info("Calibration applied (%s) on best model.", calibration_method)

        return pd.DataFrame(rows), self.best_model, self.preprocessor

    def get_evaluation_artifacts(self) -> dict:
        from sklearn.metrics import confusion_matrix, roc_curve, auc

        task_type = self.preprocessor.task_type
        y_test = self._y_test
        y_pred = self._y_pred_best
        artifacts = {}

        if task_type == "classification":
            cm = confusion_matrix(y_test, y_pred)
            le = self.preprocessor.label_encoder
            classes = le.classes_.tolist() if le else [str(c) for c in sorted(set(y_test))]
            artifacts["confusion_matrix"] = cm.tolist()
            artifacts["classes"] = classes

            if hasattr(self.best_model, "predict_proba") and len(classes) == 2:
                proba = self.best_model.predict_proba(self._X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, proba)
                artifacts["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": round(float(auc(fpr, tpr)), 4)}
        else:
            artifacts["residuals"] = (y_test - y_pred).tolist()
            artifacts["y_test"] = y_test.tolist()
            artifacts["y_pred"] = y_pred.tolist()

        return artifacts

    def save(self, model_dir: str = "models") -> None:
        Path(model_dir).mkdir(exist_ok=True)
        joblib.dump(self.best_model, f"{model_dir}/best_model.pkl")
        joblib.dump(self.best_model, f"{model_dir}/best_model_original.pkl")
        self.preprocessor.save(f"{model_dir}/preprocessor.pkl")
        if self.feature_selector is not None:
            joblib.dump(self.feature_selector, f"{model_dir}/feature_selector.pkl")
        if self.calibrated_model is not None:
            joblib.dump(self.calibrated_model, f"{model_dir}/best_model_calibrated.pkl")
        logger.info("Artifacts saved to %s/", model_dir)

    @staticmethod
    def _tune_model(model, name, task_type, X_train, y_train, scoring, strategy, n_iter, cv):
        param_grid = PARAM_GRIDS[task_type].get(name, {})
        if not param_grid:
            return model, {}
        if strategy == "grid":
            search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, refit=True)
        else:
            search = RandomizedSearchCV(
                model, param_grid, n_iter=min(n_iter, _grid_size(param_grid)),
                cv=cv, scoring=scoring, n_jobs=-1, refit=True, random_state=42,
            )
        search.fit(X_train, y_train)
        logger.info("%s best score CV=%.4f params=%s", name, search.best_score_, search.best_params_)
        return search.best_estimator_, search.best_params_

    @staticmethod
    def _compute_metrics(y_true, y_pred, task_type: str, model, X_test) -> dict:
        if task_type == "classification":
            n_classes = len(np.unique(y_true))
            avg = "binary" if n_classes == 2 else "weighted"
            metrics = {
                "accuracy": round(accuracy_score(y_true, y_pred), 4),
                "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
                "precision": round(precision_score(y_true, y_pred, average=avg, zero_division=0), 4),
                "recall": round(recall_score(y_true, y_pred, average=avg, zero_division=0), 4),
                "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
                "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
                "cohen_kappa": round(cohen_kappa_score(y_true, y_pred), 4),
            }
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                metrics["log_loss"] = round(log_loss(y_true, proba), 4)
                if n_classes == 2:
                    metrics["roc_auc"] = round(roc_auc_score(y_true, proba[:, 1]), 4)
                else:
                    metrics["roc_auc_ovr"] = round(roc_auc_score(y_true, proba, multi_class="ovr", average="weighted"), 4)
        else:
            mse = mean_squared_error(y_true, y_pred)
            metrics = {
                "mse": round(mse, 4),
                "rmse": round(np.sqrt(mse), 4),
                "mae": round(mean_absolute_error(y_true, y_pred), 4),
                "median_ae": round(median_absolute_error(y_true, y_pred), 4),
                "r2": round(r2_score(y_true, y_pred), 4),
                "explained_variance": round(explained_variance_score(y_true, y_pred), 4),
            }
            if not np.any(y_true == 0):
                metrics["mape"] = round(mean_absolute_percentage_error(y_true, y_pred), 4)
        return metrics


def _grid_size(param_grid: dict) -> int:
    size = 1
    for v in param_grid.values():
        size *= len(v)
    return size
