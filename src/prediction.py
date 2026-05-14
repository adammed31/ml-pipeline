import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)


class ModelPredictor:
    """Load a trained model + preprocessor and serve predictions."""

    def __init__(self, model_dir: str = "models"):
        model_path = Path(model_dir) / "best_model.pkl"
        preprocessor_path = Path(model_dir) / "preprocessor.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"No model found at {model_path}. Train a model first.")

        self.model = joblib.load(model_path)
        self.preprocessor: DataPreprocessor = DataPreprocessor.load(preprocessor_path)
        logger.info("Loaded model: %s", type(self.model).__name__)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self.preprocessor.transform(df)
        preds = self.model.predict(X)
        if self.preprocessor.label_encoder is not None:
            preds = self.preprocessor.label_encoder.inverse_transform(preds)
        return preds

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not hasattr(self.model, "predict_proba"):
            raise ValueError(f"{type(self.model).__name__} does not support predict_proba.")
        X = self.preprocessor.transform(df)
        return self.model.predict_proba(X)

    @property
    def task_type(self) -> str:
        return self.preprocessor.task_type

    @property
    def classes(self) -> list | None:
        if self.preprocessor.label_encoder is not None:
            return self.preprocessor.label_encoder.classes_.tolist()
        return None
