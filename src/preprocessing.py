import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)


class DataPreprocessor:

    def __init__(
        self,
        num_imputer: str = "median",
        cat_imputer: str = "most_frequent",
        cat_encoder: str = "onehot",
    ):
        self.num_imputer = num_imputer
        self.cat_imputer = cat_imputer
        self.cat_encoder = cat_encoder
        self.pipeline: ColumnTransformer | None = None
        self.label_encoder: LabelEncoder | None = None
        self.task_type: str | None = None
        self.target_col: str | None = None
        self.numerical_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.feature_names_out: list[str] = []

    def fit_transform(self, df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, np.ndarray]:
        self.target_col = target_col
        X = df.drop(columns=[target_col])
        y = df[target_col].copy()

        self.task_type = self._detect_task_type(y)
        logger.info("Task type detected: %s", self.task_type)

        self.numerical_cols, self.categorical_cols = self._split_column_types(X)
        logger.info("Numerical columns (%d): %s", len(self.numerical_cols), self.numerical_cols)
        logger.info("Categorical columns (%d): %s", len(self.categorical_cols), self.categorical_cols)

        self.pipeline = self._build_pipeline()
        X_out = self.pipeline.fit_transform(X)

        if self.task_type == "classification" and y.dtype == object:
            self.label_encoder = LabelEncoder()
            y = pd.Series(self.label_encoder.fit_transform(y))

        self.feature_names_out = self._extract_feature_names()
        return X_out, y.to_numpy()

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df.drop(columns=[self.target_col], errors="ignore")
        return self.pipeline.transform(X)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)
        logger.info("Preprocessor saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "DataPreprocessor":
        obj = joblib.load(path)
        logger.info("Preprocessor loaded from %s", path)
        return obj

    @staticmethod
    def _detect_task_type(y: pd.Series) -> str:
        if y.dtype == object or y.dtype.name == "category":
            return "classification"
        if y.nunique() <= 20:
            return "classification"
        return "regression"

    @staticmethod
    def _split_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
        numerical = X.select_dtypes(include=["number"]).columns.tolist()
        categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        return numerical, categorical

    def _build_pipeline(self) -> ColumnTransformer:
        transformers = []

        if self.numerical_cols:
            if self.num_imputer == "knn":
                num_imputer_obj = KNNImputer(n_neighbors=5)
            elif self.num_imputer == "mean":
                num_imputer_obj = SimpleImputer(strategy="mean")
            elif self.num_imputer == "constant":
                num_imputer_obj = SimpleImputer(strategy="constant", fill_value=0)
            else:
                num_imputer_obj = SimpleImputer(strategy="median")
            num_pipe = Pipeline([
                ("imputer", num_imputer_obj),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("num", num_pipe, self.numerical_cols))

        if self.categorical_cols:
            cat_strategy = self.cat_imputer if self.cat_imputer == "most_frequent" else "constant"
            cat_imputer_kwargs = {"strategy": cat_strategy}
            if cat_strategy == "constant":
                cat_imputer_kwargs["fill_value"] = "missing"

            if self.cat_encoder == "ordinal":
                encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            else:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(**cat_imputer_kwargs)),
                ("encoder", encoder),
            ])
            transformers.append(("cat", cat_pipe, self.categorical_cols))

        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _extract_feature_names(self) -> list[str]:
        names = list(self.numerical_cols)
        if self.categorical_cols and "cat" in self.pipeline.named_transformers_:
            enc = self.pipeline.named_transformers_["cat"].named_steps["encoder"]
            if isinstance(enc, OneHotEncoder):
                names += enc.get_feature_names_out(self.categorical_cols).tolist()
            else:
                names += self.categorical_cols
        return names
