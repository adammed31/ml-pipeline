import numpy as np
import pandas as pd
import pytest

from src.preprocessing import DataPreprocessor


def test_classification_task_detected(clf_df):
    prep = DataPreprocessor()
    X, y = prep.fit_transform(clf_df, "bought")
    assert prep.task_type == "classification"


def test_regression_task_detected(reg_df):
    prep = DataPreprocessor()
    X, y = prep.fit_transform(reg_df, "price")
    assert prep.task_type == "regression"


def test_output_shape(clf_df):
    prep = DataPreprocessor()
    X, y = prep.fit_transform(clf_df, "bought")
    assert X.shape[0] == len(clf_df)
    assert len(y) == len(clf_df)


def test_no_nan_after_transform(df_with_missing):
    prep = DataPreprocessor()
    X, y = prep.fit_transform(df_with_missing, "target")
    assert not np.isnan(X).any()


def test_no_nan_after_knn_impute(df_with_missing):
    prep = DataPreprocessor(num_imputer="knn")
    X, y = prep.fit_transform(df_with_missing, "target")
    assert not np.isnan(X).any()


def test_onehot_encoding(clf_df):
    prep = DataPreprocessor(cat_encoder="onehot")
    X, y = prep.fit_transform(clf_df, "bought")
    assert X.shape[1] >= 3


def test_ordinal_encoding(clf_df):
    prep_ohe = DataPreprocessor(cat_encoder="onehot")
    prep_ord = DataPreprocessor(cat_encoder="ordinal")
    X_ohe, _ = prep_ohe.fit_transform(clf_df, "bought")
    X_ord, _ = prep_ord.fit_transform(clf_df, "bought")
    assert X_ord.shape[1] < X_ohe.shape[1]


def test_mean_imputer(df_with_missing):
    prep = DataPreprocessor(num_imputer="mean")
    X, y = prep.fit_transform(df_with_missing, "target")
    assert not np.isnan(X).any()


def test_constant_imputer(df_with_missing):
    prep = DataPreprocessor(num_imputer="constant", cat_imputer="constant")
    X, y = prep.fit_transform(df_with_missing, "target")
    assert not np.isnan(X).any()


def test_transform_new_data(clf_df):
    prep = DataPreprocessor()
    prep.fit_transform(clf_df, "bought")
    X_new = prep.transform(clf_df.drop(columns=["bought"]))
    assert X_new.shape[0] == len(clf_df)


def test_feature_names_out(clf_df):
    prep = DataPreprocessor()
    X, _ = prep.fit_transform(clf_df, "bought")
    assert len(prep.feature_names_out) == X.shape[1]


def test_save_load(clf_df, tmp_path):
    prep = DataPreprocessor()
    prep.fit_transform(clf_df, "bought")
    path = str(tmp_path / "prep.pkl")
    prep.save(path)
    prep2 = DataPreprocessor.load(path)
    assert prep2.task_type == "classification"
