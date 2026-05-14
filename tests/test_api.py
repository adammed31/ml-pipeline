import io
import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient

os.environ["MLFLOW_TRACKING_URI"] = "mlruns_test"

from api.main import app

client = TestClient(app)


def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


@pytest.fixture
def clf_csv(clf_df):
    return _csv_bytes(clf_df)


@pytest.fixture
def reg_csv(reg_df):
    return _csv_bytes(reg_df)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200


def test_train_classification(clf_csv):
    resp = client.post(
        "/train",
        files={"file": ("data.csv", clf_csv, "text/csv")},
        params={"target_col": "bought", "experiment_name": "test_clf"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_type"] == "classification"
    assert "best_model" in data
    assert len(data["results"]) > 0


def test_train_regression(reg_csv):
    resp = client.post(
        "/train",
        files={"file": ("data.csv", reg_csv, "text/csv")},
        params={"target_col": "price", "experiment_name": "test_reg"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_type"] == "regression"


def test_train_invalid_target(clf_csv):
    resp = client.post(
        "/train",
        files={"file": ("data.csv", clf_csv, "text/csv")},
        params={"target_col": "nonexistent"},
    )
    assert resp.status_code == 422


def test_predict_after_train(clf_csv, clf_df):
    client.post(
        "/train",
        files={"file": ("data.csv", clf_csv, "text/csv")},
        params={"target_col": "bought", "experiment_name": "test_pred"},
    )
    pred_df = clf_df.drop(columns=["bought"])
    resp = client.post(
        "/predict",
        files={"file": ("data.csv", _csv_bytes(pred_df), "text/csv")},
    )
    assert resp.status_code == 200
    assert len(resp.json()["predictions"]) == len(clf_df)


def test_model_info_after_train(clf_csv):
    client.post(
        "/train",
        files={"file": ("data.csv", clf_csv, "text/csv")},
        params={"target_col": "bought", "experiment_name": "test_info"},
    )
    resp = client.get("/model_info")
    assert resp.status_code == 200
    assert "model_type" in resp.json()


def test_feature_importance_after_train(clf_csv):
    client.post(
        "/train",
        files={"file": ("data.csv", clf_csv, "text/csv")},
        params={"target_col": "bought", "experiment_name": "test_fi"},
    )
    resp = client.get("/feature_importance")
    assert resp.status_code == 200
    assert "feature_names" in resp.json()
