import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def clf_df():
    """Small binary classification dataset."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "age":    np.random.randint(18, 70, n).astype(float),
        "income": np.random.normal(50000, 15000, n),
        "city":   np.random.choice(["Paris", "Lyon", "Bordeaux"], n),
        "bought": np.random.choice([0, 1], n),
    })


@pytest.fixture
def reg_df():
    """Small regression dataset."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "surface": np.random.uniform(20, 200, n),
        "rooms":   np.random.randint(1, 6, n).astype(float),
        "city":    np.random.choice(["Paris", "Lyon", "Bordeaux"], n),
        "price":   np.random.uniform(100000, 800000, n),
    })


@pytest.fixture
def df_with_missing():
    """Dataset with missing values."""
    np.random.seed(42)
    n = 80
    df = pd.DataFrame({
        "age":    np.random.randint(18, 70, n).astype(float),
        "income": np.random.normal(50000, 15000, n),
        "city":   np.random.choice(["Paris", "Lyon", None], n),
        "target": np.random.choice([0, 1], n),
    })
    df.loc[::5, "age"] = np.nan
    df.loc[::7, "income"] = np.nan
    return df
