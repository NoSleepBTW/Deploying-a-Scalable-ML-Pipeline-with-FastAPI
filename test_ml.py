import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture(scope="module")
def data():
    """Load the census dataset once for all tests."""
    return pd.read_csv("data/census.csv")


@pytest.fixture(scope="module")
def processed_train(data):
    """Process the full dataset into training arrays and encoders."""
    X, y, encoder, lb = process_data(
        data,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )
    return X, y, encoder, lb


def test_train_model_returns_random_forest(processed_train):
    """train_model returns a fitted RandomForestClassifier."""
    X, y, _, _ = processed_train
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    # A fitted classifier exposes the classes_ attribute.
    assert hasattr(model, "classes_")


def test_inference_returns_expected_shape_and_values(processed_train):
    """inference returns one binary prediction per input row."""
    X, y, _, _ = processed_train
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]
    # Labels are binarized, so predictions must only be 0 or 1.
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics_known_values():
    """compute_model_metrics returns the correct precision, recall, and F1."""
    y = np.array([0, 1, 1, 0, 1])
    preds = np.array([0, 1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    # 2 true positives, 0 false positives, 1 false negative.
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(2 / 3)
    assert fbeta == pytest.approx(0.8)


def test_process_data_shapes_and_types(data):
    """process_data returns aligned arrays and fitted transformers."""
    from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

    X, y, encoder, lb = process_data(
        data,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )
    assert X.shape[0] == y.shape[0] == len(data)
    assert isinstance(encoder, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)
