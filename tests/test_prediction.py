"""Tests for validating a new wine prediction input."""

from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.dummy import DummyRegressor

from wine_quality.config import load_config
from wine_quality.data import FEATURE_COLUMNS
from wine_quality.models import create_pipeline
from wine_quality.prediction import (
    WinePredictionInput,
    predict_quality,
    read_prediction_input,
)


def valid_wine_input() -> dict[str, float | str]:
    """Return one valid example using the dataset's feature names."""
    return {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11,
        "total_sulfur_dioxide": 34,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
        "wine_type": "red",
    }


def test_valid_prediction_input_is_accepted() -> None:
    """A complete numeric record with a known type validates."""
    record = WinePredictionInput.model_validate(valid_wine_input())

    assert record.wine_type == "red"
    assert record.alcohol == 9.4


def test_invalid_wine_type_is_rejected() -> None:
    """The model accepts only red or white wine types."""
    values = valid_wine_input()
    values["wine_type"] = "rose"

    with pytest.raises(ValidationError):
        WinePredictionInput.model_validate(values)


def test_non_finite_measurement_is_rejected() -> None:
    """NaN is not accepted as a prediction measurement."""
    values = valid_wine_input()
    values["alcohol"] = float("nan")

    with pytest.raises(ValidationError):
        WinePredictionInput.model_validate(values)


def test_prediction_loads_a_saved_pipeline(tmp_path) -> None:
    """The prediction function loads a persisted fitted model."""
    values = valid_wine_input()
    training_data = pd.DataFrame(
        {column: [1.0, 2.0, 3.0] for column in FEATURE_COLUMNS}
    )
    training_data["wine_type"] = ["red", "white", "red"]
    model = create_pipeline(DummyRegressor(strategy="mean")).fit(
        training_data, np.array([5.0, 6.0, 7.0])
    )
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)
    config = replace(load_config(), model_path=model_path)

    prediction = predict_quality(WinePredictionInput.model_validate(values), config)

    assert prediction == 6.0


def test_example_payload_files_validate() -> None:
    """The stored white and red wine example JSON files parse successfully."""
    config = load_config()
    examples_dir = config.project_root / "examples"

    white_case = read_prediction_input(examples_dir / "new_case.json")
    assert white_case.wine_type == "white"

    red_case = read_prediction_input(examples_dir / "new_case_red.json")
    assert red_case.wine_type == "red"

