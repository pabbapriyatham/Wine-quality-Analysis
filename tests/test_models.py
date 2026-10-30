"""Tests for splitting, pipeline training and regression evaluation."""

from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from wine_quality.config import load_config
from wine_quality.data import FEATURE_COLUMNS
from wine_quality.evaluation import calculate_metrics
from wine_quality.models import WineQualityTrainer, create_pipeline, split_data


def make_training_data(row_count: int = 30) -> pd.DataFrame:
    """Build a small regression dataframe with deterministic values."""
    data = pd.DataFrame(
        {
            column: np.linspace(1, 10, row_count) + position
            for position, column in enumerate(FEATURE_COLUMNS)
        }
    )
    data["wine_type"] = ["red", "white"] * (row_count // 2)
    data["quality"] = np.linspace(4, 8, row_count)
    return data


def test_train_test_split_has_disjoint_rows() -> None:
    """The test indices are held out from training."""
    training_data = make_training_data()
    config = load_config()

    X_train, X_test, y_train, y_test = split_data(training_data, config)

    assert X_train.index.intersection(X_test.index).empty
    assert y_train.index.equals(X_train.index)
    assert y_test.index.equals(X_test.index)
    assert len(X_test) == 6


def test_trainer_runs_shared_cross_validation() -> None:
    """The trainer returns regression metrics from the configured folds."""
    data = make_training_data()
    X = data[FEATURE_COLUMNS + ["wine_type"]]
    y = data["quality"]
    config = replace(load_config(), cv_splits=3)
    trainer = WineQualityTrainer(config)

    scores = trainer._cross_validate(create_pipeline(Ridge()), X, y)

    assert set(scores) == {"cv_rmse", "cv_mae", "cv_r2"}
    assert scores["cv_rmse"] > 0
    assert scores["cv_mae"] > 0


def test_pipeline_fits_and_returns_finite_predictions() -> None:
    """Preprocessing and the regressor can be fitted as one pipeline."""
    data = make_training_data()
    X = data[FEATURE_COLUMNS + ["wine_type"]]
    y = data["quality"]
    pipeline = create_pipeline(Ridge(alpha=1.0))

    pipeline.fit(X, y)
    predictions = pipeline.predict(X.iloc[:4])

    assert predictions.shape == (4,)
    assert np.isfinite(predictions).all()


def test_regression_metrics_match_a_small_example() -> None:
    """RMSE, MAE and R-squared have the expected values."""
    metrics = calculate_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))

    assert np.isclose(metrics["rmse"], np.sqrt(1 / 3))
    assert np.isclose(metrics["mae"], 1 / 3)
    assert np.isclose(metrics["r2"], 0.5)


def test_model_pipeline_predictions_survive_joblib_round_trip(tmp_path) -> None:
    """A saved fitted pipeline produces the same predictions after loading."""
    data = make_training_data()
    X = data[FEATURE_COLUMNS + ["wine_type"]]
    y = data["quality"]
    pipeline = create_pipeline(Ridge(alpha=1.0)).fit(X, y)
    expected = pipeline.predict(X.iloc[:3])
    model_path = tmp_path / "model.joblib"

    joblib.dump(pipeline, model_path)
    loaded_pipeline = joblib.load(model_path)
    actual = loaded_pipeline.predict(X.iloc[:3])

    assert np.allclose(actual, expected)
