"""Tests for the preprocessing used inside model pipelines."""

import numpy as np
import pandas as pd

from wine_quality.data import FEATURE_COLUMNS
from wine_quality.features import create_preprocessor


def make_feature_data() -> pd.DataFrame:
    """Create numeric and categorical measurements for two wines."""
    data = pd.DataFrame({column: [1.0, 3.0, 5.0] for column in FEATURE_COLUMNS})
    data["wine_type"] = ["red", "white", "red"]
    return data


def test_preprocessor_outputs_scaled_numeric_and_encoded_type() -> None:
    """The transformer returns numeric features plus one-hot wine type."""
    training_data = make_feature_data()
    preprocessor = create_preprocessor()

    transformed = preprocessor.fit_transform(training_data)

    assert transformed.shape == (3, len(FEATURE_COLUMNS) + 2)
    assert np.isfinite(transformed).all()


def test_scaler_is_fitted_on_training_rows_only() -> None:
    """A very different test value does not change the fitted train mean."""
    training_data = make_feature_data()
    preprocessor = create_preprocessor()
    preprocessor.fit(training_data)
    test_data = make_feature_data().iloc[[0]].copy()
    test_data.loc[:, FEATURE_COLUMNS] = 10_000.0

    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    fitted_means = numeric_pipeline.named_steps["scaler"].mean_

    assert np.allclose(fitted_means, training_data[FEATURE_COLUMNS].mean())
    preprocessor.transform(test_data)
    assert np.allclose(fitted_means, training_data[FEATURE_COLUMNS].mean())
