"""Tests that the EDA function saves each expected chart."""

import numpy as np
import pandas as pd

from wine_quality.data import FEATURE_COLUMNS
from wine_quality.eda import create_eda_plots


def test_eda_saves_six_labeled_chart_files(tmp_path) -> None:
    """EDA writes its six charts to the requested folder."""
    generator = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            column: generator.normal(loc=5, scale=1, size=30)
            for column in FEATURE_COLUMNS
        }
    )
    data["quality"] = generator.integers(3, 9, size=30)
    data["wine_type"] = ["red", "white"] * 15

    chart_paths = create_eda_plots(data, tmp_path / "figures")

    assert len(chart_paths) == 6
    assert all(path.is_file() and path.stat().st_size > 0 for path in chart_paths)
