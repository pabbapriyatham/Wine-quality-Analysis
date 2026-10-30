"""A small integration test for the complete training workflow."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from wine_quality.config import load_config
from wine_quality.data import FEATURE_COLUMNS
from wine_quality.models import WineQualityTrainer


def write_synthetic_wines(path: Path, wine_type_offset: float) -> None:
    """Write realistic-shaped synthetic rows in the original UCI CSV format."""
    generator = np.random.default_rng(42 + int(wine_type_offset))
    rows = 30
    data = pd.DataFrame(
        {
            column: generator.uniform(1, 10, size=rows) + wine_type_offset
            for column in FEATURE_COLUMNS
        }
    )
    scores = 4 + (data["alcohol"] - 7) * 0.3 + generator.normal(0, 0.5, rows)
    data["quality"] = np.clip(np.rint(scores), 3, 9).astype(int)
    data.columns = [column.replace("_", " ") for column in data.columns]
    data.to_csv(path, sep=";", index=False)


def test_trainer_runs_cv_saves_model_and_evaluates_test_split(
    tmp_path: Path,
) -> None:
    """The training workflow saves a model and holdout evaluation summary."""
    red_path = tmp_path / "red.csv"
    white_path = tmp_path / "white.csv"
    write_synthetic_wines(red_path, 0.0)
    write_synthetic_wines(white_path, 1.0)
    config = replace(
        load_config(),
        red_data_path=red_path,
        white_data_path=white_path,
        processed_data_path=tmp_path / "processed" / "wine.csv",
        model_path=tmp_path / "models" / "model.joblib",
        evaluation_path=tmp_path / "reports" / "evaluation.json",
        figures_dir=tmp_path / "reports" / "figures",
        test_size=0.2,
        cv_splits=2,
    )

    summary = WineQualityTrainer(config).train()

    assert summary["dataset"]["raw_rows"] == 60
    assert summary["split"]["train_rows"] == 48
    assert summary["split"]["test_rows"] == 12
    assert len(summary["model_comparison"]) == 6
    assert set(summary["baseline_test_metrics"]) == {"rmse", "mae", "r2"}
    assert summary["selected_model"] in {
        "DummyRegressor",
        "LinearRegression",
        "Ridge",
        "Lasso",
        "KNeighborsRegressor",
        "LinearSVR",
    }
    assert config.model_path.is_file()
    assert config.evaluation_path.is_file()
    assert (config.figures_dir / "test_residuals.png").is_file()
