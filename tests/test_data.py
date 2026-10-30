"""Tests for loading, validating and cleaning wine data."""

from pathlib import Path

import pandas as pd
import pytest

from wine_quality.config import load_config
from wine_quality.data import WineQualityDataset


def write_source_file(path: Path, quality_values: list[int]) -> None:
    """Write a small valid wine CSV for a test."""
    row = {
        "fixed acidity": 7.4,
        "volatile acidity": 0.7,
        "citric acid": 0.0,
        "residual sugar": 1.9,
        "chlorides": 0.076,
        "free sulfur dioxide": 11.0,
        "total sulfur dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
        "quality": 5,
    }
    rows = [{**row, "quality": quality} for quality in quality_values]
    pd.DataFrame(rows).to_csv(path, sep=";", index=False)


@pytest.fixture
def dataset_loader(tmp_path: Path) -> WineQualityDataset:
    """Create a loader pointing to temporary red and white files."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    red_path = raw_dir / "red.csv"
    white_path = raw_dir / "white.csv"
    write_source_file(red_path, [5, 5])
    write_source_file(white_path, [5])

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "paths:\n"
        "  red_data: data/raw/red.csv\n"
        "  white_data: data/raw/white.csv\n"
        "  processed_data: data/processed/wine.csv\n"
        "  model: models/model.joblib\n"
        "  evaluation: reports/evaluation.json\n"
        "  figures: reports/figures\n"
        "training:\n"
        "  random_state: 42\n"
        "  test_size: 0.2\n"
        "  cv_splits: 5\n",
        encoding="utf-8",
    )
    return WineQualityDataset(load_config(config_path))


def test_loads_both_files_and_adds_wine_type(
    dataset_loader: WineQualityDataset,
) -> None:
    """The combined data retains the source wine type."""
    data = dataset_loader.load_raw_data()

    assert len(data) == 3
    assert set(data["wine_type"]) == {"red", "white"}
    assert "quality" in data.columns


def test_validation_rejects_missing_required_column(
    dataset_loader: WineQualityDataset,
) -> None:
    """A missing predictor column produces a useful error."""
    data = dataset_loader.load_raw_data().drop(columns="alcohol")

    with pytest.raises(ValueError, match="Missing required columns"):
        dataset_loader.validate_dataset(data)


def test_validation_rejects_text_in_numeric_feature(
    dataset_loader: WineQualityDataset,
) -> None:
    """Numeric predictors cannot silently become strings."""
    data = dataset_loader.load_raw_data()
    data["alcohol"] = data["alcohol"].astype(str)

    with pytest.raises(ValueError, match="must contain numeric"):
        dataset_loader.validate_dataset(data)


def test_duplicate_removal_keeps_wine_type_in_the_record(
    dataset_loader: WineQualityDataset,
) -> None:
    """Rows from different wine types are not treated as duplicates."""
    data = dataset_loader.load_raw_data()
    cleaned, duplicate_count = dataset_loader.remove_exact_duplicates(data)

    assert duplicate_count == 1
    assert len(cleaned) == 2
    assert set(cleaned["wine_type"]) == {"red", "white"}
