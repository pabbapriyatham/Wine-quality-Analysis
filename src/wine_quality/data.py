"""Load, validate, combine and clean the raw wine datasets."""

from pathlib import Path

import pandas as pd

from wine_quality.config import ProjectConfig, load_config

FEATURE_COLUMNS = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
TARGET_COLUMN = "quality"
CATEGORICAL_COLUMNS = ["wine_type"]
EXPECTED_COLUMNS = FEATURE_COLUMNS + CATEGORICAL_COLUMNS + [TARGET_COLUMN]


class WineQualityDataset:
    """Load both UCI wine files and prepare one consistent dataframe."""

    def __init__(self, config: ProjectConfig | None = None) -> None:
        """Create a loader using the supplied or default project settings."""
        self.config = config or load_config()

    def load_raw_data(self) -> pd.DataFrame:
        """Read both semicolon-separated CSVs and label their source wine type."""
        red = pd.read_csv(self.config.red_data_path, sep=";")
        white = pd.read_csv(self.config.white_data_path, sep=";")

        red["wine_type"] = "red"
        white["wine_type"] = "white"
        data = pd.concat([red, white], ignore_index=True)
        data.columns = [column.replace(" ", "_") for column in data.columns]
        self.validate_dataset(data)
        return data

    def validate_dataset(self, data: pd.DataFrame) -> None:
        """Check required columns, numeric values and valid wine labels."""
        if data.empty:
            raise ValueError("The wine dataset is empty.")

        missing_columns = set(EXPECTED_COLUMNS) - set(data.columns)
        if missing_columns:
            names = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required columns: {names}")

        for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
            if not pd.api.types.is_numeric_dtype(data[column]):
                raise ValueError(f"Column '{column}' must contain numeric values.")

        if data[TARGET_COLUMN].isna().any():
            raise ValueError("The target column 'quality' contains missing values.")

        valid_types = {"red", "white"}
        observed_types = set(data["wine_type"].dropna().unique())
        if not observed_types or not observed_types.issubset(valid_types):
            raise ValueError("Column 'wine_type' must contain 'red' and/or 'white'.")

    def remove_exact_duplicates(self, data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Drop repeated rows, counting them before removal."""
        duplicate_count = int(data.duplicated().sum())
        cleaned_data = data.drop_duplicates().reset_index(drop=True)
        return cleaned_data, duplicate_count

    def save_processed_data(self) -> tuple[pd.DataFrame, int, Path]:
        """Save a de-duplicated copy without changing the original CSV files."""
        data = self.load_raw_data()
        cleaned_data, duplicate_count = self.remove_exact_duplicates(data)
        self.config.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_data.to_csv(self.config.processed_data_path, index=False)
        return cleaned_data, duplicate_count, self.config.processed_data_path
