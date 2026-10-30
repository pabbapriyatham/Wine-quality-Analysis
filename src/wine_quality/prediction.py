"""Validate a new wine record and predict its numerical quality score."""

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from wine_quality.config import ProjectConfig, load_config


class WinePredictionInput(BaseModel):
    """Expected numeric wine measurements and the source wine type."""

    model_config = ConfigDict(extra="forbid")

    fixed_acidity: float = Field(allow_inf_nan=False)
    volatile_acidity: float = Field(allow_inf_nan=False)
    citric_acid: float = Field(allow_inf_nan=False)
    residual_sugar: float = Field(allow_inf_nan=False)
    chlorides: float = Field(allow_inf_nan=False)
    free_sulfur_dioxide: float = Field(allow_inf_nan=False)
    total_sulfur_dioxide: float = Field(allow_inf_nan=False)
    density: float = Field(allow_inf_nan=False)
    pH: float = Field(allow_inf_nan=False)
    sulphates: float = Field(allow_inf_nan=False)
    alcohol: float = Field(allow_inf_nan=False)
    wine_type: Literal["red", "white"]


def predict_quality(
    input_data: WinePredictionInput,
    config: ProjectConfig | None = None,
) -> float:
    """Load the saved pipeline and return one predicted quality score."""
    settings = config or load_config()
    if not settings.model_path.is_file():
        raise FileNotFoundError(
            f"Saved model not found: {settings.model_path}. Run train first."
        )

    model = joblib.load(settings.model_path)
    input_frame = pd.DataFrame([input_data.model_dump()])
    return float(model.predict(input_frame)[0])


def read_prediction_input(input_path: Path) -> WinePredictionInput:
    """Read and validate a prediction input from a JSON file."""
    return WinePredictionInput.model_validate_json(
        input_path.read_text(encoding="utf-8")
    )
