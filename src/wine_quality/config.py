"""Read project settings from the YAML configuration file."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    """Paths and settings shared by the project modules."""

    project_root: Path
    red_data_path: Path
    white_data_path: Path
    processed_data_path: Path
    model_path: Path
    evaluation_path: Path
    figures_dir: Path
    random_state: int
    test_size: float
    cv_splits: int


def load_config(config_path: Path | None = None) -> ProjectConfig:
    """Load YAML settings and resolve paths relative to the project folder."""
    if config_path is None:
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "config.yaml"
    else:
        config_path = config_path.resolve()
        project_root = config_path.parent

    with config_path.open(encoding="utf-8") as file:
        settings: dict[str, Any] = yaml.safe_load(file)

    paths = settings["paths"]
    training = settings["training"]

    def resolve_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else project_root / path

    return ProjectConfig(
        project_root=project_root,
        red_data_path=resolve_path(paths["red_data"]),
        white_data_path=resolve_path(paths["white_data"]),
        processed_data_path=resolve_path(paths["processed_data"]),
        model_path=resolve_path(paths["model"]),
        evaluation_path=resolve_path(paths["evaluation"]),
        figures_dir=resolve_path(paths["figures"]),
        random_state=int(training["random_state"]),
        test_size=float(training["test_size"]),
        cv_splits=int(training["cv_splits"]),
    )
