"""Basic command-line interface tests."""

import json

from typer.testing import CliRunner

import wine_quality.cli as cli
from wine_quality.cli import app


def test_cli_help_lists_all_three_commands() -> None:
    """The help output shows train, evaluate and predict commands."""
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "train" in result.stdout
    assert "evaluate" in result.stdout
    assert "predict" in result.stdout


def test_predict_command_reports_invalid_json_fields(tmp_path) -> None:
    """An incomplete JSON record produces a non-zero CLI result."""
    input_path = tmp_path / "invalid.json"
    input_path.write_text(json.dumps({"wine_type": "red"}), encoding="utf-8")

    result = CliRunner().invoke(app, ["predict", "--input", str(input_path)])

    assert result.exit_code == 1
    assert "Could not make a prediction" in result.stderr


def test_train_command_displays_saved_model_summary(monkeypatch) -> None:
    """The train command reports the trainer's result."""
    summary = {
        "selected_model": "Ridge",
        "test_metrics": {"rmse": 0.5, "mae": 0.4, "r2": 0.6},
        "baseline_test_metrics": {"rmse": 1.0, "mae": 0.8, "r2": 0.0},
        "model_path": "models/model.joblib",
    }

    class FakeTrainer:
        """Return a known summary without fitting during this CLI unit test."""

        def train(self):
            return summary

    monkeypatch.setattr(cli, "WineQualityTrainer", FakeTrainer)
    result = CliRunner().invoke(app, ["train"])

    assert result.exit_code == 0
    assert "Selected model: Ridge" in result.stdout
    assert "RMSE=0.500" in result.stdout


def test_evaluate_command_displays_saved_metrics(monkeypatch) -> None:
    """The evaluate command displays saved metrics without fitting."""
    monkeypatch.setattr(
        cli,
        "load_evaluation",
        lambda: {
            "selected_model": "Ridge",
            "baseline_test_metrics": {"rmse": 1.0, "mae": 0.8, "r2": 0.0},
            "test_metrics": {"rmse": 0.5, "mae": 0.4, "r2": 0.6},
        },
    )

    result = CliRunner().invoke(app, ["evaluate"])

    assert result.exit_code == 0
    assert '"selected_model": "Ridge"' in result.stdout
    assert '"rmse": 1.0' in result.stdout
