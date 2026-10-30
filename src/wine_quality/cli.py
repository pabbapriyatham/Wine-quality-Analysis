"""Typer command-line interface for training, evaluation and prediction."""

import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from wine_quality.models import WineQualityTrainer, load_evaluation
from wine_quality.prediction import predict_quality, read_prediction_input

app = typer.Typer(help="Predict wine quality from physicochemical measurements.")


@app.command()
def train() -> None:
    """Train and tune models, evaluate the selected model, and save it."""
    summary = WineQualityTrainer().train()
    metrics = summary["test_metrics"]
    typer.echo(f"Selected model: {summary['selected_model']}")
    typer.echo(
        "Test metrics: "
        f"RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}, "
        f"R2={metrics['r2']:.3f}"
    )
    baseline = summary["baseline_test_metrics"]
    typer.echo(f"DummyRegressor test RMSE: {baseline['rmse']:.3f}")
    typer.echo(f"Saved model: {summary['model_path']}")


@app.command()
def evaluate() -> None:
    """Display saved test metrics without retraining the model."""
    summary = load_evaluation()
    saved_results = {
        "selected_model": summary["selected_model"],
        "baseline_test_metrics": summary["baseline_test_metrics"],
        "selected_model_test_metrics": summary["test_metrics"],
    }
    typer.echo(json.dumps(saved_results, indent=2))


@app.command()
def predict(
    input: Annotated[Path, typer.Option("--input", exists=True, readable=True)],
) -> None:
    """Predict quality for one input JSON file."""
    try:
        wine = read_prediction_input(input)
        prediction = predict_quality(wine)
    except (ValidationError, ValueError, FileNotFoundError) as error:
        typer.echo(f"Could not make a prediction: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Predicted wine quality: {prediction:.2f} out of 10")
