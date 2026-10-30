"""Compare and tune regression pipelines, then evaluate one final model."""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR

from wine_quality.config import ProjectConfig, load_config
from wine_quality.data import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    WineQualityDataset,
)
from wine_quality.evaluation import calculate_metrics, create_residual_plot
from wine_quality.features import create_preprocessor

SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


def split_data(
    data: pd.DataFrame, config: ProjectConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split rows once so that later preprocessing sees training data only."""
    settings = config or load_config()
    feature_columns = FEATURE_COLUMNS + CATEGORICAL_COLUMNS
    X = data[feature_columns]
    y = data[TARGET_COLUMN]
    return train_test_split(
        X,
        y,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )


def create_pipeline(model: RegressorMixin) -> Pipeline:
    """Place preprocessing and a regressor in one scikit-learn pipeline."""
    return Pipeline(steps=[("preprocessor", create_preprocessor()), ("model", model)])


def regression_models(random_state: int) -> dict[str, RegressorMixin]:
    """Return the regression models required for the project comparison."""
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(max_iter=20_000, random_state=random_state),
        "KNeighborsRegressor": KNeighborsRegressor(),
        "LinearSVR": LinearSVR(max_iter=20_000, random_state=random_state),
    }


class WineQualityTrainer:
    """Run cross-validation, tuning and one final test evaluation."""

    def __init__(self, config: ProjectConfig | None = None) -> None:
        """Create a trainer using the supplied or default project settings."""
        self.config = config or load_config()
        self.dataset = WineQualityDataset(self.config)

    def _cross_validate(
        self,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> dict[str, float]:
        """Evaluate one pipeline using the shared shuffled K-fold strategy."""
        folds = KFold(
            n_splits=self.config.cv_splits,
            shuffle=True,
            random_state=self.config.random_state,
        )
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=folds,
            scoring=SCORING,
            n_jobs=1,
        )
        return {
            "cv_rmse": float(-np.mean(scores["test_rmse"])),
            "cv_mae": float(-np.mean(scores["test_mae"])),
            "cv_r2": float(np.mean(scores["test_r2"])),
        }

    def _tune_model(
        self,
        name: str,
        model: RegressorMixin,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> tuple[Pipeline, dict[str, Any], dict[str, float]]:
        """Tune one model with the shared folds and negative RMSE scoring."""
        parameter_grids: dict[str, dict[str, list[Any]]] = {
            "Ridge": {"model__alpha": [0.1, 1.0, 10.0]},
            "Lasso": {"model__alpha": [0.001, 0.01, 0.1, 1.0]},
            "KNeighborsRegressor": {
                "model__n_neighbors": [5, 11, 21],
                "model__weights": ["uniform", "distance"],
            },
            "LinearSVR": {"model__C": [0.1, 1.0, 10.0]},
        }
        folds = KFold(
            n_splits=self.config.cv_splits,
            shuffle=True,
            random_state=self.config.random_state,
        )
        search = GridSearchCV(
            estimator=create_pipeline(model),
            param_grid=parameter_grids[name],
            scoring=SCORING,
            refit="rmse",
            cv=folds,
            n_jobs=1,
            return_train_score=False,
        )
        search.fit(X_train, y_train)

        best_index = int(search.best_index_)
        cv_results = search.cv_results_
        metrics = {
            "cv_rmse": float(-cv_results["mean_test_rmse"][best_index]),
            "cv_mae": float(-cv_results["mean_test_mae"][best_index]),
            "cv_r2": float(cv_results["mean_test_r2"][best_index]),
        }
        return search.best_estimator_, search.best_params_, metrics

    def train(self) -> dict[str, Any]:
        """Train, compare and save the selected model and evaluation results."""
        raw_data = self.dataset.load_raw_data()
        raw_duplicate_count = int(raw_data.duplicated().sum())
        cleaned_data, duplicate_count = self.dataset.remove_exact_duplicates(raw_data)
        self.config.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_data.to_csv(self.config.processed_data_path, index=False)

        X_train, X_test, y_train, y_test = split_data(cleaned_data, self.config)
        folds = KFold(
            n_splits=self.config.cv_splits,
            shuffle=True,
            random_state=self.config.random_state,
        )

        models = regression_models(self.config.random_state)
        baseline = create_pipeline(DummyRegressor(strategy="mean"))
        baseline_scores = cross_validate(
            baseline,
            X_train,
            y_train,
            cv=folds,
            scoring=SCORING,
            n_jobs=1,
        )
        results: dict[str, dict[str, Any]] = {
            "DummyRegressor": {
                "pipeline": baseline,
                "params": {"strategy": "mean"},
                "metrics": {
                    "cv_rmse": float(-np.mean(baseline_scores["test_rmse"])),
                    "cv_mae": float(-np.mean(baseline_scores["test_mae"])),
                    "cv_r2": float(np.mean(baseline_scores["test_r2"])),
                },
            }
        }

        linear_pipeline = create_pipeline(models["LinearRegression"])
        results["LinearRegression"] = {
            "pipeline": linear_pipeline,
            "params": {},
            "metrics": self._cross_validate(linear_pipeline, X_train, y_train),
        }

        for name in ("Ridge", "Lasso", "KNeighborsRegressor", "LinearSVR"):
            best_pipeline, best_params, metrics = self._tune_model(
                name, models[name], X_train, y_train
            )
            results[name] = {
                "pipeline": best_pipeline,
                "params": best_params,
                "metrics": metrics,
            }

        selected_name = min(
            results, key=lambda name: results[name]["metrics"]["cv_rmse"]
        )
        selected_pipeline = results[selected_name]["pipeline"]
        selected_pipeline.fit(X_train, y_train)
        test_predictions = selected_pipeline.predict(X_test)
        test_metrics = calculate_metrics(y_test.to_numpy(), test_predictions)

        baseline_test_pipeline = create_pipeline(DummyRegressor(strategy="mean")).fit(
            X_train, y_train
        )
        baseline_test_predictions = baseline_test_pipeline.predict(X_test)
        baseline_test_metrics = calculate_metrics(
            y_test.to_numpy(), baseline_test_predictions
        )

        importance = permutation_importance(
            selected_pipeline,
            X_test,
            y_test,
            scoring="neg_root_mean_squared_error",
            n_repeats=5,
            random_state=self.config.random_state,
            n_jobs=1,
        )
        feature_importance = sorted(
            [
                {
                    "feature": feature,
                    "importance_mean": float(mean),
                    "importance_std": float(std),
                }
                for feature, mean, std in zip(
                    X_test.columns,
                    importance.importances_mean,
                    importance.importances_std,
                    strict=True,
                )
            ],
            key=lambda item: item["importance_mean"],
            reverse=True,
        )

        self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(selected_pipeline, self.config.model_path)
        create_residual_plot(
            y_test.to_numpy(),
            test_predictions,
            self.config.figures_dir / "test_residuals.png",
        )

        comparison = []
        for name, result in results.items():
            comparison.append(
                {
                    "model": name,
                    **result["metrics"],
                    "best_params": result["params"],
                }
            )
        comparison.sort(key=lambda item: item["cv_rmse"])

        def saved_path(path: Path) -> str:
            try:
                return path.relative_to(self.config.project_root).as_posix()
            except ValueError:
                return str(path)

        summary: dict[str, Any] = {
            "dataset": {
                "raw_rows": int(len(raw_data)),
                "clean_rows": int(len(cleaned_data)),
                "raw_duplicate_rows": raw_duplicate_count,
                "duplicate_rows_removed": duplicate_count,
                "missing_cells": int(raw_data.isna().sum().sum()),
                "wine_type_counts": {
                    key: int(value)
                    for key, value in raw_data["wine_type"].value_counts().items()
                },
            },
            "split": {
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "random_state": self.config.random_state,
                "test_size": self.config.test_size,
            },
            "model_comparison": comparison,
            "selected_model": selected_name,
            "selected_parameters": results[selected_name]["params"],
            "baseline_test_metrics": baseline_test_metrics,
            "test_metrics": test_metrics,
            "feature_importance": feature_importance,
            "model_path": saved_path(self.config.model_path),
            "residual_plot_path": saved_path(
                self.config.figures_dir / "test_residuals.png"
            ),
        }
        self.config.evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.evaluation_path.write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary


def load_evaluation(config: ProjectConfig | None = None) -> dict[str, Any]:
    """Read saved evaluation results without training or model selection."""
    settings = config or load_config()
    if not settings.evaluation_path.is_file():
        raise FileNotFoundError(
            "Evaluation results not found: "
            f"{settings.evaluation_path}. Run train first."
        )
    return json.loads(settings.evaluation_path.read_text(encoding="utf-8"))
