"""Create the focused exploratory charts used by the notebook and report."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from wine_quality.data import FEATURE_COLUMNS, TARGET_COLUMN

matplotlib.use("Agg")


def create_eda_plots(data: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Save target, feature, correlation and wine-type plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    figure, axis = plt.subplots(figsize=(7, 4))
    quality_counts = data[TARGET_COLUMN].value_counts().sort_index()
    axis.bar(quality_counts.index, quality_counts.values, color="steelblue")
    axis.set(
        title="Wine quality score distribution", xlabel="Quality score", ylabel="Wines"
    )
    axis.grid(axis="y", alpha=0.3)
    axis.set_axisbelow(True)
    figure.tight_layout()
    path = output_dir / "quality_distribution.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    saved_paths.append(path)

    figure, axes = plt.subplots(3, 4, figsize=(14, 9))
    for axis, column in zip(axes.flat, FEATURE_COLUMNS, strict=False):
        axis.hist(data[column].dropna(), bins=30, color="steelblue")
        axis.set(title=column.replace("_", " "), xlabel=column.replace("_", " "))
        axis.grid(axis="y", alpha=0.3)
        axis.set_axisbelow(True)
    axes.flat[-1].set_visible(False)
    figure.suptitle("Physicochemical feature distributions")
    figure.tight_layout()
    path = output_dir / "feature_distributions.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    saved_paths.append(path)

    figure, axis = plt.subplots(figsize=(10, 8))
    correlation = data[FEATURE_COLUMNS + [TARGET_COLUMN]].corr()
    labels = [column.replace("_", " ") for column in correlation.columns]
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    figure.colorbar(image, ax=axis, label="Correlation")
    axis.set_title("Correlation between numeric measurements")
    figure.tight_layout()
    path = output_dir / "correlation_heatmap.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    saved_paths.append(path)

    for feature, filename in (
        ("alcohol", "alcohol_vs_quality.png"),
        ("volatile_acidity", "volatile_acidity_vs_quality.png"),
    ):
        figure, axis = plt.subplots(figsize=(7, 4))
        qualities = sorted(data[TARGET_COLUMN].dropna().unique())
        groups = [
            data.loc[data[TARGET_COLUMN] == quality, feature].dropna()
            for quality in qualities
        ]
        boxplot = axis.boxplot(groups, tick_labels=qualities, patch_artist=True)
        for box in boxplot["boxes"]:
            box.set_facecolor("lightsteelblue")
        axis.set(
            title=f"{feature.replace('_', ' ').title()} by quality score",
            xlabel="Quality score",
            ylabel=feature.replace("_", " ").title(),
        )
        axis.grid(axis="y", alpha=0.3)
        axis.set_axisbelow(True)
        figure.tight_layout()
        path = output_dir / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        saved_paths.append(path)

    figure, axis = plt.subplots(figsize=(7, 4))
    wine_types = sorted(data["wine_type"].dropna().unique())
    groups = [
        data.loc[data["wine_type"] == wine_type, TARGET_COLUMN].dropna()
        for wine_type in wine_types
    ]
    boxplot = axis.boxplot(groups, tick_labels=wine_types, patch_artist=True)
    for box in boxplot["boxes"]:
        box.set_facecolor("lightsteelblue")
    axis.set(
        title="Quality scores by wine type", xlabel="Wine type", ylabel="Quality score"
    )
    axis.grid(axis="y", alpha=0.3)
    axis.set_axisbelow(True)
    figure.tight_layout()
    path = output_dir / "quality_by_wine_type.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    saved_paths.append(path)

    return saved_paths
