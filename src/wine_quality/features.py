"""Build preprocessing steps that are fitted inside each model pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from wine_quality.data import CATEGORICAL_COLUMNS, FEATURE_COLUMNS


def create_preprocessor() -> ColumnTransformer:
    """Create numeric and categorical transformations for wine features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, FEATURE_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )
