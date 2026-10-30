# Wine Quality Prediction

This is an Aivancity PGE 2 Machine Learning and Advanced Python project. It uses
regression to estimate the numerical `quality` score of red and white Vinho Verde
wines from physicochemical measurements.

## Dataset

The data come from the [UCI Wine Quality dataset (ID 186)](https://archive.ics.uci.edu/dataset/186/wine%2Bquality), licensed CC BY 4.0. The raw red and white CSVs are stored in `data/raw/`. Copies also sit beside the notebook in `notebooks/`, so it can load the data from its own folder. The project adds `wine_type` from each file's source before combining them. Please cite:

> Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Wine Quality [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56S3T.

The target is an ordinal score, but this project treats it as a numerical regression target because that is the selected assignment task. Regression predictions can be fractional and should not be interpreted as expert tasting judgments.

## Installation (Windows PowerShell)

```powershell
& "$env:USERPROFILE\wine-quality-venv\Scripts\Activate.ps1"
python -m pip install -e ".[dev]"
```

The project requires Python 3.13. Exact project and development dependency versions are in `pyproject.toml` and `requirements.txt`.

## Commands

```powershell
wine-quality train
wine-quality evaluate
wine-quality predict --input examples/new_case.json
```

`train` removes exact duplicate observations after attaching `wine_type`, saves a
processed CSV, splits the cleaned data once, compares a mean `DummyRegressor` with
LinearRegression, and tunes Ridge, Lasso, k-NN and LinearSVR using the same shuffled
five-fold training cross-validation. It selects by cross-validation RMSE, then scores
the selected model once on the untouched test split. The fitted preprocessing and
regressor are saved together.

`evaluate` reads the saved test metrics and does not retrain. `predict` validates a
single JSON record and loads the saved pipeline.

## Project structure

```text
config.yaml
data/raw/                 Original UCI CSV files and metadata
data/processed/           Combined data with exact duplicates removed
models/                   Saved scikit-learn pipeline
notebooks/                EDA and modelling notebook
reports/                  Evaluation results, figures and final documents
src/wine_quality/         Reusable Python package and CLI
tests/                    pytest unit tests
```

The notebook defines small `load_data`, `clean_data` and `train_models` functions.
They call the package code rather than maintaining a second copy of the training logic. `config.yaml` holds portable relative paths, the split seed and
cross-validation settings.

The final written report and 8-slide defence deck are in `reports/final_report.pdf`
and `reports/defence_presentation.pdf`. `COURSE_CONCEPT_MAPPING.md` links the code to
course topics, and `DEFENCE_PREPARATION.md` gives beginner-level module explanations
and oral-exam questions.

## Testing and code checks

```powershell
pytest
pytest --cov=wine_quality --cov-report=term-missing
ruff check src tests
```

## Results

The training command ran with 5,320 unique rows after exact duplicate removal. It
selected KNeighborsRegressor by its lowest mean five-fold training RMSE. Its selected
settings were 21 neighbours and distance weighting.

| Model | Cross-validation RMSE | Cross-validation MAE | Cross-validation R² |
|---|---:|---:|---:|
| KNeighborsRegressor | 0.715 | 0.553 | 0.343 |
| Ridge | 0.735 | 0.567 | 0.306 |
| LinearRegression | 0.735 | 0.567 | 0.306 |
| Lasso | 0.735 | 0.567 | 0.305 |
| LinearSVR | 0.737 | 0.566 | 0.302 |
| DummyRegressor | 0.883 | 0.697 | -0.002 |

On the untouched 1,064-row test split, the selected model reached RMSE 0.685,
MAE 0.534 and R² 0.375. The mean DummyRegressor scored RMSE 0.868, MAE 0.694
and R² -0.001 on the same split. The test results were calculated after model choice
and tuning. `reports/evaluation.json` contains the full precision values, chosen
parameters and permutation importance results.

The errors are measured in quality-score points. MAE 0.534 means the absolute
prediction error averaged about half a point on this test set. R² 0.375 means the
model explains about 37.5% of the test target variance relative to predicting the
test-set mean; it leaves substantial variation unexplained. Permutation importance
ranked alcohol highest, but importance is a predictive association and does not show
that changing alcohol causes a quality change.

## Limitations

- `quality` is an ordinal sensory rating, not a direct measure of commercial value.
- The available inputs are physicochemical tests; grape, brand, price and tasting
  context are not included.
- Exact duplicate rows are removed to avoid identical records crossing the train/test
  boundary. The raw files remain unchanged, and repeated measurements may represent
  distinct wines that happen to share the same recorded values.
- The dataset describes Portuguese Vinho Verde wines and may not generalize to other
  regions, production methods or future samples.
- Feature importance describes predictive association, not causation.

## AI assistance

See `AI_USAGE.md` for an honest record of AI assistance and the student review that
remains necessary before submission.
