# Course concept mapping

| Course idea | Where it appears | How to explain it |
|---|---|---|
| Regression target and prediction error | `quality` in `data.py`; metrics in `evaluation.py` | The score is treated as a number, so predictions may be fractional. RMSE penalizes large errors more than MAE. |
| Data inspection and cleaning | `WineQualityDataset` in `data.py`; EDA notebook and `eda.py` | The two source files are combined with a `wine_type` column. Exact repeated rows are removed after combining. |
| Train/test separation | `split_data` in `models.py` | One 80/20 split with seed 42 holds back test rows until model selection is finished. |
| Preprocessing pipeline | `create_preprocessor` in `features.py`; pipelines in `models.py` | Numeric features are imputed and standardized. Wine type is imputed and one-hot encoded. Fitting these steps inside each training fold prevents information leaking from validation/test rows. |
| Baseline model | `DummyRegressor(strategy="mean")` in `models.py` | It predicts a constant mean. A useful model should improve on this simple reference. |
| Comparing regression algorithms | `WineQualityTrainer` in `models.py` | Linear regression, Ridge, Lasso, LinearSVR and k-NN represent linear, regularized, margin-based and neighbour methods. |
| Cross-validation and tuning | `GridSearchCV` with shuffled five-fold `KFold` in `models.py` | Each parameter option is fitted on four training folds and scored on the fifth, repeated across folds. The grid is ranked by RMSE. |
| Metrics | `evaluation.py` and `reports/evaluation.json` | MAE is average absolute error in points. RMSE is also in points and gives extra weight to larger errors. R² compares squared error to a mean-prediction reference. |
| Interpretation and limits | `permutation_importance` in `models.py`; final report | Shuffling a feature and observing score loss measures how much this fitted model relies on it. It does not establish causation. |
| Python modules, classes and tests | `src/wine_quality/`, `tests/` | Small modules separate data, transforms, training, prediction, CLI and plotting. Tests check expected behavior and errors. |

The assignment asks for regression, so classifier concepts such as class probabilities,
decision thresholds and a confusion matrix are not part of this implementation.
