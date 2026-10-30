# Defence preparation guide

Use this as a study guide, then explain the project in your own words. The answers
describe the current code and executed run. They are prompts for understanding, not
claims about personal work.

## Module walkthrough

- `config.py` loads relative paths and settings from `config.yaml`, so the project can
  be moved without editing absolute paths.
- `data.py` reads the two semicolon-separated UCI files, adds their source type,
  checks the expected columns and target, and removes exact duplicate rows.
- `features.py` builds a `ColumnTransformer`. Numeric columns get median imputation
  and standard scaling; `wine_type` gets most-frequent imputation and one-hot encoding.
- `models.py` makes the seeded train/test split, builds the models, runs training-fold
  cross-validation and grid search, then evaluates the chosen model on the test rows.
- `evaluation.py` calculates regression metrics and writes results. `prediction.py`
  validates one JSON example and uses the saved fitted pipeline. `cli.py` connects these
  functions to `train`, `evaluate` and `predict` commands.
- `eda.py` saves the descriptive plots. The notebook uses the package functions so its
  modelling logic stays consistent with the command-line workflow.
- `tests/` contains unit and workflow tests for data validation, transforms, training,
  saved-model prediction, the CLI and generated exploratory plots.

## Questions a professor may ask

1. **Why did you choose regression?** The assignment defines the task as predicting
   numerical `quality`. We keep that target and use regression metrics.
2. **What do the rows represent?** Each row is a wine sample with physicochemical
   measurements and an expert quality score.
3. **How many rows are there?** The raw files contain 6,497 rows. Removing 1,177
   duplicate rows after adding wine type leaves 5,320 rows.
4. **What is the target?** `quality`, the score assigned to a wine sample.
5. **What is `wine_type`?** It is a categorical red/white label derived from which
   original CSV supplied the row.
6. **Were there missing values?** The two local source files had zero missing cells.
   The pipeline still includes imputation so it can handle missing predictor values.
7. **Why remove duplicate rows?** Identical observations could otherwise appear in
   both train and test partitions and make evaluation too optimistic. Repeated real
   samples may still be possible, which is a limitation of exact-value deduplication.
8. **Why do a train/test split?** Training data supports fitting and model choice. The
   test data gives one final estimate on rows not used in that process.
9. **Why is the split done before preprocessing?** This keeps the test observations
   out of fitted preprocessing statistics such as medians and means.
10. **Why put preprocessing in a Pipeline?** During cross-validation, each fold learns
    its own imputation and scaling from that fold's training portion only.
11. **Why scale numeric predictors?** k-NN and LinearSVR use distances or margins that
    depend on feature scales. Standardization puts numeric measurements on comparable
    scales.
12. **Why one-hot encode wine type?** The model needs numeric inputs, and the red/white
    label has no natural numeric order.
13. **What does DummyRegressor do?** It predicts the training target mean for every
    row. It is a simple reference for whether learned predictors help.
14. **Which models did you compare?** DummyRegressor, LinearRegression, Ridge, Lasso,
    LinearSVR and KNeighborsRegressor.
15. **Why Ridge and Lasso?** They add penalties to linear regression coefficients.
    This can reduce coefficient size and help control overfitting.
16. **What is k-NN regression?** It finds nearby training observations in the scaled
    feature space and combines their target values. Here closer neighbours receive
    more weight.
17. **How does five-fold cross-validation work?** The training split is divided into
    five parts. Each part takes a turn as validation data while the other four fit the
    model.
18. **What did GridSearchCV tune?** It compared specified parameter combinations, such
    as the number of neighbours and weighting rule for k-NN, using the same folds.
19. **How was the final model selected?** The lowest mean cross-validation RMSE ranked
    the candidates. The test set did not choose the model or parameters.
20. **What model won?** KNeighborsRegressor, with 21 neighbours and distance weighting.
21. **What does MAE 0.534 mean?** The average absolute difference between predicted
    and observed quality on the test rows was about 0.534 score points.
22. **What does RMSE 0.685 mean?** It is the square root of the average squared test
    error, expressed in score points. Larger errors affect it more than MAE.
23. **What does R² 0.375 mean?** On this test set, the model reduced squared error by
    about 37.5% relative to predicting the test mean. It still leaves much variation.
24. **Did the model beat the baseline?** Yes. On the same test split, RMSE fell from
    0.868 for the mean DummyRegressor to 0.685 for the selected model.
25. **Why is the test set used only once?** Repeatedly adjusting based on test results
    would turn it into another validation set and bias the reported estimate.
26. **What is permutation importance?** It shuffles one test feature and measures the
    reduction in the model's score. A larger drop means the fitted model relies more
    on that feature for these rows.
27. **Does alcohol cause better quality because it ranked first?** No. This is a
    predictive association. The analysis is observational and cannot show causality.
28. **Why save a joblib file?** It stores the fitted preprocessing and model together
    so the prediction command can load the same workflow later.
29. **What does the prediction input look like?** A JSON object with the numeric
    physicochemical fields and `wine_type` set to `red` or `white`.
30. **What are the main limitations?** The ordinal score is modeled numerically, the
    dataset only covers Vinho Verde wines, and important context such as price, grape
    variety, producer and tasting conditions is absent.
31. **What tests exist?** Tests cover valid and invalid data, preprocessing output,
    model training and selection, prediction input and persistence, CLI commands,
    EDA outputs and an end-to-end workflow.
32. **What remains for you to do?** Read the modules and assignment rubric, reproduce
    the run in your environment, make personal edits you understand, and record your
    real contributions and Git history. Do not claim unperformed work.

## Short oral summary

“I predicted the Wine Quality score as a regression problem. I combined the red and
white UCI files and added wine type, then removed exact duplicates. After an 80/20
split, a scikit-learn Pipeline handled imputation, scaling and encoding inside
five-fold cross-validation. I compared a mean baseline and several regression models
and selected k-NN by training-fold RMSE. On the untouched test split it scored RMSE
0.685, MAE 0.534 and R² 0.375. It outperformed the mean baseline, though substantial
error remains, and the dataset does not support causal conclusions.”
