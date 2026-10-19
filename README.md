# Airbnb listing-rating prediction: a reproducible learning project

This is a clean, personal learning refactor of the team capstone **[Airbnb Rating Prediction — Data Science Capstone (NEBIUS)](https://github.com/vladlanda/capstone-nebius)**, completed during the Nebius Academy Data Science program.

The original repository is the team’s work and retains credit for that project. My contribution there focused mainly on exploratory data analysis and model experimentation. This repository does **not** claim sole authorship of the original project. Instead, it is an independent, smaller implementation I am building to understand, reproduce, and explain the full machine-learning workflow end to end.

## Goal

Predict `review_scores_rating` from structured Airbnb listing attributes. The project deliberately favours a small, inspectable pipeline over maximum performance.

## Methodology

1. Put one or more source CSV files containing `review_scores_rating` in `data/raw/`.
2. `preprocess` performs only deterministic type cleaning and makes a fixed 70/15/15 train/validation/test split.
3. `train` fits a mean baseline, Ridge, XGBoost, and CatBoost candidates on the training split, selecting the lowest **validation RMSE**. Imputation, scaling, and one-hot encoding are fitted within each model pipeline on training rows only.
4. The chosen configuration is refit on train + validation data.
5. `evaluate` is then run once on the untouched test split. It never performs selection or tuning.

Every model uses RMSE, MAE, and R². Lower RMSE/MAE is better; higher R² is better.

## Reproducible workflow

```bash
uv sync

# Place source CSVs in data/raw/ first. The original team data are not committed here.
uv run python -m src.preprocess
uv run python -m src.train
uv run python -m src.evaluate

# Robustness check: fit the fixed prospective F4 model on one city and
# validate it on the other. This does not read the project's test split.
uv run python -m src.city_validation

# Optional: a fixed, 20-trial validation-only XGBoost search.
# It does not read test.csv.
uv run python -m src.tune --trials 20

# Explore executed metrics, candidate settings, and Optuna trials locally.
uv run streamlit run app.py

# Inference for a new, unlabeled listing CSV
uv run python -m src.predict --input data/new_listings.csv --output results/predictions.csv

# Optional checks
uv run pytest
```

`data/raw/`, processed split files, fitted models, and generated result files are ignored by Git. This keeps the repository lightweight and avoids redistributing source data whose terms should be checked separately.

## Executed results

The workflow above was executed using 58,259 labeled listings from the Los Angeles and New York raw CSVs present in the original team repository. After rows without a rating were excluded, the fixed split contained 40,781 training rows, 8,739 validation rows, and 8,739 test rows.

Model comparison used the validation split only. Each non-baseline family had two deliberately small candidate configurations; the best configuration per family is shown below. The full experiment selected the XGBoost configuration with 400 trees, depth 6, learning rate 0.03, and subsample 0.8.

| Model (best candidate) | Validation RMSE | Validation MAE | Validation R² |
| --- | ---: | ---: | ---: |
| Mean baseline | 0.4433 | 0.2520 | -0.0001 |
| Ridge (alpha=10) | 0.4253 | 0.2370 | 0.0796 |
| XGBoost (selected) | 0.4067 | 0.2231 | 0.1582 |
| CatBoost | 0.4070 | 0.2239 | 0.1571 |

The selected XGBoost model was then refit once on training + validation rows and evaluated once on the held-out test set:

| Final model | Test RMSE | Test MAE | Test R² |
| --- | ---: | ---: | ---: |
| XGBoost (refit after selection) | 0.4224 | 0.2244 | 0.1428 |

The final-test score is weaker than the validation score, a useful reminder that one validation split is an estimate rather than a guarantee. The test-evaluation command refuses to overwrite existing final-test metrics, helping preserve that boundary during future experimentation.

Subsequent feature experiments deliberately did **not** reuse that test set. The current prospective candidate is F4: price/capacity features, amenities, training-only top-50 amenity flags, and missingness indicators, all **without** `host_is_superhost`.

A seeded 20-trial Optuna search varied XGBoost tree count, depth, learning rate, row/column sampling, child weight, and L1/L2 regularisation. Its best validation trial achieved **RMSE 0.4055, MAE 0.2230, R² 0.1633**, versus 0.4067 for the original F4 candidate. This is a small validation-only improvement and **not** a new final-test claim. The exact trial settings and all candidates are available in the local Streamlit dashboard and [published validation summary](docs/published_validation_results.csv).

## Cross-city robustness check

The same fixed prospective F4/XGBoost configuration was also trained on one whole city and evaluated on the other, without reading the project test split.

| Training city | Validation city | RMSE | MAE | R² |
| --- | --- | ---: | ---: | ---: |
| Los Angeles | New York | 0.4751 | 0.2994 | -0.0538 |
| New York | Los Angeles | 0.4247 | 0.2258 | 0.0194 |

This is materially harder than the random split and shows that the current features do not transfer reliably between these cities. It is a robustness check, not a replacement for a properly held-out City-3 benchmark.

## Experiment record and EDA

- [Experiment log and decision record](docs/experiment_log.md) distinguishes executed results from proposed feature experiments, records F1–F4 and the city-held-out check.
- [Focused EDA findings](docs/eda_findings.md) records training-only data-quality and error-analysis signals behind the next experiment.

The F1–F4 feature experiments and a Superhost ablation are documented in the log. They used validation only: the existing test result belongs to the original Run 0 benchmark and will not be reused for later feature-set claims.

## Limitations

- Rating scores are bounded and often concentrated near the upper end, which makes improvements hard to interpret.
- The random split estimates in-distribution performance, not performance for new cities or future listings.
- Only basic structured listing features are used; free text, images, and neighbourhood context are intentionally excluded to keep the project explainable.
- Results depend on the particular source-data snapshot and its collection process.

## Lessons learned

- A held-out test set is for one final estimate, not for iterative decisions.
- Transformations such as imputation and encoding belong inside the fitted model pipeline.
- A simple baseline is essential context for an apparently strong model.
- A small, reproducible experiment is more valuable for learning than a complex system that cannot be explained end to end.

## Future work

- Compare random splits with city-based or time-based evaluation.
- Add transparent error analysis by room type, price band, and city.
- Document source-data licensing and provenance more precisely.
- Add model explanations only if they improve understanding more than they add complexity.
