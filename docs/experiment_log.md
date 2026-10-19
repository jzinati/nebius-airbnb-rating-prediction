# Experiment log and decision record

This document separates evidence from plans. A row is marked **executed** only when its metrics were produced by this repository's code. Planned work must not be described as a result.

## Rules for every experiment

- The train/validation/test split created with seed 42 is fixed for this project version.
- Fit imputers, encoders, and models on training data only during selection.
- Choose models and feature sets using validation RMSE; report MAE and R² for context.
- Do not inspect the test metrics while choosing a feature set or hyperparameters.
- Once a candidate is selected, refit on train + validation and evaluate the test set once.
- Record the feature set, parameters, data version, and all three metrics.

## Run 0 — structured-feature benchmark

**Status:** executed

**Question:** Can basic structured listing attributes predict `review_scores_rating` better than the training-set mean?

**Data:** 58,259 labelled Los Angeles and New York listings from the raw CSVs used in the original team project. The deterministic split contains 40,781 training rows, 8,739 validation rows, and 8,739 test rows.

**Feature set:** price; location coordinates; capacity and stay constraints; estimated occupancy; host listing counts, response rates, and tenure; property type; room type; response-time band; superhost flag; instant-bookable flag. Missing values are imputed inside each fitted model pipeline.

**Selection results (validation only):**

| Model | Best tested configuration | RMSE | MAE | R² |
| --- | --- | ---: | ---: | ---: |
| Mean baseline | Predict the training mean | 0.4433 | 0.2520 | -0.0001 |
| Ridge | `alpha=10` | 0.4253 | 0.2370 | 0.0796 |
| XGBoost | 400 trees, depth 6, learning rate 0.03 | 0.4067 | 0.2231 | 0.1582 |
| CatBoost | 500 iterations, depth 8, learning rate 0.03 | 0.4070 | 0.2239 | 0.1571 |

**Decision:** select XGBoost by lowest validation RMSE. The advantage over CatBoost is only 0.0003 RMSE, so it should be treated as a small difference, not proof that XGBoost is universally better.

**Final estimate:** the selected XGBoost configuration was refit on train + validation and evaluated once on test: **RMSE 0.4224, MAE 0.2244, R² 0.1428**.

**What Run 0 establishes:** the nonlinear tree models provide a meaningful improvement over the mean baseline and Ridge. It does not establish that the model generalizes to another city, a future period, or another data source.

## Run F1 — price and capacity transformations

**Status:** executed on validation only; no new test evaluation was performed.

**Question:** Do simple, listing-time transformations make the relationship between price, stay constraints, and capacity easier for the models to learn?

**Feature additions:** `log_price`, `log_minimum_nights`, `log_maximum_nights`, `log_host_listings_count`, price-per-guest, beds-per-guest, bedrooms-per-guest, and bathrooms-per-guest. Ratios with zero or missing capacity are set to missing and imputed by the fitted pipeline. These variables use no target, review count, or text-derived information.

**Validation results:**

| Comparison | Best model | RMSE | MAE | R² | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| Run 0 base features, with Superhost | XGBoost | 0.4067 | 0.2231 | 0.1582 | Original selected benchmark. |
| F1 price/capacity, with Superhost | XGBoost | 0.4060 | 0.2231 | 0.1612 | Small improvement of 0.0007 RMSE. |
| Run 0 base features, without Superhost | XGBoost | 0.4095 | 0.2275 | 0.1464 | Isolates the current flag's predictive contribution. |
| F1 price/capacity, without Superhost | XGBoost | 0.4081 | 0.2268 | 0.1524 | Recovers 0.0014 RMSE without the flag. |

**Decision:** retain F1 as the next explainable feature set. The gain is modest, so it is evidence to continue carefully, not a reason to add complexity quickly.

## Superhost feature assessment

**Status:** investigated and ablated.

`host_is_superhost` is predictive in this snapshot: removing it from the base feature set worsened XGBoost validation RMSE from 0.4067 to 0.4095. That predictive value is also the reason to be cautious.

Airbnb requires Superhosts to maintain a 4.8+ overall rating and meet reservation, response, and cancellation criteria; it reassesses the status quarterly using the prior 12 months of host performance. See Airbnb's [Superhost requirements](https://www.airbnb.com/help/article/829). The feature is therefore not merely an indicator that a host has some ratings. It is a host-level status partly determined by historical ratings, which may include the same listing's rating history represented by this project's target.

For a **prospective** use case—predicting a new listing's future rating before reviews accrue—Superhost status is unavailable for new hosts and can encode post-outcome information for established hosts. We therefore classify it as **leakage-sensitive** and exclude it from the recommended prospective feature set. For a contemporaneous descriptive model, it may be acceptable if it is explicitly described as a status observed at the same snapshot; it should not be presented as an independent pre-listing predictor.

**Decision:** keep the with-Superhost benchmark as a historical comparison, but use **F1 without Superhost** as the defensible candidate for future work. The cost is visible and documented: validation RMSE 0.4081 rather than 0.4060.

## Run F2 — fixed amenity flags

**Status:** executed on validation only; no new test evaluation was performed.

F2 adds amenity count plus fixed, listing-time flags for Wi-Fi, kitchen, air conditioning, parking, laundry, self check-in, workspace, safety equipment, and elevator.

| Variant | Best model | Validation RMSE | Validation MAE | Validation R² |
| --- | --- | ---: | ---: | ---: |
| F2 with Superhost | XGBoost | 0.4048 | 0.2226 | 0.1659 |
| F2 without Superhost | XGBoost | 0.4082 | 0.2268 | 0.1520 |

**Decision:** fixed amenity flags slightly improve the historical with-Superhost benchmark but do not improve the prospective F1-without-Superhost control. Do not promote F2 alone.

## Run F3 — training-only top-50 amenities

**Status:** executed on validation only; no new test evaluation was performed.

F3 adds 50 binary amenity flags. The vocabulary is learned from training rows only and is saved with the experiment artifact, then applied unchanged to validation rows. This avoids using validation frequency information.

| Variant | Best model | Validation RMSE | Validation MAE | Validation R² |
| --- | --- | ---: | ---: | ---: |
| F3 with Superhost | XGBoost | 0.4047 | 0.2229 | 0.1663 |
| F3 without Superhost | CatBoost | 0.4075 | 0.2250 | 0.1548 |

**Decision:** F3 without Superhost is the current leading prospective candidate, improving on F1 without Superhost by 0.0006 RMSE. The improvement is small and should not be oversold. F3 with Superhost is retained only as a leakage-sensitive historical benchmark.

## Run F4 — missingness indicators

**Status:** executed on validation only; no new test evaluation was performed.

F4 adds binary indicators for whether price, beds, bedrooms, bathrooms, response time, response rate, and acceptance rate were absent in the raw listing. The imputed numeric values remain in the model, but it can now distinguish an observed value from a filled-in value.

| Variant | Best model | Validation RMSE | Validation MAE | Validation R² |
| --- | --- | ---: | ---: | ---: |
| F4 with Superhost | XGBoost | 0.4035 | 0.2199 | 0.1715 |
| F4 without Superhost | XGBoost | 0.4067 | 0.2237 | 0.1583 |

**Decision:** retain F4 as the leading prospective candidate. It improves F3 without Superhost by 0.0009 RMSE and confirms that listing completeness is useful information. The with-Superhost figure is a historical benchmark only.

## Test-set integrity after Run 0

The Run 0 test metric has already been observed. It remains a valid report for Run 0, but it must not be reused to claim final performance for F1 or later feature sets. Later experiments in this log use validation only.

To make a final performance claim for the eventual prospective model, obtain a new, chronologically later data snapshot or reserve a new holdout before further selection. This is more honest than repeatedly evaluating the original test file.

## Run F6 — city-held-out robustness check

**Status:** executed; the original random-split test set was not read.

**Question:** Does the fixed prospective F4 model transfer from one city to another, rather than only to randomly held-out listings from the same pooled snapshot?

**Protocol:** each city was cleaned independently. The F4 feature set used the same fixed amenity flags, while the top-50 amenity vocabulary was fitted only on the source-city training rows. The model was the fixed F4-without-Superhost XGBoost configuration (400 trees, depth 6, learning rate 0.03, subsample 0.8), with no tuning or model selection in this check.

| Training city | Validation city | Rows fitted | Rows evaluated | RMSE | MAE | R² |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Los Angeles | New York | 33,368 | 24,891 | 0.4751 | 0.2994 | -0.0538 |
| New York | Los Angeles | 24,891 | 33,368 | 0.4247 | 0.2258 | 0.0194 |

**Decision:** do not claim that the random-split improvement transfers to a new city. The model performs much worse across cities, particularly from Los Angeles to New York. This is an informative negative result: price, room type, coordinates, amenities, and missingness do not yet capture enough city-specific context. It does not reproduce or disprove the remembered City-3 competition result, because the checked-in City-3 files cannot be aligned to targets.

## Benchmark audit: is RMSE below 0.39 a comparable target?

The original team repository contains saved metrics for two relevant models. Its plain XGBoost run reported train RMSE 0.3315 and test RMSE 0.4139. Its mixture-of-experts run reported train RMSE 0.3880, validation RMSE 0.4012, and test RMSE 0.4067.

The original task scripts also contain a City-3 prediction/evaluation workflow, so a strong new-city result is plausible. However, no City-3 metric was saved in the repository. The checked-in City-3 feature file has 14,589 rows while its target file has 6,822 labels, with no stable row identifier or documented filtering rule to align them. That makes the new-city score impossible to reproduce from the checked-in artifacts alone.

Therefore, the value below 0.39 should be treated as an **unverified external-city benchmark**, not dismissed as a training result. The current F4 prospective result (0.4067 on a random validation split) is in the same range as the original team's saved random-split results, but its city-held-out results are not competitive evidence for the City-3 challenge.

**Decision:** do not optimise blindly toward a remembered threshold. Before claiming a performance improvement, recover the City-3 source, target alignment, and scoring script—or obtain a new, properly keyed unseen-city holdout. Then compare the same target scale, source snapshot, rows retained, split policy, features available at prediction time, and evaluation partition. The next legitimate improvement is a controlled coarse-location experiment; it should not use post-review variables merely to lower RMSE.

## Feature-experiment roadmap

The experiments below are intentionally ordered from lowest risk and easiest to explain to more complex. Status distinguishes completed work from plans.

| ID | Hypothesis | Additions | Keep fixed | Decision rule | Status |
| --- | --- | --- | --- | --- | --- |
| F1 | Price and capacity are nonlinear; ratios express listing value more directly. | `log_price`, `log_minimum_nights`, price-per-guest, beds-per-guest, bedrooms-per-guest, bathrooms-per-guest. | Split, candidate models, metrics. | Retained as the next feature set; prospective variant excludes Superhost. | Executed |
| F2 | Amenity availability captures practical listing quality without modelling text. | Amenity count and a fixed set of binary flags. | F1-without-Superhost feature set and selection protocol. | Not promoted: no prospective validation gain. | Executed |
| F3 | Broader amenity coverage captures listing quality without text embeddings. | Training-only top-50 amenity flags. | F2 and selection protocol. | Retained: small prospective validation gain. | Executed |
| F4 | Missingness itself may describe listing completeness or host behaviour. | Indicators for high-missingness listing-time fields. | F3-without-Superhost feature set and selection protocol. | Retained: small prospective validation gain. | Executed |
| F5 | Nearby listings tend to share market conditions and guest expectations. | A coarse, fixed location cell from rounded latitude/longitude. | No target encoding; same split. | Keep only if improvement is stable and cardinality remains easy to explain. | Planned |
| F6 | The random split may be optimistic. | City-held-out validation. | Fixed F4 prospective model; final test set untouched. | Use as a robustness check, not as a chance to keep trying models on test. | Executed; transfer was poor. |

## Explicit non-goals for the next iteration

- No mixture-of-experts model: it adds several moving parts before we have shown that simpler features have been exhausted.
- No free-text embeddings or LLM-derived features: they make the project harder to reproduce, validate, and defend.
- No target encoding in the first feature-engineering round: it is useful but requires careful out-of-fold implementation to avoid leakage.
- No broad hyperparameter search: the current small candidate grids are sufficient while testing feature hypotheses.

## Decision for the next implementation

F1–F4 and F6 have now been run. Do not add model complexity yet. First recover the competition evaluator or obtain a properly keyed City-3 target file. In parallel, F5 may be tested as a separate, explainable random-split experiment, but any gain must then be checked on a city-held-out or time-held-out dataset before it is presented as portable.

Do not combine F5 with other new feature families in one run: otherwise we will not know which change helped. The Superhost issue has been resolved for the prospective variant; the cross-city result shows that validation design—not more tuning—is the current limiting factor.

## Experiment-record template

Copy this section before each new run.

```text
ID and date:
Question / hypothesis:
Data snapshot and split seed:
Feature changes:
Models and candidate parameters:
Validation RMSE / MAE / R²:
Selection decision:
Final test metrics (only once, after selection):
Interpretation and next action:
```
