# Focused EDA findings

This EDA was run on the 40,781-row **training split only**. It does not inspect the held-out test set. The reproducible summary command is:

```bash
uv run python -m src.eda
```

## Target shape

- Mean rating: **4.7555**; standard deviation: **0.4423**.
- **31.9%** of listings have a perfect score of 5.0; only **2.64%** are below 4.0.
- The median is 4.89 and the 5th percentile is 4.00.

This ceiling-heavy distribution explains why RMSE is difficult to reduce: a model must identify a relatively small number of lower-rated listings while avoiding errors around the large mass of near-perfect scores.

## Data quality signals

Several fields are missing often enough that a missingness indicator may be informative:

- `host_response_time` and `host_response_rate`: 27.3% missing;
- `price`: 27.2% missing;
- `beds`: 26.8% missing;
- `host_acceptance_rate`: 25.5% missing;
- `bedrooms`: 10.4% missing.

The current pipeline imputes these fields but does not explicitly tell the model that a value was originally missing. Missingness is observable at listing time, so indicators are a valid next feature experiment.

## Associations worth testing, not causal claims

- The largest absolute univariate correlations with the rating are estimated occupancy (0.108), host tenure (0.103), and host listing counts (0.082–0.093). None is strong on its own, so interactions and nonlinearity matter more than a single “magic” variable.
- Mean ratings differ by room type: entire homes average 4.782, private rooms 4.707, hotel rooms 4.470, and shared rooms 4.634. Hotel and private-room listings also have the largest validation absolute errors, though the hotel sample is small.
- Higher price quartiles have somewhat higher mean ratings, but the lowest-price quartile has the largest validation absolute error (0.264 versus 0.176 in the highest quartile). Price transformations and capacity ratios are therefore justified.
- Los Angeles and New York have a modest distribution shift: mean ratings are 4.778 and 4.722 respectively. This supports a city-aware evaluation before making a new-city performance claim.

## Leakage-sensitive observation

Superhost listings average 4.867 versus 4.684 for non-Superhosts in the training data. This is a large predictive association, but it is expected because Superhost eligibility includes historical ratings. It supports excluding the field from the prospective model rather than treating it as a discovery.

## Next decision

F4 was executed and improved the prospective validation RMSE from 0.4075 to **0.4067**. The next priority is now city-held-out validation (or recovery of the competition evaluator), because random-split gains do not establish improvement on an unseen city.
