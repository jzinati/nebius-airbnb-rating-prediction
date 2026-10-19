"""Small model catalogue and a single preprocessing pipeline."""
from __future__ import annotations

from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.settings import RANDOM_SEED


def feature_transformer(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformers = [
        ("numeric", numeric, numeric_features),
        ("categorical", categorical, categorical_features),
    ]
    return ColumnTransformer(transformers)


def build_model(
    model_name: str,
    params: dict,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    if model_name == "baseline":
        return Pipeline([("model", DummyRegressor(strategy="mean"))])
    if model_name == "ridge":
        estimator = Ridge(random_state=RANDOM_SEED, **params)
    elif model_name == "xgboost":
        estimator = XGBRegressor(
            objective="reg:squarederror", random_state=RANDOM_SEED, n_jobs=-1, **params
        )
    elif model_name == "catboost":
        estimator = CatBoostRegressor(
            loss_function="RMSE", random_seed=RANDOM_SEED, verbose=False,
            allow_writing_files=False, thread_count=-1, **params
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([
        ("features", feature_transformer(numeric_features, categorical_features)),
        ("model", estimator),
    ])


def candidate_models(feature_set_name: str = "base") -> list[tuple[str, dict]]:
    """A deliberately tiny search space, selected solely by validation RMSE."""
    return [
        ("baseline", {}),
        ("ridge", {"alpha": 0.1}),
        ("ridge", {"alpha": 10.0}),
        ("xgboost", {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8}),
        ("xgboost", {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.03, "subsample": 0.8}),
        ("catboost", {"iterations": 300, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3.0}),
        ("catboost", {"iterations": 500, "depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 5.0}),
    ]
