from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class ExperimentConfig:
    data_dir: Path = Path("data")
    output_dir: Path = Path("research")
    random_seed: int = 42
    horizons: List[int] = field(default_factory=lambda: [30, 60, 90])
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    min_history_days: int = 30
    lookback_window: int = 90
    rolling_windows: List[int] = field(default_factory=lambda: [7, 14, 30])
    lag_windows: List[int] = field(default_factory=lambda: [1, 7, 14, 30])
    cv_n_splits: int = 3
    cv_initial_train_days: int = 120
    cv_step_days: int = 60
    cv_forecast_horizon: int = 30
    lgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": -1,
        "random_state": 42,
    })
    benchmark_models: List[str] = field(default_factory=lambda: [
        "naive", "historical_mean", "seasonal_naive", "linear_regression",
        "random_forest", "xgboost", "lightgbm", "ensemble",
    ])
    feature_ablation_groups: List[str] = field(default_factory=lambda: [
        "rolling", "lag", "ratio", "time", "all",
    ])
