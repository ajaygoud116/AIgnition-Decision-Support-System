from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Configuration loader.

    Load order:
      1. Default values
      2. YAML file (config.yaml)
      3. Environment variable overrides (prefixed with FC_)

    The resulting config is accessed via dict-like get() or attribute access.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self._data: Dict[str, Any] = self._defaults()
        if config_path and config_path.exists():
            with open(config_path, "r") as f:
                file_config = yaml.safe_load(f) or {}
            self._deep_merge(self._data, file_config)

    @staticmethod
    def _defaults() -> dict:
        return {
            "project": {"name": "aignition-forecasting", "random_seed": 42},
            "forecast": {
                "horizons": [30, 60, 90],
                "quantiles": [0.1, 0.5, 0.9],
                "min_history_days": 30,
                "lookback_window": 90,
            },
            "features": {
                "rolling_windows": [7, 14, 30],
                "lag_windows": [1, 7, 14, 30],
                "max_na_ratio": 0.3,
            },
            "validation": {
                "max_missing_ratio": 0.3,
                "max_future_days": 7,
                "max_duplicate_ratio": 0.05,
                "valid_campaign_types": {
                    "google": ["SEARCH", "PERFORMANCE_MAX", "DISPLAY", "VIDEO", "DEMAND_GEN", "SHOPPING"],
                    "meta": ["Generic", "Prospecting", "Remarketing"],
                    "bing": ["Search", "PerformanceMax", "Audience", "Shopping"],
                },
            },
            "simulation": {"max_adjustments_per_run": 10},
            "decision": {
                "min_roas_target": 3.0,
                "volatility_threshold": 0.5,
                "diminishing_returns_periods": 3,
                "cost_inflation_threshold": 0.2,
                "concentration_threshold": 0.6,
                "zero_revenue_threshold_days": 45,
            },
            "report": {
                "include_raw_forecasts": False,
                "decimal_places": 2,
                "output_format": "csv",
            },
            "logging": {"level": "INFO", "format": "json"},
        }

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> None:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                Config._deep_merge(base[key], value)
            else:
                base[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    def as_dict(self) -> dict:
        return dict(self._data)
