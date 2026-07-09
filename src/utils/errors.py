class ForecastError(Exception):
    """Base exception for all forecast system errors."""


class IngestionError(ForecastError):
    """Raised when data ingestion fails."""


class ValidationError(ForecastError):
    """Raised when data validation fails."""


class FeatureComputationError(ForecastError):
    """Raised when feature computation fails."""


class ModelError(ForecastError):
    """Raised when model loading or prediction fails."""


class SimulationError(ForecastError):
    """Raised when scenario simulation fails."""


class ConfigurationError(ForecastError):
    """Raised when configuration is invalid or missing."""
