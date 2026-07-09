from typing import Callable, List, Optional, Tuple

import pandas as pd

from src.models.validation import ValidationIssue, ValidationReport
from src.utils.config import Config
from src.utils.logger import StructuredLogger

from .checks import (
    CheckOutput,
    check_budget_consistency,
    check_channel_coverage,
    check_duplicate_rows,
    check_future_dates,
    check_invalid_campaign_types,
    check_missing_values,
    check_negative_spend,
    check_zero_activity,
)


class ValidationEngine:
    """Orchestrates all validation checks against a unified DataFrame.

    Each check is a pure function that receives the DataFrame and returns
    (issues, cleaned_df). The engine chains them in order, accumulating
    issues and passing the modified DataFrame between checks.

    The final ValidationReport contains all issues, summary stats, and
    the cleaned DataFrame ready for feature engineering.
    """

    def __init__(self, config: Optional[Config] = None):
        self._config = config or Config()
        self._logger = StructuredLogger("validation.engine")
        self._checks: List[Tuple[str, Callable, dict]] = [
            (
                "missing_values",
                check_missing_values,
                {"max_ratio": self._config.get("validation.max_missing_ratio", 0.3)},
            ),
            (
                "negative_spend",
                check_negative_spend,
                {},
            ),
            (
                "duplicate_rows",
                check_duplicate_rows,
                {
                    "max_duplicate_ratio": self._config.get(
                        "validation.max_duplicate_ratio", 0.05
                    )
                },
            ),
            (
                "future_dates",
                check_future_dates,
                {
                    "max_future_days": self._config.get(
                        "validation.max_future_days", 7
                    )
                },
            ),
            (
                "invalid_campaign_types",
                check_invalid_campaign_types,
                {
                    "valid_types": self._config.get(
                        "validation.valid_campaign_types", {}
                    )
                },
            ),
            (
                "budget_consistency",
                check_budget_consistency,
                {},
            ),
            (
                "zero_activity",
                check_zero_activity,
                {},
            ),
            (
                "channel_coverage",
                check_channel_coverage,
                {"expected_channels": ["google", "meta", "bing"]},
            ),
        ]

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """Run all checks in sequence and produce a ValidationReport."""
        original_shape = df.shape
        all_errors: List[ValidationIssue] = []
        all_warnings: List[ValidationIssue] = []
        working_df = df.copy()

        self._logger.info("validation_started", rows=len(working_df), columns=len(working_df.columns))

        for name, check_fn, kwargs in self._checks:
            try:
                issues, working_df = check_fn(working_df, **kwargs)
            except Exception as exc:
                self._logger.error(
                    "check_failed",
                    check=name,
                    exception=str(exc),
                )
                issues = [
                    ValidationIssue(
                        check_name=name,
                        severity="error",
                        message=f"Check raised an exception: {exc}",
                        count=0,
                    )
                ]

            for issue in issues:
                if issue.severity == "error":
                    all_errors.append(issue)
                else:
                    all_warnings.append(issue)

            self._logger.info(
                "check_complete",
                check=name,
                errors=sum(1 for i in issues if i.severity == "error"),
                warnings=sum(1 for i in issues if i.severity == "warning"),
            )

        final_shape = working_df.shape
        passed = len(all_errors) == 0

        stats = {
            "original_rows": original_shape[0],
            "original_columns": original_shape[1],
            "final_rows": final_shape[0],
            "final_columns": final_shape[1],
            "rows_removed": original_shape[0] - final_shape[0],
            "total_errors": len(all_errors),
            "total_warnings": len(all_warnings),
            "channels_present": sorted(working_df["channel"].unique().tolist())
            if "channel" in working_df.columns
            else [],
            "date_range": (
                f"{working_df['date'].min()} to {working_df['date'].max()}"
                if "date" in working_df.columns
                else "N/A"
            ),
        }

        self._logger.info(
            "validation_complete",
            passed=passed,
            errors=len(all_errors),
            warnings=len(all_warnings),
            rows_removed=stats["rows_removed"],
        )

        return ValidationReport(
            passed=passed,
            errors=all_errors,
            warnings=all_warnings,
            stats=stats,
            original_shape=original_shape,
            final_shape=final_shape,
            cleaned_df=working_df,
        )
