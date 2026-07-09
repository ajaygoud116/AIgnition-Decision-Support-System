from typing import List, Optional, Tuple

import pandas as pd

from src.models.validation import ValidationIssue


CheckOutput = Tuple[List[ValidationIssue], pd.DataFrame]


def check_missing_values(
    df: pd.DataFrame,
    max_ratio: float = 0.3,
    max_fraction_dropped: float = 0.3,
) -> CheckOutput:
    """Detect columns with missing value ratio above threshold.

    Issues a warning per problematic column. Does not drop rows.
    """
    issues: List[ValidationIssue] = []
    result = df.copy()

    for col in result.columns:
        null_count = result[col].isna().sum()
        if null_count == 0:
            continue
        ratio = null_count / len(result)
        if ratio > max_ratio:
            sample = result.loc[result[col].isna(), col].index[:5].tolist()
            issues.append(
                ValidationIssue(
                    check_name="missing_values",
                    severity="error",
                    message=(
                        f"Column '{col}' has {null_count} missing values "
                        f"({ratio:.1%}), exceeding {max_ratio:.0%} threshold."
                    ),
                    count=null_count,
                    sample_values=[str(i) for i in sample],
                )
            )
        elif ratio > 0:
            sample = result.loc[result[col].isna(), col].index[:3].tolist()
            issues.append(
                ValidationIssue(
                    check_name="missing_values",
                    severity="warning",
                    message=(
                        f"Column '{col}' has {null_count} missing values ({ratio:.1%})."
                    ),
                    count=null_count,
                    sample_values=[str(i) for i in sample],
                )
            )

    return issues, result


def check_negative_spend(df: pd.DataFrame) -> CheckOutput:
    """Detect negative spend values and clamp them to zero.

    Raises a warning for each negative spend occurrence.
    """
    issues: List[ValidationIssue] = []
    result = df.copy()

    if "spend" not in result.columns:
        return issues, result

    neg_mask = result["spend"] < 0
    neg_count = neg_mask.sum()
    if neg_count > 0:
        sample = result.loc[neg_mask, "spend"].head(5).tolist()
        result.loc[neg_mask, "spend"] = 0.0
        issues.append(
            ValidationIssue(
                check_name="negative_spend",
                severity="warning",
                message=(
                    f"Found {neg_count} rows with negative spend. "
                    "Values have been clamped to 0.0."
                ),
                count=neg_count,
                sample_values=[str(v) for v in sample],
            )
        )

    return issues, result


def check_duplicate_rows(
    df: pd.DataFrame,
    max_duplicate_ratio: float = 0.05,
) -> CheckOutput:
    """Detect fully duplicate rows and remove them.

    Errors if duplicate ratio exceeds threshold, otherwise warns.
    """
    issues: List[ValidationIssue] = []
    result = df.copy()

    if result.empty:
        return issues, result

    dup_mask = result.duplicated(keep="first")
    dup_count = dup_mask.sum()
    if dup_count == 0:
        return issues, result

    dup_ratio = dup_count / len(result)
    result = result[~dup_mask].copy()

    severity = "error" if dup_ratio > max_duplicate_ratio else "warning"
    issues.append(
        ValidationIssue(
            check_name="duplicate_rows",
            severity=severity,
            message=(
                f"Found and removed {dup_count} duplicate rows "
                f"({dup_ratio:.1%} of data)."
            ),
            count=dup_count,
            sample_values=[str(i) for i in range(min(5, dup_count))],
        )
    )

    return issues, result


def check_future_dates(
    df: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
    max_future_days: int = 7,
) -> CheckOutput:
    """Detect dates in the future relative to reference_date.

    Default reference_date is the maximum date in the dataset + 1 day
    (treating the dataset's own max as 'today').
    Campaign-days with future dates beyond max_future_days are flagged as errors;
    within max_future_days they are warnings.
    """
    issues: List[ValidationIssue] = []
    result = df.copy()

    if "date" not in result.columns or result.empty:
        return issues, result

    if reference_date is None:
        reference_date = result["date"].max() + pd.Timedelta(days=1)

    future_mask = result["date"] > reference_date
    future_count = future_mask.sum()
    if future_count == 0:
        return issues, result

    future_dates = df.loc[future_mask, "date"].head(5).tolist()
    result = result[~future_mask].copy()

    severity = "error" if future_count > max_future_days else "warning"
    issues.append(
        ValidationIssue(
            check_name="future_dates",
            severity=severity,
            message=(
                f"Found {future_count} rows with dates after {reference_date.date()}. "
                "These rows have been excluded."
            ),
            count=future_count,
            sample_values=[str(d) for d in future_dates],
        )
    )

    return issues, result


def check_invalid_campaign_types(
    df: pd.DataFrame,
    valid_types: Optional[dict] = None,
) -> CheckOutput:
    """Flag campaign_type values not in the known set for their channel.

    Valid types are provided as a dict: {"google": [...], "meta": [...], "bing": [...]}.
    Unknown types are flagged as warnings for investigation.
    """
    issues: List[ValidationIssue] = []
    result = df.copy()

    if "campaign_type" not in result.columns or "channel" not in result.columns:
        return issues, result

    if valid_types is None:
        return issues, result

    for channel, valid in valid_types.items():
        channel_mask = result["channel"] == channel
        if not channel_mask.any():
            continue

        channel_types = result.loc[channel_mask, "campaign_type"]
        invalid_mask = ~channel_types.isin(valid)
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            invalid_values = channel_types[invalid_mask].unique().tolist()
            issues.append(
                ValidationIssue(
                    check_name="invalid_campaign_types",
                    severity="warning",
                    message=(
                        f"Channel '{channel}' has {invalid_count} rows with "
                        f"unrecognized campaign types: {invalid_values}. "
                        "These may indicate new campaign structures."
                    ),
                    count=invalid_count,
                    sample_values=[str(v) for v in invalid_values[:5]],
                )
            )

    return issues, result


def check_budget_consistency(df: pd.DataFrame) -> CheckOutput:
    """Detect rows where spend exceeds daily_budget.

    This is a warning — budgets can be overridden or capped differently.
    """
    issues: List[ValidationIssue] = []
    result = df.copy()

    if "spend" not in result.columns or "daily_budget" not in result.columns:
        return issues, result

    budget_mask = (result["daily_budget"] > 0) & (result["spend"] > result["daily_budget"])
    over_count = budget_mask.sum()
    if over_count > 0:
        sample = result.loc[budget_mask, ["spend", "daily_budget"]].head(5)
        ratio = over_count / len(result)
        issues.append(
            ValidationIssue(
                check_name="budget_consistency",
                severity="warning",
                message=(
                    f"Found {over_count} rows ({ratio:.1%}) where spend exceeds "
                    "daily budget. This may indicate budget changes not captured in data."
                ),
                count=over_count,
                sample_values=[
                    f"spend={row['spend']}, budget={row['daily_budget']}"
                    for _, row in sample.iterrows()
                ],
            )
        )

    return issues, result


def check_zero_activity(df: pd.DataFrame) -> CheckOutput:
    """Detect campaign-days with zero spend AND zero impressions.

    These rows provide no signal and should be reviewed.
    """
    issues: List[ValidationIssue] = []
    result = df.copy()

    if "spend" not in result.columns or "impressions" not in result.columns:
        return issues, result

    zero_mask = (result["spend"] == 0) & (result["impressions"] == 0)
    zero_count = zero_mask.sum()
    if zero_count > 0:
        ratio = zero_count / len(result)
        sample = (
            result.loc[zero_mask, ["date", "campaign_id", "channel"]]
            .head(5)
            .apply(lambda r: f"{r['date'].date()}/{r['campaign_id']}", axis=1)
            .tolist()
        )
        issues.append(
            ValidationIssue(
                check_name="zero_activity",
                severity="warning",
                message=(
                    f"Found {zero_count} rows ({ratio:.1%}) with zero spend and "
                    "zero impressions. These provide no signal."
                ),
                count=zero_count,
                sample_values=sample,
            )
        )

    return issues, result


def check_channel_coverage(df: pd.DataFrame, expected_channels: Optional[List[str]] = None) -> CheckOutput:
    """Verify that all expected channels are present in the data.

    Missing channels are flagged as errors — the forecast cannot proceed
    for absent channels.
    """
    issues: List[ValidationIssue] = []
    result = df.copy()

    if expected_channels is None:
        expected_channels = ["google", "meta", "bing"]

    if "channel" not in result.columns:
        issues.append(
            ValidationIssue(
                check_name="channel_coverage",
                severity="error",
                message="No 'channel' column found in data.",
                count=0,
            )
        )
        return issues, result

    present = result["channel"].unique()
    for ch in expected_channels:
        if ch not in present:
            issues.append(
                ValidationIssue(
                    check_name="channel_coverage",
                    severity="error",
                    message=(
                        f"Channel '{ch}' is missing from the data. "
                        "Forecasts for this channel will not be generated."
                    ),
                    count=0,
                    sample_values=[ch],
                )
            )

    if not issues:
        issues.append(
            ValidationIssue(
                check_name="channel_coverage",
                severity="warning",
                message=f"All expected channels present: {expected_channels}.",
                count=0,
            )
        )

    return issues, result
