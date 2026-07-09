from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd


@dataclass
class ValidationIssue:
    check_name: str
    severity: str  # "error" or "warning"
    message: str
    count: int
    sample_values: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    passed: bool
    errors: List[ValidationIssue]
    warnings: List[ValidationIssue]
    stats: dict
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]
    cleaned_df: pd.DataFrame
