from pathlib import Path
from typing import List

import pandas as pd


def discover_csv_files(data_dir: Path) -> List[Path]:
    """Return all .csv files in data_dir, sorted by name for determinism."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return files


def read_csv_safe(path: Path) -> pd.DataFrame:
    """Read a CSV file and strip whitespace from column names.

    Handles inconsistent whitespace in raw marketing exports.
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df
