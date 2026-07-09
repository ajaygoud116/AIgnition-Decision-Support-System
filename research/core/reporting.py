import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


class ExperimentReporter:
    """Saves experiment results to CSV, JSON, and markdown summaries."""

    def __init__(self, output_dir: Path):
        self._output_dir = Path(output_dir)
        self._tables_dir = self._output_dir / "tables"
        self._reports_dir = self._output_dir / "reports"
        self._tables_dir.mkdir(parents=True, exist_ok=True)
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    def save_metrics_table(self, results: Dict[str, Dict[str, float]],
                           name: str) -> Path:
        """Save a dict-of-dicts as a CSV table."""
        rows = []
        for model, metrics in results.items():
            row = {"model": model}
            row.update(metrics)
            rows.append(row)
        df = pd.DataFrame(rows)
        path = self._tables_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        return path

    def save_ranking_table(self, rankings: Dict[str, Dict], name: str) -> Path:
        rows = []
        for model, ranks in rankings.items():
            row = {"model": model}
            row.update(ranks)
            rows.append(row)
        df = pd.DataFrame(rows)
        df = df.sort_values("avg_rank")
        path = self._tables_dir / f"{name}_ranking.csv"
        df.to_csv(path, index=False)
        return path

    def save_significance_table(self, pairs: List[dict], name: str) -> Path:
        df = pd.DataFrame(pairs)
        path = self._tables_dir / f"{name}_significance.csv"
        df.to_csv(path, index=False)
        return path

    def save_summary(self, data: dict, name: str) -> Path:
        path = self._reports_dir / f"{name}_summary.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def save_dataframe(self, df: pd.DataFrame, name: str) -> Path:
        path = self._tables_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        return path

    def write_report_section(self, name: str, content: str) -> Path:
        path = self._reports_dir / f"{name}.md"
        with open(path, "w") as f:
            f.write(content)
        return path
