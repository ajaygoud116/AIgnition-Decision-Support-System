from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.ingestion.base import AbstractIngestor
from src.ingestion.google_ads import GoogleAdsIngestor
from src.ingestion.meta_ads import MetaAdsIngestor
from src.ingestion.bing_ads import BingAdsIngestor
from src.utils.errors import IngestionError
from src.utils.io import discover_csv_files, read_csv_safe
from src.utils.logger import StructuredLogger


class IngestionPipeline:
    """Orchestrates multi-platform CSV ingestion.

    Discovers all CSV files in a directory, auto-detects the platform using
    registered ingesters, normalizes each into the unified schema, and
    concatenates into a single DataFrame.

    New platform ingesters can be added by subclassing AbstractIngestor and
    registering them via add_ingestor() or passing them to the constructor.
    """

    def __init__(self, ingesters: Optional[List[AbstractIngestor]] = None):
        self._ingesters = ingesters or [
            GoogleAdsIngestor(),
            MetaAdsIngestor(),
            BingAdsIngestor(),
        ]
        self._logger = StructuredLogger("ingestion.pipeline")

    def add_ingestor(self, ingestor: AbstractIngestor) -> None:
        self._ingesters.append(ingestor)

    def ingest_all(self, data_dir: Path) -> pd.DataFrame:
        """Read all CSV files in data_dir, auto-detect platform, normalize.

        Returns a concatenated DataFrame with unified schema.
        Raises IngestionError if no files found or no ingestor matches a file.
        """
        csv_files = discover_csv_files(data_dir)
        self._logger.info("discovered_csv_files", count=len(csv_files))

        unified_frames: List[pd.DataFrame] = []

        for csv_path in csv_files:
            self._logger.info("processing_file", path=str(csv_path))
            raw = read_csv_safe(csv_path)
            ingestor = self._match_ingestor(raw)

            if ingestor is None:
                raise IngestionError(
                    f"No ingestor found for {csv_path.name}. "
                    f"Columns detected: {list(raw.columns)}"
                )

            self._logger.info(
                "matched_ingestor",
                file=csv_path.name,
                ingestor=type(ingestor).__name__,
            )

            unified = ingestor.normalize(raw)
            self._logger.info(
                "normalized",
                file=csv_path.name,
                records=len(unified),
                columns=list(unified.columns),
            )

            unified_frames.append(unified)

        if not unified_frames:
            raise IngestionError("No data was ingested from any CSV file.")

        result = pd.concat(unified_frames, ignore_index=True)
        self._logger.info(
            "ingestion_complete",
            total_records=len(result),
            channels=sorted(result["channel"].unique().tolist()),
            date_range=f"{result['date'].min()} to {result['date'].max()}",
        )

        return result

    def _match_ingestor(self, raw: pd.DataFrame) -> Optional[AbstractIngestor]:
        """Return the first ingestor that claims it can handle this DataFrame."""
        for ingestor in self._ingesters:
            if ingestor.can_handle(raw):
                return ingestor
        return None
