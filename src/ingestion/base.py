from abc import ABC, abstractmethod

import pandas as pd


class AbstractIngestor(ABC):
    """Base class for platform-specific CSV ingesters.

    Each ingestor knows how to detect and normalize one ad platform's export format
    into the unified schema. No platform-specific logic exists outside this class.
    """

    @abstractmethod
    def can_handle(self, raw: pd.DataFrame) -> bool:
        """Detect if this ingestor can handle the given DataFrame.

        Inspects column names to determine the source platform.
        """

    @abstractmethod
    def normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Convert raw platform-specific DataFrame into unified schema.

        Returns a DataFrame with columns matching the UnifiedRecord contract.
        """
