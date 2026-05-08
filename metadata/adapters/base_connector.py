"""
Base connector interface for all data source adapters.

Every adapter (CSV, Excel, SQL, S3) must subclass BaseConnector and
implement the `connect()` and `extract()` methods. This enforces a
consistent contract across all ingestion sources and allows the pipeline
orchestrator to swap connectors without changing downstream logic.

Usage:
    class CSVConnector(BaseConnector):
        def connect(self): ...
        def extract(self) -> pd.DataFrame: ...
"""

import pandas as pd
from abc import ABC, abstractmethod
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import logging
import os


# I want to replace os.environ.get() with os.getenv()
from pathlib import Path
from dotenv import load_dotenv
# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

#load .env
load_dotenv(BASE_DIR / ".env")






logger = logging.getLogger(__name__)

# Files with a metadata score at or above this threshold are skipped.
METADATA_SCORE_THRESHOLD = 0.7


class BaseConnector(ABC):
    """
    Abstract base class for all data source connectors.

    Subclasses must implement connect() and extract().
    Shared DB connection logic and metadata-score filtering live here
    so every connector inherits them without duplication.
    """

    # Subclasses declare which file extensions they handle, e.g. ['.csv']
    SUPPORTED_EXTENSIONS: list[str] = []

    def __init__(self):
        self._engine: Engine | None = None
        self._setup_logging()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _setup_logging(self) -> None:
        """Attach a named logger scoped to the concrete subclass."""
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Database connection  (shared by all connectors)
    # ------------------------------------------------------------------

    def get_engine(self) -> Engine:
        """
        Return a live SQLAlchemy engine, creating one on first call.

        Connection string is read from the DB_CONNECTION_STRING environment
        variable. Never hardcode credentials here.

        Raises:
            EnvironmentError: if DB_CONNECTION_STRING is not set.
            sqlalchemy.exc.SQLAlchemyError: if the connection cannot be established.
        """
        if self._engine is not None:
            return self._engine

        #connection_string = os.environ.get("DB_CONNECTION_STRING")
        connection_string = os.getenv("DB_CONNECTION_STRING")
        if not connection_string:
            raise EnvironmentError(
                "DB_CONNECTION_STRING environment variable is not set. "
                "Set it before running the pipeline."
            )

        self.logger.info("Creating database engine.")
        self._engine = create_engine(connection_string, future=True)
        return self._engine

    # ------------------------------------------------------------------
    # Metadata-score filtering  (shared by all connectors)
    # ------------------------------------------------------------------

    def fetch_eligible_files(self) -> list[dict]:
        """
        Query the database registry and return only the file records whose
        metadata_score is below METADATA_SCORE_THRESHOLD (< 0.7) and whose
        file type is handled by the concrete subclass.

        Expected DB table — `dataset_files`:
            id              INTEGER  PRIMARY KEY
            file_path       TEXT     path / identifier stored in the DB
            file_type       TEXT     'csv', 'xls', 'xlsx', 'sql', …
            metadata_score  FLOAT    completeness score  0.0 – 1.0

        Returns:
            List of dicts, one per eligible file:
                [{"id": 1, "file_path": "...", "file_type": "csv",
                  "metadata_score": 0.45}, ...]

        Raises:
            RuntimeError: if the query fails or the table does not exist.
        """
        if not self.SUPPORTED_EXTENSIONS:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define SUPPORTED_EXTENSIONS."
            )

        engine = self.get_engine()

        # Build a safe IN-list from the subclass's declared extensions.
        # Extensions are stored without the leading dot in the DB, e.g. 'csv'.
        types = [ext.lstrip(".") for ext in self.SUPPORTED_EXTENSIONS]
        placeholders = ", ".join(f":type_{i}" for i in range(len(types)))
        params : dict[str, str | float] = {f"type_{i}": t for i, t in enumerate(types)}
        params["threshold"] = METADATA_SCORE_THRESHOLD

        query = text(
            f"""
            SELECT id, file_path, file_type, metadata_score
            FROM   dataset_files
            WHERE  metadata_score < :threshold
              AND  file_type IN ({placeholders})
            ORDER  BY metadata_score ASC
            """
        )

        try:
            with engine.connect() as conn:
                result = conn.execute(query, params)
                rows = [dict(row._mapping) for row in result]

            self.logger.info(
                "fetch_eligible_files: found %d file(s) with metadata_score < %.2f "
                "for types %s.",
                len(rows),
                METADATA_SCORE_THRESHOLD,
                types,
            )
            return rows

        except Exception as exc:
            self.logger.error("fetch_eligible_files failed: %s", exc)
            raise RuntimeError(f"Could not fetch eligible files from DB: {exc}") from exc

    # ------------------------------------------------------------------
    # Source validation  (shared utility, can be overridden)
    # ------------------------------------------------------------------

    def validate_source(self, file_record: dict) -> bool:
        """
        Basic sanity-check on a file record returned by fetch_eligible_files.

        Checks:
          - file_path is present and non-empty.
          - metadata_score is genuinely below the threshold.

        Returns True if the record is valid, False otherwise.
        Subclasses may override to add format-specific checks.
        """
        file_path = file_record.get("file_path", "")
        score = file_record.get("metadata_score")

        if not file_path:
            self.logger.warning("Skipping record with empty file_path: %s", file_record)
            return False

        if score is None or score >= METADATA_SCORE_THRESHOLD:
            self.logger.warning(
                "Skipping '%s': metadata_score %s is not below threshold %.2f.",
                file_path,
                score,
                METADATA_SCORE_THRESHOLD,
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def connect(self) -> None:
        """
        Perform any connector-specific setup (e.g. open a file handle,
        verify the DB table exists). Called once before extract().
        """
        raise NotImplementedError

    @abstractmethod
    def extract(self) -> pd.DataFrame:
        """
        Pull data from all eligible files (those returned by
        fetch_eligible_files) and return a single concatenated DataFrame.

        Every returned DataFrame must include at minimum:
            source_file, file_type, metadata_score, ingested_at
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement extract()."
        )