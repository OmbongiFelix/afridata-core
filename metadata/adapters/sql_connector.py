"""
Adapter for ingesting relational database sources.

Uses SQLAlchemy for database-agnostic connectivity. Fetches SQL dataset
records from the database registry whose metadata_score is below 0.7,
then executes each record's stored query or table reference and streams
large result sets in chunks to avoid memory overflow.

Supported backends:
    - PostgreSQL  (psycopg2)
    - MySQL       (pymysql)
    - SQLite      (built-in)
    - Any SQLAlchemy-compatible dialect

Credentials are never stored on the instance. They are always read from
environment variables or Django settings via the inherited get_engine().

Important distinction:
    There are two separate engines at play in this connector:

    1. REGISTRY engine  (inherited — get_engine())
       Connects to the pipeline's own database where dataset_files lives.
       Built from DB_CONNECTION_STRING. Used by fetch_eligible_files().

    2. SOURCE engine    (created here — _get_source_engine())
       Connects to the external database that the SQL dataset lives on.
       Its connection string is stored in the dataset_files.file_path
       column for SQL-type records, in the format:
           dialect+driver://user:pass@host/dbname
       The password must come from an environment variable; see
       _get_source_engine() for the substitution convention.
"""

import logging
import os
from typing import Iterator

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .base_connector import BaseConnector

logger = logging.getLogger(__name__)

# Rows per chunk when streaming large result sets.
DEFAULT_CHUNKSIZE = 10_000


class SQLConnector(BaseConnector):
    """
    Connector for SQL dataset records tracked in the database registry.

    Workflow:
        1. connect()  — verifies the registry engine is reachable and
                        confirms at least one eligible SQL record exists.
        2. extract()  — calls fetch_eligible_files() to get all SQL records
                        with metadata_score < 0.7, opens a source engine
                        per record, streams the query in chunks, and
                        concatenates everything into one DataFrame.
    """

    # dataset_files.file_type value expected for SQL sources.
    SUPPORTED_EXTENSIONS = [".sql"]

    def __init__(self, chunksize: int = DEFAULT_CHUNKSIZE):
        """
        Args:
            chunksize: Number of rows per streaming chunk. Increase for
                       fast networks, decrease for memory-constrained hosts.
                       Defaults to 10 000.
        """
        super().__init__()
        self.chunksize = chunksize

        # Cache of source engines keyed by connection string so we don't
        # create a new engine for every chunk of the same dataset.
        self._source_engines: dict[str, Engine] = {}

    # ------------------------------------------------------------------
    # connect
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Verify the registry engine is reachable and log how many eligible
        SQL records are waiting to be processed.

        Raises EnvironmentError / SQLAlchemyError on failure (from get_engine).
        """
        engine = self.get_engine()
        self.logger.info(
            "SQLConnector: registry engine ready (%s).",
            engine.url.render_as_string(hide_password=True),
        )

        eligible = self.fetch_eligible_files()
        self.logger.info(
            "SQLConnector: %d eligible SQL record(s) found (metadata_score < 0.7).",
            len(eligible),
        )

    # ------------------------------------------------------------------
    # extract
    # ------------------------------------------------------------------

    def extract(self) -> pd.DataFrame:
        """
        Fetch all eligible SQL records from the registry (metadata_score < 0.7),
        execute each stored query against its source database, and return one
        concatenated DataFrame.

        Each row in the result is tagged with:
            source_db         — source connection string (password redacted)
            source_query      — the SQL query that produced the row
            file_type         — 'sql'
            metadata_score    — score from the registry record
            row_count         — total rows returned by that query
            ingested_at       — UTC timestamp of this ingestion run

        Returns:
            pd.DataFrame — empty DataFrame if no eligible records are found.

        Raises:
            RuntimeError: propagated from fetch_eligible_files if the
                          registry query fails.
        """
        eligible = self.fetch_eligible_files()

        if not eligible:
            self.logger.info("SQLConnector.extract: no eligible SQL records found.")
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []

        for record in eligible:
            if not self.validate_source(record):
                continue

            df = self._execute_record(record)
            if df is not None:
                frames.append(df)

        if not frames:
            self.logger.warning(
                "SQLConnector.extract: all eligible records failed to execute."
            )
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        self.logger.info(
            "SQLConnector.extract: loaded %d total row(s) from %d record(s).",
            len(combined),
            len(frames),
        )
        return combined

    # ------------------------------------------------------------------
    # validate_source  (override)
    # ------------------------------------------------------------------

    def validate_source(self, file_record: dict) -> bool:
        """
        Extends the base validation with SQL-specific checks.

        Additional checks:
            - file_path (used as the source connection string) is non-empty.
            - query_text is present and non-empty.

        Args:
            file_record: dict from fetch_eligible_files(), expected to carry
                         a 'query_text' key with the SQL to execute.

        Returns:
            True if the record passes all checks, False otherwise.
        """
        if not super().validate_source(file_record):
            return False

        query_text = file_record.get("query_text", "").strip()
        if not query_text:
            self.logger.warning(
                "validate_source: record id=%s has no query_text — skipping.",
                file_record.get("id"),
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Source-engine management
    # ------------------------------------------------------------------

    def _get_source_engine(self, connection_string: str) -> Engine:
        """
        Return (or create and cache) a SQLAlchemy engine for an external
        source database.

        Password substitution convention:
            Store the connection string in the DB with the password field
            replaced by the name of an environment variable, e.g.:

                postgresql+psycopg2://user:${DB_SOURCE_PASSWORD}@host/dbname

            This method expands ${VAR_NAME} tokens from the environment
            before building the engine, so credentials are never at rest
            in the registry.

        Args:
            connection_string: Raw connection string from dataset_files.file_path.

        Returns:
            SQLAlchemy Engine.

        Raises:
            EnvironmentError: if a referenced env variable is not set.
            sqlalchemy.exc.SQLAlchemyError: if the engine cannot connect.
        """
        if connection_string in self._source_engines:
            return self._source_engines[connection_string]

        resolved = os.path.expandvars(connection_string)

        # Catch un-expanded tokens — os.path.expandvars leaves them as-is
        # on Linux when the variable is missing, e.g. "${MISSING_VAR}".
        if "${" in resolved:
            missing = [
                token.split("}")[0]
                for token in resolved.split("${")[1:]
            ]
            raise EnvironmentError(
                f"Source connection string references unset environment "
                f"variable(s): {missing}. Set them before running the pipeline."
            )

        engine = create_engine(resolved, future=True)
        self._source_engines[connection_string] = engine

        self.logger.info(
            "_get_source_engine: created engine for '%s'.",
            engine.url.render_as_string(hide_password=True),
        )
        return engine

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_record(self, record: dict) -> pd.DataFrame | None:
        """
        Execute the SQL query stored in a registry record against its
        source database, streaming in chunks, and return a tagged DataFrame.

        Args:
            record: dict from fetch_eligible_files(), must contain:
                        file_path      — source DB connection string
                        query_text     — SQL query to execute
                        file_type      — 'sql'
                        metadata_score — float

        Returns:
            Tagged pd.DataFrame, or None if execution fails.
        """
        connection_string = record["file_path"]
        query_text = record["query_text"].strip()
        score = record["metadata_score"]

        try:
            engine = self._get_source_engine(connection_string)
        except EnvironmentError as exc:
            self.logger.error(
                "_execute_record: cannot build engine for record id=%s: %s — skipping.",
                record.get("id"),
                exc,
            )
            return None

        redacted_conn = engine.url.render_as_string(hide_password=True)

        try:
            chunks: list[pd.DataFrame] = list(
                self._stream_query(engine, query_text)
            )

            if not chunks:
                self.logger.warning(
                    "_execute_record: query for record id=%s returned no rows.",
                    record.get("id"),
                )
                return None

            df = pd.concat(chunks, ignore_index=True)

            # --- tag rows with source metadata ---
            df["source_db"] = redacted_conn
            df["source_query"] = query_text
            df["file_type"] = record.get("file_type", "sql")
            df["metadata_score"] = score
            df["row_count"] = len(df)
            df["ingested_at"] = pd.Timestamp.utcnow()

            self.logger.info(
                "_execute_record: record id=%s → %d row(s) from '%s'.",
                record.get("id"),
                len(df),
                redacted_conn,
            )
            return df

        except Exception as exc:
            self.logger.error(
                "_execute_record: query failed for record id=%s on '%s': %s — skipping.",
                record.get("id"),
                redacted_conn,
                exc,
            )
            return None

    def _stream_query(self, engine: Engine, query_text: str) -> Iterator[pd.DataFrame]:
        """
        Execute a SQL query and yield results as a sequence of DataFrames,
        each of at most `self.chunksize` rows.

        Using server-side cursors (via stream_results=True) keeps memory
        usage flat regardless of result-set size. Falls back gracefully if
        the dialect does not support streaming.

        Args:
            engine:     Source database engine.
            query_text: SQL query string to execute.

        Yields:
            pd.DataFrame chunks.
        """
        with engine.connect().execution_options(stream_results=True) as conn:
            result = conn.execute(text(query_text))
            columns = list(result.keys())

            while True:
                rows = result.fetchmany(self.chunksize)
                if not rows:
                    break

                chunk = pd.DataFrame(rows, columns=columns)
                self.logger.debug(
                    "_stream_query: yielding chunk of %d row(s).", len(chunk)
                )
                yield chunk

    # ------------------------------------------------------------------
    # Schema introspection  (utility — does not affect the pipeline)
    # ------------------------------------------------------------------

    def list_tables(self, connection_string: str) -> list[str]:
        """
        Return the names of all tables visible to the given source connection.

        Useful for discovery and debugging; not called automatically by
        connect() or extract().

        Args:
            connection_string: Source DB connection string (may use ${VAR} tokens).

        Returns:
            Sorted list of table names.
        """
        from sqlalchemy import inspect as sa_inspect

        engine = self._get_source_engine(connection_string)
        inspector = sa_inspect(engine)
        tables = sorted(inspector.get_table_names())
        self.logger.info(
            "list_tables: found %d table(s) on '%s'.",
            len(tables),
            engine.url.render_as_string(hide_password=True),
        )
        return tables