"""
--> For External db Data Sources
Schema metadata extractor for SQL/database sources.

Uses SQLAlchemy's Inspector to query the database information schema
and attach relational metadata to each ColumnProfile:

    - Primary key flags
    - Foreign key references (target table + column)
    - NOT NULL constraints
    - Index membership
    - Native SQL type (VARCHAR(255), DECIMAL(10,2), etc.)

This metadata is unavailable from Pandas alone and significantly
improves the quality of semantic type classification.

Usage:
    from profiler import DataFrameProfiler
    from sql_extractor import SqlExtractor

    engine = create_engine("postgresql+psycopg2://user:pass@host/db")
    profiles = DataFrameProfiler(df).run()
    extractor = SqlExtractor(engine, table_name="orders", schema="public")
    augmented = extractor.augment(profiles)
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import inspect as sa_inspect
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Augmented field keys injected into each profile dict.
# Documented here so downstream consumers have a single reference.
# ---------------------------------------------------------------------------
#
#   is_primary_key      bool        True if the column is part of the PK.
#   is_foreign_key      bool        True if the column has a FK constraint.
#   fk_references       list[dict]  Each dict has keys:
#                                       "referred_table"  – str
#                                       "referred_column" – str
#                                       "referred_schema" – str | None
#   is_nullable         bool        False when a NOT NULL constraint exists.
#   index_names         list[str]   Names of indexes that include this column.
#   is_indexed          bool        True when index_names is non-empty.
#   is_unique_indexed   bool        True when any covering index is unique.
#   native_sql_type     str | None  Rendered SQL type, e.g. "VARCHAR(255)".
#   sql_default         str | None  Column default expression as a string.


@dataclass
class _TableSchema:
    """
    Holds raw Inspector output for a single table.

    Built once per ``SqlExtractor.augment()`` call and used as a
    look-up cache so every column does not re-query the database.
    """

    primary_key_cols: set[str] = field(default_factory=set)
    # Maps column name -> list of FK reference dicts
    foreign_keys: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    # Maps column name -> (nullable: bool, server_default: str | None, type_str: str | None)
    column_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Maps column name -> list of (index_name, is_unique)
    index_membership: dict[str, list[tuple[str, bool]]] = field(default_factory=dict)


class SqlExtractor:
    """
    Augments profiler output with relational schema metadata from SQLAlchemy.

    The extractor queries the database once (via the Inspector reflection
    API) and annotates each column profile dict with keys that Pandas-based
    profiling cannot provide.

    Augmented keys (always present after augmentation):

        is_primary_key      (bool)
        is_foreign_key      (bool)
        fk_references       (list[dict])  — empty list if no FK
        is_nullable         (bool)
        index_names         (list[str])   — empty list if not indexed
        is_indexed          (bool)
        is_unique_indexed   (bool)
        native_sql_type     (str | None)
        sql_default         (str | None)

    Usage:
        engine = create_engine("postgresql+psycopg2://user:pass@host/db")
        profiles = DataFrameProfiler(df).run()
        extractor = SqlExtractor(engine, table_name="orders", schema="public")
        augmented = extractor.augment(profiles)
    """

    def __init__(
        self,
        engine: Engine,
        table_name: str,
        schema: str | None = None,
    ):
        """
        Args:
            engine:
                A connected SQLAlchemy ``Engine`` instance.  The extractor
                does not own the engine's lifecycle — callers are responsible
                for disposal.
            table_name:
                Unquoted table name as it appears in the database catalogue
                (case-sensitive on case-sensitive engines).
            schema:
                Optional schema/namespace (e.g. ``"public"``, ``"dbo"``).
                When None the engine's default schema is used.
        """
        if not table_name:
            raise ValueError("SqlExtractor requires a non-empty table_name.")
        self.engine = engine
        self.table_name = table_name
        self.schema = schema
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def augment(self, profiles: dict[str, dict]) -> dict[str, dict]:
        """
        Augment an existing profile dict produced by ``DataFrameProfiler``.

        Mutates each inner dict in-place and returns the same mapping for
        chaining.  All injected keys are always present (defaulting to
        ``False``, ``None``, or ``[]``) so downstream consumers need no
        defensive ``.get()`` calls.

        Args:
            profiles:
                Output of ``DataFrameProfiler.run()`` — a dict keyed by
                column name, each value being a ``ColumnProfile.to_dict()``
                result.

        Returns:
            The same ``profiles`` dict with SQL schema keys injected per column.

        Raises:
            sqlalchemy.exc.NoSuchTableError:
                If ``table_name`` does not exist in the target database.
            sqlalchemy.exc.OperationalError:
                If the engine cannot reach the database.
        """
        self.logger.info(
            "SqlExtractor.augment: reflecting schema for table '%s' (schema=%s).",
            self.table_name,
            self.schema,
        )

        table_schema = self._reflect_table()

        self.logger.info(
            "SqlExtractor.augment: annotating %d column profile(s).",
            len(profiles),
        )

        for col, profile in profiles.items():
            self._annotate_column(col, profile, table_schema)

        self.logger.info(
            "SqlExtractor.augment: completed annotation for %d column(s).",
            len(profiles),
        )
        return profiles

    # ------------------------------------------------------------------
    # Schema reflection
    # ------------------------------------------------------------------

    def _reflect_table(self) -> _TableSchema:
        """
        Query the database information schema via SQLAlchemy Inspector and
        return a ``_TableSchema`` cache object.

        All Inspector calls are batched here so the database is hit once
        regardless of how many columns are being profiled.

        Returns:
            Populated ``_TableSchema`` instance.

        Raises:
            sqlalchemy.exc.NoSuchTableError: Table not found.
            sqlalchemy.exc.OperationalError: Connection failure.
        """
        inspector = sa_inspect(self.engine)
        ts = _TableSchema()

        # --- Primary keys ---
        try:
            pk_info = inspector.get_pk_constraint(self.table_name, schema=self.schema)
            ts.primary_key_cols = set(pk_info.get("constrained_columns", []))
            self.logger.debug(
                "Primary key columns for '%s': %s", self.table_name, ts.primary_key_cols
            )
        except Exception as exc:
            self.logger.warning(
                "SqlExtractor: could not retrieve PK constraint for '%s': %s",
                self.table_name, exc,
            )

        # --- Foreign keys ---
        try:
            for fk in inspector.get_foreign_keys(self.table_name, schema=self.schema):
                # A single FK constraint may span multiple columns; zip them.
                local_cols: list[str] = fk.get("constrained_columns", [])
                ref_cols: list[str] = fk.get("referred_columns", [])
                ref_table: str = fk.get("referred_table", "")
                ref_schema: str | None = fk.get("referred_schema")

                for local_col, ref_col in zip(local_cols, ref_cols):
                    ref_entry = {
                        "referred_table": ref_table,
                        "referred_column": ref_col,
                        "referred_schema": ref_schema,
                    }
                    ts.foreign_keys.setdefault(local_col, []).append(ref_entry)
            self.logger.debug(
                "Foreign key columns for '%s': %s",
                self.table_name,
                list(ts.foreign_keys.keys()),
            )
        except Exception as exc:
            self.logger.warning(
                "SqlExtractor: could not retrieve FK constraints for '%s': %s",
                self.table_name, exc,
            )

        # --- Column-level metadata (nullable, default, type) ---
        try:
            for col_info in inspector.get_columns(self.table_name, schema=self.schema):
                col_name: str = col_info["name"]
                raw_type = col_info.get("type")
                type_str: str | None = None
                try:
                    type_str = str(raw_type.compile(dialect=self.engine.dialect))
                except Exception:
                    # Fallback: use the class name if compile() is unavailable.
                    type_str = type(raw_type).__name__ if raw_type is not None else None

                default = col_info.get("default")

                ts.column_meta[col_name] = {
                    "nullable": bool(col_info.get("nullable", True)),
                    "sql_default": str(default) if default is not None else None,
                    "native_sql_type": type_str,
                }
        except Exception as exc:
            self.logger.warning(
                "SqlExtractor: could not retrieve column metadata for '%s': %s",
                self.table_name, exc,
            )

        # --- Index membership ---
        try:
            for idx in inspector.get_indexes(self.table_name, schema=self.schema):
                idx_name: str = idx.get("name") or ""
                is_unique: bool = bool(idx.get("unique", False))
                for col_name in (idx.get("column_names") or []):
                    if not isinstance(col_name, str):
                        continue
                    ts.index_membership.setdefault(col_name, []).append(
                        (idx_name, is_unique)
                    )
            self.logger.debug(
                "Indexed columns for '%s': %s",
                self.table_name,
                list(ts.index_membership.keys()),
            )
        except Exception as exc:
            self.logger.warning(
                "SqlExtractor: could not retrieve indexes for '%s': %s",
                self.table_name, exc,
            )

        return ts

    # ------------------------------------------------------------------
    # Per-column annotation
    # ------------------------------------------------------------------

    def _annotate_column(
        self,
        col: str,
        profile: dict[str, Any],
        ts: _TableSchema,
    ) -> None:
        """
        Inject SQL schema keys into a single column profile dict.

        All keys are always set so downstream consumers never encounter a
        missing key.  When the column is absent from the reflected schema
        (e.g. a computed/virtual column not in the catalogue) safe defaults
        are used and a warning is emitted.

        Args:
            col:     Column name.
            profile: Mutable profile dict to annotate.
            ts:      Reflected ``_TableSchema`` for this table.
        """
        if col not in ts.column_meta:
            self.logger.warning(
                "SqlExtractor: column '%s' not found in reflected schema — "
                "SQL annotation will use safe defaults.",
                col,
            )

        col_meta = ts.column_meta.get(col, {})
        fk_refs = ts.foreign_keys.get(col, [])
        index_entries = ts.index_membership.get(col, [])

        # Primary / foreign key flags
        profile["is_primary_key"] = col in ts.primary_key_cols
        profile["is_foreign_key"] = bool(fk_refs)
        profile["fk_references"] = fk_refs  # list[dict], may be empty

        # Nullability — Inspector returns True when the column IS nullable;
        # we expose is_nullable to mirror standard SQL NOT NULL semantics.
        profile["is_nullable"] = col_meta.get("nullable", True)

        # Index membership
        index_names = [name for name, _ in index_entries]
        profile["index_names"] = index_names
        profile["is_indexed"] = bool(index_names)
        profile["is_unique_indexed"] = any(unique for _, unique in index_entries)

        # Native SQL type and default
        profile["native_sql_type"] = col_meta.get("native_sql_type")
        profile["sql_default"] = col_meta.get("sql_default")

        self.logger.debug(
            "Annotated column '%s': pk=%s fk=%s nullable=%s indexed=%s type=%s",
            col,
            profile["is_primary_key"],
            profile["is_foreign_key"],
            profile["is_nullable"],
            profile["is_indexed"],
            profile["native_sql_type"],
        )