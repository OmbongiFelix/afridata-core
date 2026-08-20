"""
Main pipeline orchestrator for the Metadata Extraction system.

This module is intentionally thin. It imports one class or function
from each stage module and chains them in order:

    1. Adapter       → ingest raw data into a DataFrame
    2. Profiler      → compute column-level statistics
    3. Extractor     → extract text / SQL schema metadata
    4. Classifier    → assign semantic types via ML model
    5. LLM Generator → enrich metadata using an LLM prompt
    6. SchemaBuilder → serialise final output to JSON Schema



Usage:
    # CSV source
    result = MetadataPipeline(source="csv", path="data.csv").run()

    # Excel source
    result = MetadataPipeline(source="excel", path="report.xlsx").run()

    # SQL source
    from sqlalchemy import create_engine
    engine = create_engine("postgresql+psycopg2://user:pass@host/db")
    result = MetadataPipeline(
        source="sql",
        engine=engine,
        table_name="orders",
        schema="public",
    ).run()

    # Access outputs
    json_schema = result.json_schema      # JSON Schema draft-07 string
    profiles    = result.profiles         # enriched column profiles dict
    report      = result.schema_report    # summary diagnostics dict
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .profiler import DataFrameProfiler
from metadata.core.extractors.csv_excel_extractor import CsvExcelExtractor
from metadata.core.extractors.sql_extractor import SqlExtractor
from metadata.core.enhancement.semantic_classifier import SemanticClassifier
from  metadata.core.enhancement.llm_generator import LLMGenerator
from metadata.core.schema_builder import SchemaBuilder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported source types
# ---------------------------------------------------------------------------
_VALID_SOURCES = frozenset({"csv", "excel", "sql"})


# ---------------------------------------------------------------------------
# Pipeline result container
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """
    Immutable result object returned by ``MetadataPipeline.run()``.

    Attributes:
        profiles      Fully enriched column profile dict produced by all
                      pipeline stages.  Key = column name, value = profile
                      dict.  Useful for inspection or passing to custom
                      downstream consumers.

        schema        Raw JSON Schema dict (draft-07) produced by
                      ``SchemaBuilder.build()``.

        json_schema   Pretty-printed JSON string of ``schema``.  Ready for
                      storage or API serialisation.

        schema_report Diagnostic summary produced by
                      ``SchemaBuilder.schema_report()``.  Contains type
                      distributions, nullable counts, tagged columns, etc.

        elapsed_s     Wall-clock time in seconds for the complete
                      ``MetadataPipeline.run()`` call.

        stage_times   Per-stage timing dict keyed by stage name.
                      Example: {"profiler": 0.12, "extractor": 0.03, ...}
    """
    profiles:      dict[str, dict]
    schema:        dict[str, Any]
    json_schema:   str
    schema_report: dict[str, Any]
    elapsed_s:     float
    stage_times:   dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class MetadataPipeline:
    """
    Orchestrates the end-to-end Metadata Extraction Pipeline.

    The pipeline is source-agnostic: pass ``source="csv"``, ``"excel"``,
    or ``"sql"`` and supply the corresponding keyword arguments.

    Each stage is called exactly once in sequence; results are passed
    directly into the next stage without intermediate persistence.

    Args (common):
        source (str):
            One of ``"csv"``, ``"excel"``, or ``"sql"`` (case-insensitive).

        dataset_title (str):
            Human-readable title embedded in the output JSON Schema.
            Defaults to the file basename or table name when omitted.

        dataset_description (str):
            Optional description for the top-level JSON Schema document.

        additional_properties (bool):
            Passed to ``SchemaBuilder``; controls whether the output schema
            permits undeclared columns.  Default ``True``.

    Args (CSV / Excel sources):
        path (str | Path):
            File system path to the CSV or Excel file.

        read_kwargs (dict):
            Extra keyword arguments forwarded verbatim to
            ``pd.read_csv()`` or ``pd.read_excel()``.  Use this for
            encoding, sheet names, header rows, etc.

        raw_header_rows (list[list]):
            Optional pre-header rows from an Excel file (e.g. title
            banners).  Forwarded to ``CsvExcelExtractor`` for
            multi-header detection.

    Args (SQL source):
        engine (sqlalchemy.engine.Engine):
            Connected SQLAlchemy engine.  The pipeline does not manage
            its lifecycle — callers are responsible for disposal.

        table_name (str):
            Table to profile (unquoted, as it appears in the catalogue).

        schema (str | None):
            Optional database schema/namespace (e.g. ``"public"``).

        sql_query (str | None):
            Optional raw SELECT statement.  When supplied, the DataFrame
            is loaded from this query instead of a full table scan.
            ``table_name`` is still required for SQL metadata reflection.

        read_kwargs (dict):
            Extra keyword arguments forwarded to ``pd.read_sql()``.
    """

    def __init__(
        self,
        source: str,
        *,
        # CSV / Excel
        path: str | None = None,
        read_kwargs: dict[str, Any] | None = None,
        raw_header_rows: list[list[Any]] | None = None,
        # SQL
        engine: Any | None = None,
        table_name: str | None = None,
        schema: str | None = None,
        sql_query: str | None = None,
        # SchemaBuilder
        dataset_title: str = "",
        dataset_description: str = "",
        additional_properties: bool = True,
    ):
        self.source = source.lower()
        if self.source not in _VALID_SOURCES:
            raise ValueError(
                f"Invalid source '{source}'. Must be one of: {sorted(_VALID_SOURCES)}."
            )

        # CSV / Excel params
        self.path             = path
        self.read_kwargs      = read_kwargs or {}
        self.raw_header_rows  = raw_header_rows or []

        # SQL params
        self.engine           = engine
        self.table_name       = table_name
        self.schema           = schema
        self.sql_query        = sql_query

        # SchemaBuilder params
        self.dataset_title    = dataset_title or table_name or (
            str(path).split("/")[-1].rsplit(".", 1)[0] if path else "Dataset"
        )
        self.dataset_description = dataset_description
        self.additional_properties = additional_properties

        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> PipelineResult:
        """
        Execute all pipeline stages in sequence and return a PipelineResult.

        Stages:
            1. **Adapter**       — load raw source into a ``pd.DataFrame``.
            2. **Profiler**      — compute per-column statistics.
            3. **Extractor**     — augment profiles with source-specific
                                   structural metadata (CSV/Excel or SQL).
            4. **Classifier**    — assign semantic types via rules + ML.
            5. **LLM Generator** — enrich profiles with business descriptions,
                                   tags, and notes.
            6. **SchemaBuilder** — serialise enriched profiles to JSON Schema.

        Returns:
            ``PipelineResult`` containing the final JSON Schema, raw profiles
            dict, diagnostic report, and per-stage timing information.

        Raises:
            ValueError:  Invalid ``source``, missing required arguments, or
                         an empty DataFrame returned by the adapter.
            FileNotFoundError:
                         CSV/Excel file not found at ``path``.
            sqlalchemy.exc.NoSuchTableError:
                         SQL table not found.
            sqlalchemy.exc.OperationalError:
                         Database connection failure.
        """
        pipeline_start = time.perf_counter()
        stage_times: dict[str, float] = {}

        self.logger.info(
            "MetadataPipeline.run: starting [source=%s title='%s'].",
            self.source,
            self.dataset_title,
        )

        # ----------------------------------------------------------
        # Stage 1 — Adapter: ingest raw data into a DataFrame
        # ----------------------------------------------------------
        t0 = time.perf_counter()
        df = self._run_adapter()
        stage_times["adapter"] = round(time.perf_counter() - t0, 4)
        self.logger.info(
            "Stage 1 adapter: loaded DataFrame [%d rows × %d cols] in %.3fs.",
            len(df), len(df.columns), stage_times["adapter"],
        )

        # ----------------------------------------------------------
        # Stage 2 — Profiler: compute column-level statistics
        # ----------------------------------------------------------
        t0 = time.perf_counter()
        profiles = self._run_profiler(df)
        stage_times["profiler"] = round(time.perf_counter() - t0, 4)
        self.logger.info(
            "Stage 2 profiler: profiled %d column(s) in %.3fs.",
            len(profiles), stage_times["profiler"],
        )

        # ----------------------------------------------------------
        # Stage 3 — Extractor: augment with source-specific metadata
        # ----------------------------------------------------------
        t0 = time.perf_counter()
        profiles = self._run_extractor(df, profiles)
        stage_times["extractor"] = round(time.perf_counter() - t0, 4)
        self.logger.info(
            "Stage 3 extractor: augmented %d column(s) in %.3fs.",
            len(profiles), stage_times["extractor"],
        )

        # ----------------------------------------------------------
        # Stage 4 — Classifier: assign semantic types
        # ----------------------------------------------------------
        t0 = time.perf_counter()
        profiles = self._run_classifier(profiles)
        stage_times["classifier"] = round(time.perf_counter() - t0, 4)
        self.logger.info(
            "Stage 4 classifier: classified %d column(s) in %.3fs.",
            len(profiles), stage_times["classifier"],
        )

        # ----------------------------------------------------------
        # Stage 5 — LLM Generator: enrich with business metadata
        # ----------------------------------------------------------
        t0 = time.perf_counter()
        profiles = self._run_llm_generator(profiles)
        stage_times["llm_generator"] = round(time.perf_counter() - t0, 4)
        self.logger.info(
            "Stage 5 llm_generator: enriched %d column(s) in %.3fs.",
            len(profiles), stage_times["llm_generator"],
        )

        # ----------------------------------------------------------
        # Stage 6 — SchemaBuilder: serialise to JSON Schema
        # ----------------------------------------------------------
        t0 = time.perf_counter()
        schema, json_schema, report = self._run_schema_builder(profiles)
        stage_times["schema_builder"] = round(time.perf_counter() - t0, 4)
        self.logger.info(
            "Stage 6 schema_builder: built schema with %d property/properties in %.3fs.",
            len(schema.get("properties", {})), stage_times["schema_builder"],
        )

        elapsed = round(time.perf_counter() - pipeline_start, 4)
        self.logger.info(
            "MetadataPipeline.run: completed in %.3fs. Stage breakdown: %s",
            elapsed,
            {k: f"{v:.3f}s" for k, v in stage_times.items()},
        )

        return PipelineResult(
            profiles=profiles,
            schema=schema,
            json_schema=json_schema,
            schema_report=report,
            elapsed_s=elapsed,
            stage_times=stage_times,
        )

    # ------------------------------------------------------------------
    # Stage implementations — one private method per stage.
    # Keep each method focused: validate inputs, call the stage, return.
    # ------------------------------------------------------------------

    def _run_adapter(self) -> pd.DataFrame:
        """
        Stage 1 — Load source data into a ``pd.DataFrame``.

        Dispatches to the appropriate pandas reader based on ``self.source``.
        For SQL sources the engine is used directly; for file sources
        ``self.path`` is required.

        Returns:
            Non-empty ``pd.DataFrame``.

        Raises:
            ValueError:        Missing required arguments for the chosen source.
            FileNotFoundError: ``path`` does not exist (CSV/Excel).
        """
        if self.source in ("csv", "excel"):
            if not self.path:
                raise ValueError(
                    f"MetadataPipeline: 'path' is required for source='{self.source}'."
                )
            if self.source == "csv":
                df = pd.read_csv(self.path, **self.read_kwargs)
                self.logger.debug("Adapter: read CSV from '%s'.", self.path)
            else:
                df = pd.read_excel(self.path, **self.read_kwargs)
                self.logger.debug("Adapter: read Excel from '%s'.", self.path)

        elif self.source == "sql":
            if self.engine is None:
                raise ValueError(
                    "MetadataPipeline: 'engine' is required for source='sql'."
                )
            if not self.table_name:
                raise ValueError(
                    "MetadataPipeline: 'table_name' is required for source='sql'."
                )
            query = self.sql_query or self.table_name
            df = pd.read_sql(query, self.engine, **self.read_kwargs)
            self.logger.debug(
                "Adapter: read SQL [table=%s query=%s].",
                self.table_name,
                self.sql_query or "<full table scan>",
            )

        else:
            # Guard — _VALID_SOURCES checked in __init__ so this is unreachable
            # in normal operation, but kept for defensive completeness.
            raise ValueError(f"Unknown source: '{self.source}'.")

        if df.empty:
            raise ValueError(
                f"MetadataPipeline adapter returned an empty DataFrame "
                f"[source={self.source}]. Verify the source contains data."
            )

        return df

    def _run_profiler(self, df: pd.DataFrame) -> dict[str, dict]:
        """
        Stage 2 — Compute column-level statistics.

        Args:
            df: Non-empty DataFrame from the adapter stage.

        Returns:
            Profile dict keyed by column name.
        """
        return DataFrameProfiler(df).run()

    def _run_extractor(
        self,
        df: pd.DataFrame,
        profiles: dict[str, dict],
    ) -> dict[str, dict]:
        """
        Stage 3 — Augment profiles with source-specific structural metadata.

        For CSV/Excel sources delegates to ``CsvExcelExtractor``.
        For SQL sources delegates to ``SqlExtractor``.

        Both extractors operate in-place on ``profiles`` and return the
        same dict, so the return value is used directly.

        Args:
            df:       DataFrame from the adapter stage.
            profiles: Profile dict from the profiler stage.

        Returns:
            Augmented profile dict.
        """
        if self.source in ("csv", "excel"):
            return CsvExcelExtractor(
                df,
                source_format=self.source,
                raw_header_rows=self.raw_header_rows,
            ).augment(profiles)

        # SQL source
        assert self.engine is not None, "engine must be set for source='sql'"
        assert self.table_name is not None, "table_name must be set for source='sql'"
        return SqlExtractor(
            engine=self.engine,
            table_name=self.table_name,
            schema=self.schema,
        ).augment(profiles)

    def _run_classifier(self, profiles: dict[str, dict]) -> dict[str, dict]:
        """
        Stage 4 — Assign a semantic type to every column.

        Args:
            profiles: Augmented profile dict from the extractor stage.

        Returns:
            Profile dict with ``semantic_type`` and ``semantic_confidence``
            injected per column.
        """
        return SemanticClassifier().classify(profiles)

    def _run_llm_generator(self, profiles: dict[str, dict]) -> dict[str, dict]:
        """
        Stage 5 — Enrich column profiles with LLM-generated business metadata.

        Reads LLM configuration from Django settings (backend, model, API
        key, batch size, etc.).  See ``llm_generator.py`` for the full
        settings reference.

        On partial failure (e.g. one batch times out) the generator applies
        safe defaults and continues — no exception propagates here.

        Args:
            profiles: Classified profile dict from the classifier stage.

        Returns:
            Profile dict with ``description``, ``tags``, ``business_name``,
            and ``notes`` injected per column.
        """
        return LLMGenerator().enrich(profiles)

    def _run_schema_builder(
        self,
        profiles: dict[str, dict],
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        """
        Stage 6 — Serialise enriched profiles to a JSON Schema document.

        Args:
            profiles: Fully enriched profile dict from the LLM generator.

        Returns:
            3-tuple of:
                schema       (dict)  — raw JSON Schema draft-07 document.
                json_schema  (str)   — pretty-printed JSON string.
                report       (dict)  — diagnostic summary from
                                       ``SchemaBuilder.schema_report()``.
        """
        builder = SchemaBuilder(
            dataset_title=self.dataset_title,
            dataset_description=self.dataset_description,
            additional_properties=self.additional_properties,
        )
        schema = builder.build(profiles)
        json_schema = builder.to_json(schema)
        report = builder.schema_report(schema)
        return schema, json_schema, report