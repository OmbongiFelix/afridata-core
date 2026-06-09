#metadata/core/schema_builder.py
"""
JSON Schema builder — final stage of the Metadata Extraction Pipeline.

Takes the enriched metadata list produced by LLMMetadataGenerator and
serialises it into a valid JSON Schema (draft-07) document, ready for
storage in the database or return via the REST API.

Output structure:
    {
      '$schema': 'http://json-schema.org/draft-07/schema#',
      'title':   '<dataset name>',
      'type':    'object',
      'properties': { <column_name>: { type, description, ... } },
      'required': [<non-nullable column names>]
    }

Usage:
    from profiler import DataFrameProfiler
    from csv_excel_extractor import CsvExcelExtractor
    from semantic_classifier import SemanticClassifier
    from llm_generator import LLMGenerator
    from schema_builder import SchemaBuilder

    profiles  = DataFrameProfiler(df).run()
    augmented = CsvExcelExtractor(df, source_format="csv").augment(profiles)
    classified = SemanticClassifier().classify(augmented)
    enriched   = LLMGenerator().enrich(classified)

    builder = SchemaBuilder(dataset_title="orders")
    schema  = builder.build(enriched)
    json_str = builder.to_json(schema)
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON Schema draft identifier emitted in every output document.
# ---------------------------------------------------------------------------
JSON_SCHEMA_DRAFT = "http://json-schema.org/draft-07/schema#"

# ---------------------------------------------------------------------------
# Mapping from semantic_type → JSON Schema "type" string.
# Types not listed here fall back to _dtype_to_json_type().
# ---------------------------------------------------------------------------
_SEMANTIC_TYPE_MAP: dict[str, str] = {
    #"id":          "string",
    "name":        "string",
    "email":       "string",
    "phone":       "string",
    "url":         "string",
    "address":     "string",
    "description": "string",
    "category":    "string",
    "age":         "integer",
    "year":        "integer",
    "count":       "integer",
    "date":        "string",
    "datetime":    "string",
    "boolean":     "boolean",
    "currency":    "number",
    "percentage":  "number",
    "score":       "number",
    "latitude":    "number",
    "longitude":   "number",
    "unknown":     "string",   # safe fallback
}

# ---------------------------------------------------------------------------
# Mapping from semantic_type → JSON Schema "format" string (where applicable).
# Columns whose semantic_type has no entry here will not include a "format".
# ---------------------------------------------------------------------------
_SEMANTIC_FORMAT_MAP: dict[str, str] = {
    "email":    "email",
    "url":      "uri",
    "date":     "date",
    "datetime": "date-time",
    "id":       "uuid",       # heuristic — may be overridden by is_id_like
}

# ---------------------------------------------------------------------------
# Mapping from pandas dtype prefix → JSON Schema "type" string.
# Used as a fallback when semantic_type is absent or "unknown".
# ---------------------------------------------------------------------------
_PANDAS_DTYPE_MAP: list[tuple[str, str]] = [
    ("int",      "integer"),
    ("float",    "number"),
    ("bool",     "boolean"),
    ("datetime", "string"),
    ("object",   "string"),
    ("string",   "string"),
    ("category", "string"),
]


def _dtype_to_json_type(dtype: str) -> str:
    """
    Convert a pandas dtype string to a JSON Schema primitive type.

    Args:
        dtype: Pandas dtype string, e.g. ``"int64"``, ``"object"``.

    Returns:
        JSON Schema type string: ``"string"``, ``"number"``,
        ``"integer"``, or ``"boolean"``.  Falls back to ``"string"``
        when no match is found.
    """
    dtype_lower = dtype.lower()
    for prefix, json_type in _PANDAS_DTYPE_MAP:
        if dtype_lower.startswith(prefix):
            return json_type
    return "string"


def _resolve_json_type(profile: dict[str, Any]) -> str:
    """
    Determine the JSON Schema ``type`` for a column.

    Resolution order:
        1. ``semantic_type`` lookup in ``_SEMANTIC_TYPE_MAP``.
        2. ``dtype`` lookup via ``_dtype_to_json_type()``.
        3. Hard fallback to ``"string"``.

    Args:
        profile: Enriched column profile dict.

    Returns:
        JSON Schema type string.
    """
    semantic = profile.get("semantic_type", "unknown")
    if semantic and semantic in _SEMANTIC_TYPE_MAP:
        return _SEMANTIC_TYPE_MAP[semantic]
    dtype = profile.get("dtype") or ""
    return _dtype_to_json_type(dtype) if dtype else "string"


def _resolve_format(profile: dict[str, Any]) -> str | None:
    """
    Return a JSON Schema ``format`` keyword for the column, or None.

    The ``format`` is derived from the semantic_type.  It is suppressed
    when the semantic_type is ``"id"`` but the column does not look like a
    UUID (i.e. ``is_id_like`` is True but values are numeric keys).

    Args:
        profile: Enriched column profile dict.

    Returns:
        Format string (e.g. ``"email"``, ``"date-time"``) or None.
    """
    semantic = profile.get("semantic_type", "")
    fmt = _SEMANTIC_FORMAT_MAP.get(semantic)

    # For ID columns: only apply uuid format when the column is string-typed.
    if fmt == "uuid":
        json_type = _resolve_json_type(profile)
        if json_type != "string":
            return None

    return fmt


def _build_column_schema(profile: dict[str, Any]) -> dict[str, Any]:
    """
    Build a single JSON Schema property object from an enriched column profile.

    The property object always contains at minimum ``type`` and
    ``description``.  Optional keywords (``format``, ``minimum``,
    ``maximum``, ``examples``, ``x-*`` extensions) are added when the
    profile contains relevant data.

    Args:
        profile: Fully enriched column profile dict containing at minimum:
                 ``dtype``, ``semantic_type``, ``description``,
                 ``business_name``, ``tags``, ``notes``.

    Returns:
        A dict representing the JSON Schema property definition for this
        column, e.g.::

            {
                "type": "string",
                "format": "email",
                "description": "Customer's primary contact email address.",
                "title": "Customer Email",
                "examples": ["alice@example.com"],
                "x-semantic-type": "email",
                "x-tags": ["PII"],
                "x-notes": ""
            }
    """
    col_schema: dict[str, Any] = {}

    # --- Core type ---
    json_type = _resolve_json_type(profile)

    # Nullable columns become a union type: [actual_type, "null"]
    # is_nullable from SqlExtractor; null_pct > 0 used as CSV/Excel fallback.
    is_nullable = profile.get("is_nullable", True)
    null_pct = profile.get("null_pct", 0.0) or 0.0
    has_nulls = null_pct > 0.0

    if is_nullable or has_nulls:
        col_schema["type"] = [json_type, "null"]
    else:
        col_schema["type"] = json_type

    # --- Format ---
    fmt = _resolve_format(profile)
    if fmt:
        col_schema["format"] = fmt

    # --- Title (business name from LLM, or column name as fallback) ---
    business_name = profile.get("business_name") or ""
    col_schema["title"] = business_name if business_name else str(
        profile.get("column_name", "")
    ).replace("_", " ").title()

    # --- Description (LLM-generated) ---
    description = profile.get("description") or ""
    col_schema["description"] = description

    # --- Numeric constraints ---
    if json_type in ("number", "integer"):
        min_val = profile.get("min")
        max_val = profile.get("max")
        if min_val is not None:
            col_schema["minimum"] = min_val
        if max_val is not None:
            col_schema["maximum"] = max_val

    # --- String constraints ---
    if json_type == "string":
        min_len = profile.get("min_length")
        max_len = profile.get("max_length")
        if min_len is not None:
            col_schema["minLength"] = min_len
        if max_len is not None:
            col_schema["maxLength"] = max_len

    # --- Enum (low-cardinality categories) ---
    # When unique_count is small and the column is categorical, list all
    # known values so consumers can validate against them.
    semantic = profile.get("semantic_type", "")
    unique_count = profile.get("unique_count") or 0
    sample_values = profile.get("sample_values") or []
    if (
        semantic in ("category", "boolean")
        and 0 < unique_count <= 20
        and sample_values
        and unique_count == len(sample_values)   # sample captured every value
    ):
        col_schema["enum"] = [v for v in sample_values if v is not None]

    # --- Examples (sample_values when not already used for enum) ---
    elif sample_values and "enum" not in col_schema:
        col_schema["examples"] = sample_values[:5]

    # --- X- extension fields (non-standard, prefixed per draft-07 convention) ---
    col_schema["x-semantic-type"]       = semantic
    col_schema["x-semantic-confidence"] = profile.get("semantic_confidence", 0.0)

    tags = profile.get("tags") or []
    col_schema["x-tags"] = tags

    notes = profile.get("notes") or ""
    if notes:
        col_schema["x-notes"] = notes

    # Expose SQL metadata when present
    if profile.get("is_primary_key"):
        col_schema["x-primary-key"] = True
    if profile.get("is_foreign_key"):
        col_schema["x-foreign-key"] = True
        fk_refs = profile.get("fk_references") or []
        if fk_refs:
            col_schema["x-fk-references"] = fk_refs

    if profile.get("native_sql_type"):
        col_schema["x-native-sql-type"] = profile["native_sql_type"]

    # Expose CSV/Excel extractor flags when they influenced typing
    if profile.get("detected_date_format"):
        col_schema["x-detected-date-format"] = profile["detected_date_format"]
    if profile.get("is_currency"):
        col_schema["x-is-currency"] = True
    if profile.get("is_percentage"):
        col_schema["x-is-percentage"] = True
    if profile.get("excel_col_letter"):
        col_schema["x-excel-col-letter"] = profile["excel_col_letter"]

    return col_schema


class SchemaBuilder:
    """
    Converts a dict of enriched column profiles into a JSON Schema document.

    The builder is the final stage of the Metadata Extraction Pipeline.
    It expects profiles that have already passed through:

        DataFrameProfiler  →  CsvExcelExtractor (or SqlExtractor)
                           →  SemanticClassifier
                           →  LLMGenerator

    The resulting JSON Schema (draft-07) document is suitable for:
        - Storage alongside the dataset record in the database.
        - Return via the REST API for downstream consumers.
        - Dataset validation via ``jsonschema.validate()``.

    Usage::

        builder = SchemaBuilder(dataset_title="Customer Orders")
        schema  = builder.build(enriched_profiles)
        json_str = builder.to_json(schema, indent=2)
    """

    def __init__(
        self,
        dataset_title: str = "Dataset",
        dataset_description: str = "",
        additional_properties: bool = True,
    ):
        """
        Args:
            dataset_title:
                Human-readable title for the dataset; emitted as the top-level
                ``"title"`` keyword in the JSON Schema document.
            dataset_description:
                Optional description of the dataset as a whole.
            additional_properties:
                When ``False``, the schema will include
                ``"additionalProperties": false``, disallowing any column not
                listed in ``properties``.  Defaults to ``True`` for
                permissive validation.
        """
        self.dataset_title = dataset_title
        self.dataset_description = dataset_description
        self.additional_properties = additional_properties
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build(self, profiles: dict[str, dict]) -> dict[str, Any]:
        """
        Build a JSON Schema document from an enriched profiles dict.

        Iterates over every column in ``profiles``, constructs a property
        sub-schema for each, and assembles the top-level schema object.
        Columns that fail individual property-schema construction are
        logged and skipped — they will be absent from the output schema's
        ``properties`` but will not abort the build.

        Args:
            profiles:
                Output of ``LLMGenerator.enrich()`` — a dict keyed by column
                name, each value a fully enriched profile dict containing
                (at minimum) ``dtype``, ``semantic_type``, ``description``,
                ``business_name``, ``tags``, and ``notes``.

        Returns:
            A dict representing the complete JSON Schema document::

                {
                  "$schema": "http://json-schema.org/draft-07/schema#",
                  "title":   "Customer Orders",
                  "type":    "object",
                  "properties": { ... },
                  "required": ["order_id", "customer_id"],
                  "additionalProperties": True
                }

        Raises:
            ValueError: If ``profiles`` is empty.
        """
        if not profiles:
            raise ValueError(
                "SchemaBuilder.build received an empty profiles dict. "
                "Ensure the pipeline produced at least one column profile."
            )

        self.logger.info(
            "SchemaBuilder.build: building JSON Schema for '%s' from %d column profile(s).",
            self.dataset_title,
            len(profiles),
        )

        properties: dict[str, dict] = {}
        required_cols: list[str] = []

        for col, profile in profiles.items():
            try:
                col_schema = _build_column_schema(profile)
                properties[col] = col_schema

                # A column is "required" when it is explicitly NOT nullable
                # (is_nullable=False from SqlExtractor) AND has no observed
                # null values in the profiled data.
                is_nullable = profile.get("is_nullable", True)
                null_pct = profile.get("null_pct", 0.0) or 0.0
                if not is_nullable and null_pct == 0.0:
                    required_cols.append(col)

            except Exception as exc:
                self.logger.error(
                    "SchemaBuilder: failed to build property schema for column '%s': %s — skipping.",
                    col,
                    exc,
                )

        schema: dict[str, Any] = {
            "$schema": JSON_SCHEMA_DRAFT,
            "title":   self.dataset_title,
            "type":    "object",
        }

        if self.dataset_description:
            schema["description"] = self.dataset_description

        schema["properties"] = properties

        if required_cols:
            schema["required"] = required_cols

        schema["additionalProperties"] = self.additional_properties

        self.logger.info(
            "SchemaBuilder.build: completed schema with %d property/properties and "
            "%d required column(s).",
            len(properties),
            len(required_cols),
        )
        return schema

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def to_json(schema: dict[str, Any], indent: int = 2) -> str:
        """
        Serialise a schema dict produced by ``build()`` to a JSON string.

        Uses ``default=str`` so that any non-serialisable values (e.g.
        numpy scalars that slipped through) are safely converted rather
        than raising a ``TypeError``.

        Args:
            schema: Schema dict returned by ``build()``.
            indent: Indentation level for pretty-printing.  Pass ``None``
                    for compact output.

        Returns:
            JSON string representation of the schema.
        """
        return json.dumps(schema, indent=indent, ensure_ascii=False, default=str)

    @staticmethod
    def from_json(json_str: str) -> dict[str, Any]:
        """
        Deserialise a JSON string back into a schema dict.

        Useful for loading a stored schema from the database or a file.

        Args:
            json_str: JSON string produced by ``to_json()``.

        Returns:
            Schema dict.

        Raises:
            json.JSONDecodeError: If the input is not valid JSON.
        """
        return json.loads(json_str)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def schema_report(self, schema: dict[str, Any]) -> dict[str, Any]:
        """
        Return a summary of the built schema for logging or API responses.

        Args:
            schema: Schema dict returned by ``build()``.

        Returns:
            Dict with the following keys:
                title                (str)
                total_properties     (int)
                required_count       (int)
                type_distribution    (dict[str, int])
                semantic_types       (dict[str, int])
                tagged_columns       (dict[str, list[str]])  — col → tags
                nullable_count       (int)
        """
        properties: dict[str, dict] = schema.get("properties", {})
        required: list[str] = schema.get("required", [])

        type_dist: dict[str, int] = {}
        semantic_dist: dict[str, int] = {}
        tagged: dict[str, list[str]] = {}
        nullable_count = 0

        for col, col_schema in properties.items():
            raw_type = col_schema.get("type", "unknown")
            # Normalise union types like ["string", "null"] → "string (nullable)"
            if isinstance(raw_type, list):
                base = next((t for t in raw_type if t != "null"), "unknown")
                type_key = f"{base} (nullable)"
                nullable_count += 1
            else:
                type_key = raw_type

            type_dist[type_key] = type_dist.get(type_key, 0) + 1

            sem = col_schema.get("x-semantic-type", "unknown")
            semantic_dist[sem] = semantic_dist.get(sem, 0) + 1

            tags = col_schema.get("x-tags", [])
            if tags:
                tagged[col] = tags

        return {
            "title":             schema.get("title", ""),
            "total_properties":  len(properties),
            "required_count":    len(required),
            "type_distribution": dict(
                sorted(type_dist.items(), key=lambda kv: kv[1], reverse=True)
            ),
            "semantic_types":    dict(
                sorted(semantic_dist.items(), key=lambda kv: kv[1], reverse=True)
            ),
            "tagged_columns":    tagged,
            "nullable_count":    nullable_count,
        }