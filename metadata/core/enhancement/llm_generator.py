"""
LLM-powered metadata enrichment module.

Consumes the output of ``SemanticClassifier.classify()`` — a dict of
augmented, classified column profiles — and enriches each profile with
four LLM-generated fields:

    description   : plain-English explanation of the column's business purpose
    tags          : domain tags, e.g. ['PII', 'financial', 'temporal']
    business_name : human-friendly display name for the column
    notes         : data-quality concerns or usage caveats

Profiles are sent to the LLM in batches (to minimise round-trips) and
responses are parsed from strict JSON.  Three backends are supported:

    - OpenAI    (gpt-4o-mini by default)
    - Anthropic (claude-haiku-4-5-20251001 by default)
    - Ollama    (llama3 by default, local inference)
    - Gemini     (gemini-1.5-pro by default)

--------------------------------------------------------------------
CONFIGURATION — Django settings (settings.py)
--------------------------------------------------------------------

    # ----------------------------------------------------------------
    #    REQUIRED — choose one backend
    # ----------------------------------------------------------------
    LLM_BACKEND = "openai"          # "gemini" |"openai" | "anthropic" | "ollama"

    # ----------------------------------------------------------------
    #    API KEYS — set the key for whichever backend you choose.
    #     Never hard-code secrets here; use environment variables or
    #     a secrets manager and reference them like so:
    # ----------------------------------------------------------------
    import os

    # OpenAI
    # Get your key at https://platform.openai.com/api-keys
    OPENAI_API_KEY    = os.environ["OPENAI_API_KEY"]          #   REQUIRED for openai backend
    LLM_MODEL         = "gpt-4o-mini"                          # or "gpt-4o", "gpt-3.5-turbo"

    # Anthropic
    # Get your key at https://console.anthropic.com/settings/keys
    ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]       #   REQUIRED for anthropic backend
    LLM_MODEL         = "claude-haiku-4-5-20251001"            # or "claude-sonnet-4-6"

    # Ollama (local — no API key needed)
    OLLAMA_BASE_URL   = "http://localhost:11434"               # default Ollama address
    LLM_MODEL         = "llama3"                               # any model pulled via `ollama pull`

    # Gemini
    # Get your key at https://aistudio.google.com/app/apikey
    GEMINI_API_KEY    = os.environ["GEMINI_API_KEY"]          #   REQUIRED for gemini backend
    LLM_MODEL         = "gemini-1.5-pro"                       # or "gemini-1.5-flash"

    # ----------------------------------------------------------------
    # Optional tuning
    # ----------------------------------------------------------------
    LLM_BATCH_SIZE         = 10     # columns per LLM request  (default: 10)
    LLM_MAX_TOKENS         = 1024   # max tokens in LLM reply  (default: 1024)
    LLM_TEMPERATURE        = 0.2    # lower = more deterministic (default: 0.2)
    LLM_REQUEST_TIMEOUT    = 30     # seconds per HTTP request  (default: 30)
    LLM_MAX_RETRIES        = 3      # retry attempts on failure (default: 3)
    LLM_RETRY_BACKOFF      = 2.0    # exponential backoff base  (default: 2.0)

--------------------------------------------------------------------
Usage
--------------------------------------------------------------------

    from profiler import DataFrameProfiler
    from csv_excel_extractor import CsvExcelExtractor
    from semantic_classifier import SemanticClassifier
    from llm_generator import LLMGenerator

    profiles  = DataFrameProfiler(df).run()
    augmented = CsvExcelExtractor(df, source_format="csv").augment(profiles)
    classified = SemanticClassifier().classify(augmented)

    generator = LLMGenerator()          # reads Django settings automatically
    enriched  = generator.enrich(classified)

    # Each profile now has: description, tags, business_name, notes
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sentinel – used when Django settings key is absent
# ---------------------------------------------------------------------------
_MISSING = object()


def _setting(name: str, default: Any = _MISSING) -> Any:
    """
    Read a value from Django settings with an optional default.

    Raises ``ImproperlyConfigured`` when the key is absent and no default
    was supplied.
    """
    try:
        from django.conf import settings
        value = getattr(settings, name, _MISSING)
    except Exception:
        value = _MISSING

    if value is _MISSING:
        if default is _MISSING:
            from django.core.exceptions import ImproperlyConfigured
            raise ImproperlyConfigured(
                f"LLMGenerator requires '{name}' in Django settings. "
                f"See llm_generator.py module docstring for configuration instructions."
            )
        return default
    return value


# ---------------------------------------------------------------------------
# Default constants (overridable via Django settings)
# ---------------------------------------------------------------------------
_DEFAULT_BATCH_SIZE      = 10
_DEFAULT_MAX_TOKENS      = 1024
_DEFAULT_TEMPERATURE     = 0.2
_DEFAULT_TIMEOUT         = 30
_DEFAULT_MAX_RETRIES     = 3
_DEFAULT_RETRY_BACKOFF   = 2.0

# Tags the LLM may assign; included in the prompt so output is constrained.
ALLOWED_TAGS = [
    "PII",
    "financial",
    "temporal",
    "geographic",
    "identifier",
    "categorical",
    "metric",
    "boolean",
    "free_text",
    "derived",
    "sensitive",
    "nullable",
    "high_cardinality",
    "low_cardinality",
]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a data governance assistant.
Your job is to analyse structured column-profile metadata and produce
concise, accurate business-level annotations for each column.

Respond ONLY with a valid JSON object — no prose, no markdown fences.
The JSON must have exactly one top-level key per column name provided.
Each value must be an object with these four keys:
    "description"   : string  — one or two sentences explaining the column's business purpose.
    "tags"          : array   — zero or more tags chosen ONLY from this list: {allowed_tags}.
    "business_name" : string  — a human-friendly display name (title-case, spaces allowed).
    "notes"         : string  — data-quality concerns, usage caveats, or empty string if none.

Do not invent tags outside the allowed list.
Do not include any key not listed above.
"""

_USER_PROMPT_TEMPLATE = """\
Annotate the following {n} column(s).
Each entry shows the column name and its profile metadata as JSON.

{columns_json}

Return a single JSON object keyed by column name.
"""


def _build_system_prompt() -> str:
    return _SYSTEM_PROMPT.format(allowed_tags=json.dumps(ALLOWED_TAGS))


def _build_user_prompt(batch: dict[str, dict]) -> str:
    """Serialise a batch of profiles into the user prompt."""
    # We send a curated subset of the profile to keep the prompt concise
    # and avoid leaking internal implementation details to the LLM.
    curated: dict[str, dict] = {}
    for col, profile in batch.items():
        curated[col] = _curate_profile(profile)

    columns_json = json.dumps(curated, indent=2, default=str)
    return _USER_PROMPT_TEMPLATE.format(
        n=len(batch),
        columns_json=columns_json,
    )


def _curate_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """
    Extract the subset of profile keys most useful for LLM annotation.

    Strips internal/numeric-only keys that add noise to the prompt.
    """
    keys_of_interest = [
        "dtype",
        "semantic_type",
        "semantic_confidence",
        "null_rate",
        "unique_rate",
        "is_primary_key",
        "is_foreign_key",
        "is_nullable",
        "is_currency",
        "is_percentage",
        "detected_date_format",
        "native_sql_type",
        "fk_references",
        "is_indexed",
        "excel_col_letter",
        "multi_header_row",
        "mean",
        "std",
        "min",
        "max",
        "top_values",        # if the profiler includes a frequency table
        "sample_values",     # if the profiler includes sample values
    ]
    return {k: profile[k] for k in keys_of_interest if k in profile}


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_ENRICHMENT_KEYS = ("description", "tags", "business_name", "notes")


def _parse_llm_response(
    raw: str, expected_columns: list[str]
) -> dict[str, dict[str, Any]]:
    """
    Parse the LLM's raw text response into a dict keyed by column name.

    Gracefully handles:
        - Leading/trailing whitespace
        - Markdown code fences (```json ... ```)
        - Missing columns (fills with safe defaults)
        - Extra columns (silently ignored)
        - Malformed JSON (raises ``LLMParseError``)

    Args:
        raw:              Raw text returned by the LLM.
        expected_columns: Column names that should appear in the response.

    Returns:
        Dict mapping column name → enrichment dict with keys
        ``description``, ``tags``, ``business_name``, ``notes``.

    Raises:
        LLMParseError: JSON could not be decoded.
    """
    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove opening fence (```json or ```) and closing fence
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMParseError(
            f"LLM response could not be decoded as JSON: {exc}\n"
            f"Raw response (first 500 chars): {raw[:500]}"
        ) from exc

    if not isinstance(parsed, dict):
        raise LLMParseError(
            f"Expected a JSON object at the top level; got {type(parsed).__name__}."
        )

    result: dict[str, dict[str, Any]] = {}
    for col in expected_columns:
        col_data = parsed.get(col, {})
        result[col] = {
            "description":   str(col_data.get("description",   "")),
            "tags":          _sanitise_tags(col_data.get("tags", [])),
            "business_name": str(col_data.get("business_name", _default_business_name(col))),
            "notes":         str(col_data.get("notes",         "")),
        }
    return result


def _sanitise_tags(raw_tags: Any) -> list[str]:
    """Filter tags to the allowed set; coerce to list[str] defensively."""
    if not isinstance(raw_tags, list):
        return []
    allowed = set(ALLOWED_TAGS)
    return [str(t) for t in raw_tags if str(t) in allowed]


def _default_business_name(col: str) -> str:
    """Generate a title-case display name from a snake_case column name."""
    return col.replace("_", " ").replace("-", " ").title()


def _safe_defaults(col: str) -> dict[str, Any]:
    """Return empty-but-valid enrichment fields for a column."""
    return {
        "description":   "",
        "tags":          [],
        "business_name": _default_business_name(col),
        "notes":         "LLM enrichment unavailable — using defaults.",
    }


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Base class for LLM generator errors."""


class LLMConfigError(LLMError):
    """Raised when the LLM backend is misconfigured (e.g. missing API key)."""


class LLMParseError(LLMError):
    """Raised when the LLM response cannot be parsed as valid JSON."""


class LLMRequestError(LLMError):
    """Raised when an API request fails after all retries are exhausted."""


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------

class _LLMBackend(ABC):
    """
    Abstract base for LLM backend adapters.

    Each subclass wraps one provider's SDK/HTTP interface and exposes a
    single ``complete()`` method that accepts a system prompt + user prompt
    and returns the model's text response.
    """

    def __init__(
        self,
        model: str,
        max_tokens: int,
        temperature: float,
        timeout: int,
    ):
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.timeout     = timeout

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send a prompt to the LLM and return the raw text response.

        Args:
            system_prompt: Instructions for the model role/output format.
            user_prompt:   The actual data/question to process.

        Returns:
            Raw text string from the model.

        Raises:
            LLMRequestError: On HTTP/network/API errors.
        """



class _GeminiBackend(_LLMBackend):
    """
    Google Gemini backend (Google GenAI SDK).

       Requires:
        pip install google-genai>=1.0.0

       API key:
        Set GEMINI_API_KEY in Django settings (loaded from os.environ).
        Obtain at: https://aistudio.google.com/app/apikey
    """

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        #    API KEY — passed in from Django settings; never hard-code here.
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise LLMConfigError(
                "Gemini backend requires the 'google-genai' package. "
                "Install it with: pip install google-genai>=1.0.0"
            ) from exc
        self._types = types
        self._client = genai.Client(api_key=api_key)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=self._types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                    response_mime_type="application/json",  # enforces JSON mode
                ),
            )
            return response.text or ""
        except Exception as exc:
            raise LLMRequestError(f"Gemini API request failed: {exc}") from exc



class _OpenAIBackend(_LLMBackend):
    """
    OpenAI ChatCompletion backend.

       Requires:
        pip install openai>=1.0.0

       API key:
        Set OPENAI_API_KEY in Django settings (loaded from os.environ).
        Obtain at: https://platform.openai.com/api-keys
    """

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        #    API KEY — passed in from Django settings; never hard-code here.
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMConfigError(
                "OpenAI backend requires the 'openai' package. "
                "Install it with: pip install openai>=1.0.0"
            ) from exc
        self._client = OpenAI(api_key=api_key, timeout=self.timeout)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"},  # enforces JSON mode
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise LLMRequestError(f"OpenAI API request failed: {exc}") from exc


class _AnthropicBackend(_LLMBackend):
    """
    Anthropic Messages API backend.

       Requires:
        pip install anthropic>=0.25.0

       API key:
        Set ANTHROPIC_API_KEY in Django settings (loaded from os.environ).
        Obtain at: https://console.anthropic.com/settings/keys
    """

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        #    API KEY — passed in from Django settings; never hard-code here.
        try:
            import anthropic as anthropic_sdk
        except ImportError as exc:
            raise LLMConfigError(
                "Anthropic backend requires the 'anthropic' package. "
                "Install it with: pip install anthropic>=0.25.0"
            ) from exc
        self._sdk = anthropic_sdk
        self._client = anthropic_sdk.Anthropic(
            api_key=api_key,
            timeout=float(self.timeout),
        )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=self.temperature,
            )
            # Extract text from the first content block
            for block in message.content:
                if block.type == "text":
                    return block.text
            return ""
        except Exception as exc:
            raise LLMRequestError(f"Anthropic API request failed: {exc}") from exc


class _OllamaBackend(_LLMBackend):
    """
    Ollama local-inference backend (HTTP REST API).

       Requires:
        - Ollama running locally: https://ollama.com/download
        - Target model pulled:    ollama pull llama3
        - No API key needed for local inference.

       Set OLLAMA_BASE_URL in Django settings if Ollama is not on
        localhost:11434 (e.g. a remote GPU server).
    """

    def __init__(self, base_url: str, **kwargs):
        super().__init__(**kwargs)
        #    BASE URL — set OLLAMA_BASE_URL in Django settings.
        self._base_url = base_url.rstrip("/")
        try:
            import httpx
            self._httpx = httpx
        except ImportError as exc:
            raise LLMConfigError(
                "Ollama backend requires the 'httpx' package. "
                "Install it with: pip install httpx"
            ) from exc

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self._base_url}/api/chat"
        payload = {
            "model": self.model,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        }
        try:
            resp = self._httpx.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except Exception as exc:
            raise LLMRequestError(
                f"Ollama request to '{url}' failed: {exc}. "
                f"Is Ollama running? Try: ollama serve"
            ) from exc


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def _build_backend(
    backend_name: str,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> _LLMBackend:
    """
    Instantiate the correct ``_LLMBackend`` subclass from Django settings.

    Args:
        backend_name: ``"gemini"``, ``"openai"``, ``"anthropic"``, or ``"ollama"``.
        model:        Model identifier string.
        max_tokens:   Maximum tokens for LLM response.
        temperature:  Sampling temperature.
        timeout:      HTTP timeout in seconds.

    Returns:
        Configured ``_LLMBackend`` instance.

    Raises:
        LLMConfigError: Unknown backend or missing required settings key.
    """
    name = backend_name.lower()
    common = dict(model=model, max_tokens=max_tokens, temperature=temperature, timeout=timeout)
    

    if name == "openai":
        #    REQUIRES: OPENAI_API_KEY in Django settings
        api_key = _setting("OPENAI_API_KEY")
        return _OpenAIBackend(api_key=api_key, **common)

    if name == "anthropic":
        #    REQUIRES: ANTHROPIC_API_KEY in Django settings
        api_key = _setting("ANTHROPIC_API_KEY")
        return _AnthropicBackend(api_key=api_key, **common)

    if name == "ollama":
        # No API key — but OLLAMA_BASE_URL must point to your Ollama server.
        base_url = _setting("OLLAMA_BASE_URL", default="http://localhost:11434")
        return _OllamaBackend(base_url=base_url, **common)
    
    if name == "gemini":
        #    REQUIRES: GEMINI_API_KEY in Django settings
        api_key = _setting("GEMINI_API_KEY")
        return _GeminiBackend(api_key=api_key, **common)

    raise LLMConfigError(
        f"Unknown LLM_BACKEND '{backend_name}'. "
        f"Supported values: 'openai', 'anthropic', 'ollama', 'gemini'."
    )


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _with_retries(
    fn,
    max_retries: int,
    backoff_base: float,
    logger_: logging.Logger,
    label: str,
):
    """
    Call ``fn()`` up to ``max_retries`` times with exponential back-off.

    Args:
        fn:          Zero-argument callable that may raise ``LLMRequestError``.
        max_retries: Maximum number of attempts (1 = no retry).
        backoff_base: Multiplier for exponential sleep: sleep = base ** attempt.
        logger_:     Logger instance for warning messages.
        label:       Human-readable label for log messages.

    Returns:
        Return value of ``fn()`` on success.

    Raises:
        LLMRequestError: If all attempts fail.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except LLMRequestError as exc:
            last_exc = exc
            if attempt < max_retries:
                sleep_secs = backoff_base ** attempt
                logger_.warning(
                    "%s: attempt %d/%d failed — retrying in %.1fs. Error: %s",
                    label, attempt, max_retries, sleep_secs, exc,
                )
                time.sleep(sleep_secs)
            else:
                logger_.error(
                    "%s: all %d attempts failed. Last error: %s",
                    label, max_retries, exc,
                )
    raise LLMRequestError(
        f"{label}: all {max_retries} attempts failed."
    ) from last_exc


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

class LLMGenerator:
    """
    Enriches classified column profiles with LLM-generated metadata.

    Consumes the output of ``SemanticClassifier.classify()`` and injects
    four new keys into each profile dict:

        description   (str)        plain-English column purpose
        tags          (list[str])  domain tags from ALLOWED_TAGS
        business_name (str)        human-friendly display name
        notes         (str)        quality concerns / usage caveats

    Profiles are grouped into batches before being sent to the LLM to
    minimise the number of API round-trips.

       CONFIGURATION — see module docstring for full settings reference.
    The minimum required settings are:
        LLM_BACKEND  and the corresponding API key setting.

    Usage
    -----
        generator = LLMGenerator()
        enriched  = generator.enrich(classified_profiles)
    """

    def __init__(self):
        # --- Read settings with documented defaults ---
        backend_name = _setting("LLM_BACKEND")                           #   REQUIRED
        model        = _setting("LLM_MODEL",          default=_default_model(backend_name if isinstance(backend_name, str) else "gemini"))
        max_tokens   = int(_setting("LLM_MAX_TOKENS",      default=_DEFAULT_MAX_TOKENS))
        temperature  = float(_setting("LLM_TEMPERATURE",   default=_DEFAULT_TEMPERATURE))
        timeout      = int(_setting("LLM_REQUEST_TIMEOUT", default=_DEFAULT_TIMEOUT))

        self.batch_size   = int(_setting("LLM_BATCH_SIZE",     default=_DEFAULT_BATCH_SIZE))
        self.max_retries  = int(_setting("LLM_MAX_RETRIES",    default=_DEFAULT_MAX_RETRIES))
        self.retry_backoff = float(_setting("LLM_RETRY_BACKOFF", default=_DEFAULT_RETRY_BACKOFF))

        self._backend = _build_backend(
            backend_name=backend_name,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        self._system_prompt = _build_system_prompt()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.info(
            "LLMGenerator initialised: backend=%s model=%s batch_size=%d",
            backend_name, model, self.batch_size,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def enrich(self, profiles: dict[str, dict]) -> dict[str, dict]:
        """
        Enrich all column profiles with LLM-generated metadata.

        Sends profiles to the LLM in batches, parses JSON responses, and
        injects ``description``, ``tags``, ``business_name``, and ``notes``
        into each profile dict in-place.

        When a batch request fails after all retries, or the response
        cannot be parsed, safe defaults are injected and a warning is
        logged — the pipeline continues rather than raising.

        Args:
            profiles:
                Output of ``SemanticClassifier.classify()`` — a dict keyed
                by column name, each value a profile dict that already
                contains ``semantic_type`` and ``semantic_confidence``.

        Returns:
            The same ``profiles`` dict with enrichment keys injected.
        """
        if not profiles:
            self.logger.warning("LLMGenerator.enrich called with empty profiles dict.")
            return profiles

        col_names = list(profiles.keys())
        batches   = _chunk(col_names, self.batch_size)

        self.logger.info(
            "LLMGenerator.enrich: enriching %d column(s) in %d batch(es).",
            len(col_names), len(batches),
        )

        for batch_num, batch_cols in enumerate(batches, start=1):
            batch = {col: profiles[col] for col in batch_cols}
            self.logger.debug(
                "LLMGenerator: processing batch %d/%d (%d columns): %s",
                batch_num, len(batches), len(batch_cols), batch_cols,
            )
            self._enrich_batch(batch, profiles)

        self.logger.info("LLMGenerator.enrich: all batches processed.")
        return profiles

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def _enrich_batch(
        self,
        batch: dict[str, dict],
        profiles: dict[str, dict],
    ) -> None:
        """
        Send one batch to the LLM, parse the response, and update profiles.

        On any failure (request error or parse error) safe defaults are
        applied to every column in the batch so the pipeline keeps running.

        Args:
            batch:    Subset of profiles for this batch (col → profile dict).
            profiles: Full profiles dict — updated in-place with enrichment.
        """
        user_prompt = _build_user_prompt(batch)
        batch_cols  = list(batch.keys())

        try:
            raw = _with_retries(
                fn=lambda: self._backend.complete(self._system_prompt, user_prompt),
                max_retries=self.max_retries,
                backoff_base=self.retry_backoff,
                logger_=self.logger,
                label=f"LLMGenerator batch [{', '.join(batch_cols[:3])}{'…' if len(batch_cols) > 3 else ''}]",
            )
        except LLMRequestError as exc:
            self.logger.error(
                "LLMGenerator: batch request permanently failed — "
                "applying safe defaults to %d column(s). Error: %s",
                len(batch_cols), exc,
            )
            for col in batch_cols:
                profiles[col].update(_safe_defaults(col))
            return

        try:
            enrichments = _parse_llm_response(raw, batch_cols)
        except LLMParseError as exc:
            self.logger.error(
                "LLMGenerator: could not parse LLM response for batch %s — "
                "applying safe defaults. Error: %s",
                batch_cols, exc,
            )
            for col in batch_cols:
                profiles[col].update(_safe_defaults(col))
            return

        # Inject enrichment into profiles
        for col, enrichment in enrichments.items():
            profiles[col].update(enrichment)
            self.logger.debug(
                "Enriched column '%s': business_name='%s' tags=%s",
                col, enrichment.get("business_name"), enrichment.get("tags"),
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def enrichment_report(self, profiles: dict[str, dict]) -> dict[str, Any]:
        """
        Return a summary report of LLM enrichment coverage.

        Useful for validating pipeline output or debugging low-quality
        enrichments in development.

        Args:
            profiles: Profiles dict after ``enrich()`` has been called.

        Returns:
            Dict with keys:
                total_columns      (int)
                enriched_columns   (int)  — have a non-empty description
                unenriched_columns (list[str])
                tag_frequency      (dict[str, int])
        """
        total       = len(profiles)
        enriched    = 0
        unenriched  : list[str] = []
        tag_freq    : dict[str, int] = {}

        for col, profile in profiles.items():
            desc = profile.get("description", "")
            if desc and "unavailable" not in desc:
                enriched += 1
            else:
                unenriched.append(col)
            for tag in profile.get("tags", []):
                tag_freq[tag] = tag_freq.get(tag, 0) + 1

        return {
            "total_columns":      total,
            "enriched_columns":   enriched,
            "unenriched_columns": unenriched,
            "tag_frequency":      dict(
                sorted(tag_freq.items(), key=lambda kv: kv[1], reverse=True)
            ),
        }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _chunk(lst: list, size: int) -> list[list]:
    """Split ``lst`` into consecutive sub-lists of at most ``size`` items."""
    return [lst[i: i + size] for i in range(0, len(lst), size)]


def _default_model(backend: str) -> str:
    """Return a sensible default model name for a given backend."""
    return {
        "openai":    "gpt-4o-mini",
        "anthropic": "claude-haiku-4-5-20251001",
        "ollama":    "llama3",
        "gemini":    "gemini-1.5-pro",
    }.get(backend.lower(), "gpt-4o-mini")