"""
ML-based semantic type classifier for dataset columns.

Takes a list of ColumnProfile dicts (augmented by CsvExcelExtractor or
SqlExtractor) and assigns a ``semantic_type`` label to each column.
Semantic types go beyond primitive dtypes to capture business meaning:

    int     → 'age', 'count', 'id', 'year'
    string  → 'email', 'phone', 'address', 'name', 'url', 'category'
    float   → 'currency', 'percentage', 'latitude', 'longitude', 'score'
    bool    → 'boolean'

Strategy
--------
1. **Rule-based fast path** — high-confidence heuristics derived from
   column-name tokens, SQL metadata (PK/FK flags, native type), and
   extractor flags (is_currency, is_percentage, detected_date_format).
   Rules are checked first and short-circuit the ML step when they fire.

2. **ML classifier** — a lightweight scikit-learn ``SGDClassifier`` (log
   loss ≈ logistic regression) trained on a hand-crafted feature vector
   built from the augmented profile.  Used when no rule fires.

3. **Confidence threshold** — if the ML model's top-class probability is
   below ``ML_CONFIDENCE_THRESHOLD`` the classifier falls back to the
   generic type derived from the column's pandas dtype.

The classifier is *self-contained*: it ships with synthetic training data
so it can run without an external model artefact.  In production ``_build_training_data()`` 
will be replaced with a call that loads a pre-trained ``joblib`` model from disk.
 
Usage
-----
    from profiler import DataFrameProfiler
    from csv_excel_extractor import CsvExcelExtractor   # or SqlExtractor
    from semantic_classifier import SemanticClassifier

    profiles = DataFrameProfiler(df).run()
    extractor = CsvExcelExtractor(df, source_format="csv")
    augmented = extractor.augment(profiles)

    classifier = SemanticClassifier()
    classified = classifier.classify(augmented)

    # Each profile now has a 'semantic_type' and 'semantic_confidence' key.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Semantic type labels
# ---------------------------------------------------------------------------
SEMANTIC_TYPES = [
    "id",
    "name",
    "email",
    "phone",
    "url",
    "address",
    "age",
    "year",
    "date",
    "datetime",
    "boolean",
    "category",
    "count",
    "currency",
    "percentage",
    "score",
    "latitude",
    "longitude",
    "description",
    "unknown",
]

# ---------------------------------------------------------------------------
# Confidence threshold below which ML output is replaced by a dtype-derived
# fallback label.
# ---------------------------------------------------------------------------
ML_CONFIDENCE_THRESHOLD = 0.40

# ---------------------------------------------------------------------------
# Column-name token patterns for rule-based fast path.
# Each entry: (compiled regex, semantic_type)
# Evaluated in order; first match wins.
# ---------------------------------------------------------------------------
_NAME_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(email|e_mail|e-mail)\b",        re.I), "email"),
    (re.compile(r"\b(phone|mobile|cell|tel)\b",      re.I), "phone"),
    (re.compile(r"\b(url|uri|link|href|website|site)\b", re.I), "url"),
    (re.compile(r"\b(lat|latitude)\b",               re.I), "latitude"),
    (re.compile(r"\b(lon|lng|longitude)\b",          re.I), "longitude"),
    (re.compile(r"\b(address|addr|street|city|state|zip|postcode|postal)\b", re.I), "address"),
    (re.compile(r"\b(age)\b",                        re.I), "age"),
    (re.compile(r"\b(year|yr)\b",                    re.I), "year"),
    (re.compile(r"\b(date|day|month)\b",             re.I), "date"),
    (re.compile(r"\b(datetime|timestamp|created_at|updated_at|deleted_at)\b", re.I), "datetime"),
    (re.compile(r"\b(name|full_name|first_name|last_name|fname|lname|surname)\b", re.I), "name"),
    (re.compile(r"\b(desc|description|notes?|comment|remarks?|summary|bio)\b", re.I), "description"),
    (re.compile(r"\b(is_|has_|flag_|active|enabled|deleted|visible)\b",       re.I), "boolean"),
    (re.compile(r"\b(count|qty|quantity|num|number_of|total)\b",              re.I), "count"),
    (re.compile(r"\b(score|rating|rank|percentile|gpa|grade)\b",              re.I), "score"),
    (re.compile(r"\b(price|amount|cost|revenue|salary|fee|balance|payment|currency|spend)\b", re.I), "currency"),
    (re.compile(r"\b(pct|percent|percentage|rate)\b", re.I), "percentage"),
    (re.compile(r"\b(category|cat|type|kind|group|class|genre|segment)\b",    re.I), "category"),
    # ID last — broad pattern, avoid masking more specific matches above
    (re.compile(r"(^id$|_id$|^id_|\bid\b)",          re.I), "id"),
]

# ---------------------------------------------------------------------------
# dtype → fallback semantic type when ML confidence is too low
# ---------------------------------------------------------------------------
_DTYPE_FALLBACK: dict[str, str] = {
    "int":      "count",
    "float":    "score",
    "bool":     "boolean",
    "datetime": "datetime",
    "object":   "category",
    "string":   "category",
}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class _FeatureExtractor:
    """
    Converts an augmented column profile dict into a fixed-length numeric
    feature vector for the ML classifier.

    Feature index map (23 features total):
        [0]  name contains digit
        [1]  name length (log-scaled)
        [2]  name token count
        [3]  dtype is int
        [4]  dtype is float
        [5]  dtype is bool
        [6]  dtype is datetime
        [7]  dtype is object/string
        [8]  null_rate                 (0-1)
        [9]  unique_rate               (0-1)
        [10] is_primary_key            (SQL extractor)
        [11] is_foreign_key            (SQL extractor)
        [12] is_nullable               (SQL extractor)
        [13] is_indexed                (SQL extractor)
        [14] is_unique_indexed         (SQL extractor)
        [15] is_currency               (CSV/Excel extractor)
        [16] is_percentage             (CSV/Excel extractor)
        [17] has detected_date_format  (CSV/Excel extractor)
        [18] mean value (clipped 0-1 via sigmoid)
        [19] std dev  (clipped 0-1 via sigmoid)
        [20] min value < 0            (bool)
        [21] max value < 1000         (bool – hints at score/rate)
        [22] max value > 1_000_000    (bool – hints at currency/id)
    """

    N_FEATURES = 23

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-float(x)))

    def transform(self, profile: dict[str, Any]) -> np.ndarray:
        vec = np.zeros(self.N_FEATURES, dtype=np.float32)
        name: str = str(profile.get("column_name") or profile.get("name") or "")
        dtype: str = str(profile.get("dtype") or "object").lower()

        # --- Name features ---
        vec[0] = float(bool(re.search(r"\d", name)))
        vec[1] = float(np.log1p(len(name)))
        tokens = re.split(r"[\s_\-]+", name.strip())
        vec[2] = float(min(len(tokens), 8))

        # --- dtype flags ---
        vec[3] = float("int" in dtype)
        vec[4] = float("float" in dtype)
        vec[5] = float("bool" in dtype)
        vec[6] = float("datetime" in dtype or "date" in dtype)
        vec[7] = float("object" in dtype or "string" in dtype or "str" in dtype)

        # --- Statistical features ---
        null_rate  = float(profile.get("null_rate")   or 0.0)
        unique_rate = float(profile.get("unique_rate") or 0.0)
        vec[8]  = max(0.0, min(1.0, null_rate))
        vec[9]  = max(0.0, min(1.0, unique_rate))

        # --- SQL extractor features ---
        vec[10] = float(bool(profile.get("is_primary_key",    False)))
        vec[11] = float(bool(profile.get("is_foreign_key",    False)))
        vec[12] = float(bool(profile.get("is_nullable",       True)))
        vec[13] = float(bool(profile.get("is_indexed",        False)))
        vec[14] = float(bool(profile.get("is_unique_indexed", False)))

        # --- CSV/Excel extractor features ---
        vec[15] = float(bool(profile.get("is_currency",  False)))
        vec[16] = float(bool(profile.get("is_percentage", False)))
        vec[17] = float(profile.get("detected_date_format") is not None)

        # --- Numeric distribution features ---
        mean_val = profile.get("mean")
        std_val  = profile.get("std")
        min_val  = profile.get("min")
        max_val  = profile.get("max")

        vec[18] = self._sigmoid(float(mean_val)) if mean_val is not None else 0.5
        vec[19] = self._sigmoid(float(std_val))  if std_val  is not None else 0.5
        vec[20] = float(float(min_val) < 0)      if min_val  is not None else 0.0
        vec[21] = float(float(max_val) < 1_000)  if max_val  is not None else 0.0
        vec[22] = float(float(max_val) > 1_000_000) if max_val is not None else 0.0

        return vec


# ---------------------------------------------------------------------------
# Synthetic training data
# ---------------------------------------------------------------------------

def _build_training_data() -> tuple[np.ndarray, list[str]]:
    """
    Generate a hand-crafted synthetic training set.

    Each sample is defined as a ``(feature_overrides_dict, label)`` pair.
    The feature extractor is not used here; instead we directly set the
    feature vector indices that most strongly signal each class.

    In production replace this function with::

        import joblib
        model = joblib.load("semantic_classifier.joblib")
        return model  # skip _train() entirely

    Returns:
        X: np.ndarray of shape (n_samples, N_FEATURES)
        y: list of label strings
    """
    N = _FeatureExtractor.N_FEATURES
    samples: list[tuple[np.ndarray, str]] = []

    def _make(overrides: dict[int, float], label: str, repeat: int = 6) -> None:
        for _ in range(repeat):
            vec = np.zeros(N, dtype=np.float32)
            # Add slight Gaussian noise for variance
            vec += np.random.default_rng(42).normal(0, 0.02, N).astype(np.float32)
            for idx, val in overrides.items():
                vec[idx] = val
            samples.append((vec, label))

    # id
    _make({3: 1, 9: 1.0, 10: 1, 14: 1, 0: 0},      "id")
    _make({3: 1, 9: 0.95, 11: 1},                   "id")

    # name
    _make({7: 1, 9: 0.9, 2: 2},                     "name")
    _make({7: 1, 9: 0.8, 2: 3, 8: 0.05},            "name")

    # email
    _make({7: 1, 9: 0.95, 2: 1},                    "email")

    # phone
    _make({7: 1, 9: 0.9, 0: 1},                     "phone")

    # url
    _make({7: 1, 9: 0.9, 1: 3.5},                   "url")

    # address
    _make({7: 1, 9: 0.5, 2: 3, 8: 0.1},             "address")

    # age
    _make({3: 1, 18: 0.55, 19: 0.3, 21: 1},         "age")
    _make({4: 1, 18: 0.55, 21: 1, 20: 0},           "age")

    # year
    _make({3: 1, 18: 0.7, 21: 0, 22: 0, 19: 0.1},  "year")

    # date
    _make({17: 1, 6: 0},                             "date")
    _make({6: 1},                                    "date")

    # datetime
    _make({17: 1, 6: 1},                             "datetime")
    _make({6: 1, 7: 0, 17: 1},                      "datetime")

    # boolean
    _make({5: 1},                                    "boolean")
    _make({7: 1, 9: 0.5, 21: 1, 18: 0.5},           "boolean")

    # category
    _make({7: 1, 9: 0.1, 2: 1, 8: 0.05},            "category")
    _make({7: 1, 9: 0.05},                           "category")

    # count
    _make({3: 1, 18: 0.4, 21: 1, 20: 0, 19: 0.4},  "count")

    # currency
    _make({15: 1, 4: 1, 22: 0},                     "currency")
    _make({15: 1, 3: 1},                             "currency")
    _make({4: 1, 22: 1, 15: 0},                     "currency")

    # percentage
    _make({16: 1},                                   "percentage")
    _make({4: 1, 21: 1, 18: 0.5, 16: 0},            "percentage")

    # score
    _make({4: 1, 21: 1, 18: 0.6, 20: 0, 16: 0},    "score")
    _make({4: 1, 9: 0.8, 21: 1, 18: 0.5},           "score")

    # latitude
    _make({4: 1, 18: 0.5, 21: 1, 20: 1},            "latitude")

    # longitude
    _make({4: 1, 18: 0.5, 21: 1, 20: 1, 19: 0.6},  "longitude")

    # description
    _make({7: 1, 9: 0.99, 2: 4, 8: 0.2, 1: 3.5},   "description")

    # unknown
    _make({},                                        "unknown")

    X = np.vstack([s[0] for s in samples])
    y = [s[1] for s in samples]
    return X, y


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

class SemanticClassifier:
    """
    Assigns a ``semantic_type`` label to each column profile dict.

    Pipeline
    --------
    1. **Rule-based fast path** — always runs first.  High-confidence
       heuristics from column-name tokens, SQL metadata, and extractor flags.
       Returns with ``confidence=1.0`` when a rule fires; short-circuits all
       subsequent stages.

    2. **ML classifier** — runs only when a ``.joblib`` model is successfully
       loaded *and* no rule fired.  Skipped entirely when the model file is
       absent or fails to load (rule-based-only mode).

    3. **Dtype-derived fallback** — used when ML confidence is below
       ``ML_CONFIDENCE_THRESHOLD``, or when no model is available.

    If the model file is missing the classifier degrades gracefully:
    a warning is logged and stages 1 + 3 continue to operate normally.
    Use ``classifier.model_loaded`` to check which mode is active.

    After ``classify()`` each profile dict gains two new keys:

        semantic_type        (str)   — one of ``SEMANTIC_TYPES``
        semantic_confidence  (float) — probability in [0, 1]; 1.0 for rules,
                                       0.0 when dtype fallback is used without
                                       a model.

    Usage
    -----
        classifier = SemanticClassifier()
        classified = classifier.classify(augmented_profiles)
    """
    MODEL_PATH = Path(__file__).parent / "artifacts" / "semantic_classifier.joblib"

    def __init__(self, confidence_threshold: float = ML_CONFIDENCE_THRESHOLD, model_path=MODEL_PATH):
        """
        Args:
            confidence_threshold:
                ML probability below which the dtype-derived fallback label
                is used instead of the model's top prediction.
            model_path:
                Path to a pre-trained joblib model artefact.  If the file does
                not exist or fails to load, the classifier operates in
                **rule-based-only mode**: rules → dtype fallback.  No exception
                is raised; a warning is logged instead.
        """
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
        self._feature_extractor = _FeatureExtractor()
        # Bug fix: was assigned to self.model (no underscore), but _apply_ml
        # and _classify_column both reference self._model.  Unified to _model.
        self._model = self._load(model_path)

    @staticmethod
    def _load(path):
        """
        Attempt to load a pre-trained joblib model from *path*.

        Returns the fitted pipeline on success, or ``None`` if the file is
        absent or cannot be loaded.  A ``None`` return puts the classifier
        into rule-based-only mode — no exception is propagated.
        """
        try:
            import joblib
            model = joblib.load(path)
            logger.info("SemanticClassifier: loaded model from %s", path)
            return model
        except FileNotFoundError:
            logger.warning(
                "SemanticClassifier: model file not found at '%s'. "
                "Running in rule-based-only mode (rules → dtype fallback).",
                path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "SemanticClassifier: could not load model from '%s' (%s: %s). "
                "Running in rule-based-only mode (rules → dtype fallback).",
                path, type(exc).__name__, exc,
            )
        return None

    @property
    def model_loaded(self) -> bool:
        """``True`` when a trained ML model is available, ``False`` otherwise."""
        return self._model is not None


    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def classify(self, profiles: dict[str, dict]) -> dict[str, dict]:
        """
        Classify all columns in ``profiles`` and inject semantic type keys.

        Operates in-place on each inner dict and returns the same mapping.

        Args:
            profiles:
                Augmented profile dict from ``CsvExcelExtractor.augment()``
                or ``SqlExtractor.augment()``.  Keys are column names;
                values are profile dicts.

        Returns:
            The same ``profiles`` dict with ``semantic_type`` and
            ``semantic_confidence`` injected per column.
        """
        self.logger.info(
            "SemanticClassifier.classify: classifying %d column(s) "
            "[mode: %s].",
            len(profiles),
            "rule-based + ML" if self._model is not None else "rule-based only",
        )

        for col, profile in profiles.items():
            # Ensure column_name is available in the profile for feature extraction.
            if "column_name" not in profile and "name" not in profile:
                profile["column_name"] = col

            sem_type, confidence = self._classify_column(col, profile)
            profile["semantic_type"]       = sem_type
            profile["semantic_confidence"] = round(confidence, 4)

            self.logger.debug(
                "Column '%s' → semantic_type='%s' (confidence=%.2f)",
                col, sem_type, confidence,
            )

        self.logger.info(
            "SemanticClassifier.classify: done. "
            "Type distribution: %s",
            self._type_distribution(profiles),
        )
        return profiles

    # ------------------------------------------------------------------
    # Internal classification logic
    # ------------------------------------------------------------------

    def _classify_column(
        self, col: str, profile: dict[str, Any]
    ) -> tuple[str, float]:
        """
        Classify a single column, returning (semantic_type, confidence).

        Pipeline:
            1. Rule-based fast path (always runs first).
            2. ML classifier — only when a model is loaded AND a rule did not
               fire.  Skipped entirely if ``self._model is None``.
            3. Dtype-derived fallback — used when ML confidence is below the
               threshold, or when no model is available.
        """
        # Stage 1: rule-based fast path (always first)
        rule_type = self._apply_rules(col, profile)
        if rule_type is not None:
            return rule_type, 1.0

        # Stage 2: ML classifier (only when a model is present)
        if self._model is not None:
            ml_type, ml_conf = self._apply_ml(profile)
            if ml_conf >= self.confidence_threshold:
                return ml_type, ml_conf
            # ML fired but confidence too low → dtype fallback
            fallback = self._dtype_fallback(profile)
            self.logger.debug(
                "Column '%s': ML confidence %.2f below threshold — "
                "using dtype fallback '%s'.",
                col, ml_conf, fallback,
            )
            return fallback, ml_conf  # surface the actual (low) confidence

        # Stage 3: no model available — dtype fallback
        fallback = self._dtype_fallback(profile)
        self.logger.debug(
            "Column '%s': no ML model — using dtype fallback '%s'.",
            col, fallback,
        )
        return fallback, 0.0


    def _apply_rules(self, col: str, profile: dict[str, Any]) -> str | None:
        """
        Return a semantic type string if a high-confidence rule fires,
        otherwise return None.

        Rule priority (highest → lowest):
            1. SQL extractor signals (is_primary_key, is_foreign_key)
            2. CSV/Excel extractor flags (is_currency, is_percentage,
               detected_date_format)
            3. Dtype is bool
            4. Column-name token matching
        """
        # --- SQL metadata signals ---
        if profile.get("is_primary_key"):
            return "id"
        if profile.get("is_foreign_key"):
            return "id"

        # --- Extractor flags ---
        if profile.get("is_currency"):
            return "currency"
        if profile.get("is_percentage"):
            return "percentage"

        detected_fmt = profile.get("detected_date_format")
        if detected_fmt is not None:
            # ISO_DATETIME / SLASH_DMY with time component → datetime
            return "datetime" if "DATETIME" in detected_fmt else "date"

        # --- Dtype signals ---
        dtype = str(profile.get("dtype") or "").lower()
        if "bool" in dtype:
            return "boolean"
        if "datetime" in dtype:
            return "datetime"

        # --- Column-name token matching ---
        # Split on underscores/hyphens/spaces so that names like
        # "user_email" or "birth_date" match word-boundary patterns
        # (e.g. \bemail\b would not fire on "user_email" otherwise).
        name = " ".join(re.split(r"[\s_\-]+", str(col).strip()))
        for pattern, sem_type in _NAME_RULES:
            if pattern.search(name):
                return sem_type

        return None

    def _apply_ml(self, profile: dict[str, Any]) -> tuple[str, float]:
        """
        Run the ML classifier and return (predicted_label, confidence).

        Args:
            profile: Augmented column profile dict.

        Returns:
            2-tuple of the predicted semantic type string and the model's
            top-class probability.
        """
        vec = self._feature_extractor.transform(profile).reshape(1, -1)
        proba = self._model.predict_proba(vec)[0]
        top_idx = int(np.argmax(proba))
        return self._model.classes_[top_idx], float(proba[top_idx])

    @staticmethod
    def _dtype_fallback(profile: dict[str, Any]) -> str:
        """
        Derive a coarse semantic type from the column's pandas dtype.

        Args:
            profile: Column profile dict (must contain a 'dtype' key).

        Returns:
            A semantic type string from ``_DTYPE_FALLBACK``, defaulting to
            ``"unknown"`` if the dtype cannot be mapped.
        """
        dtype = str(profile.get("dtype") or "").lower()
        for key, label in _DTYPE_FALLBACK.items():
            if key in dtype:
                return label
        return "unknown"

    # ------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------

    @staticmethod
    def _train():
        """
        Fit and return a scikit-learn SGDClassifier on synthetic training data.

        The pipeline wraps the raw feature matrix in a ``StandardScaler``
        so that gradient-descent training is numerically stable.

        Returns:
            A fitted ``sklearn.pipeline.Pipeline`` instance whose
            ``predict_proba()`` method accepts (n_samples, N_FEATURES)
            arrays and returns class probabilities.
        """
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X, y = _build_training_data()

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SGDClassifier(
                loss="log_loss",
                max_iter=1_000,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            )),
        ])
        pipeline.fit(X, y)
        logger.info(
            "SemanticClassifier: model trained on %d synthetic samples, "
            "%d classes.",
            len(y), len(set(y)),
        )
        return pipeline

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def _type_distribution(profiles: dict[str, dict]) -> dict[str, int]:
        """
        Return a frequency count of assigned semantic types.

        Useful for logging and smoke-testing pipeline output.

        Args:
            profiles: Classified profile dict (post ``classify()`` call).

        Returns:
            Dict mapping semantic_type → count, sorted by count descending.
        """
        counts: dict[str, int] = {}
        for p in profiles.values():
            t = p.get("semantic_type", "unknown")
            counts[t] = counts.get(t, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))