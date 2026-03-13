"""
Unit tests for the ranking module.

Tests domain/ranking.py in isolation.
No database, no cache, no engine calls — purely functional.

Test classes:
  TestScoreOrdering   — candidates sorted by s_hybrid descending
  TestTopNCutoff      — exactly N items returned when pool > N
  TestTieBreaking     — ties resolved by item_id ascending (deterministic)
  TestEmptyInput      — empty list returns empty RankedList without error
  TestDiversityRerank — MMR re-ranking reduces same-category clustering
  TestRankedListSchema — returned object is a valid RankedList dataclass
"""

