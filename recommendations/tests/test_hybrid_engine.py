"""
Unit tests for the hybrid fusion engine.

Tests domain/engines/hybrid.py in isolation.
Both CollaborativeEngine and ContentBasedEngine are mocked.
This file tests fusion logic only — not engine correctness.

Test classes:
  TestAlphaWeighting    — alpha=1.0 gives CF-only; alpha=0.0 gives CBF-only
  TestFusionArithmetic  — alpha=0.5 gives arithmetic mean of both scores
  TestMissingItemScore  — item absent from one engine treated as 0.0
  TestScoreNormalisation — fused scores clamped to [0, 1]
  TestRankedListOutput  — return type is a valid RankedList from ranking.py
"""
