"""
Unit tests for the candidate generation engine.

Tests domain/engines/candidate_gen.py in isolation.
All database calls are mocked — no real ORM queries.

Test classes:
  TestCandidateFiltering   — seen items excluded correctly
  TestColdStartCandidate   — no interactions → full pool returned
  TestEmptyPool            — user has seen all items → empty CandidateSet
  TestPopularityPrefilter  — pool capped at candidate_pool_size
  TestCandidateSchema      — returned object is a valid CandidateSet
"""
