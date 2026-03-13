"""
Weighted Hybrid Fusion Engine — central orchestrator of the pipeline.

Formula:  S_hybrid = α · S_CF  +  (1 − α) · S_CBF

Orchestration sequence:
  1. Accept a CandidateSet and EngineConfig
  2. Call CollaborativeEngine.score() → S_CF dict
  3. Call ContentBasedEngine.score()  → S_CBF dict
  4. Fuse both dicts using the alpha formula
  5. Normalise fused scores to [0, 1]
  6. Pass ScoredCandidate list to domain/ranking.py
  7. Return the RankedList from ranking.rank()

This module does NOT sort, filter, or apply Top-N cutoff.
All post-fusion ordering belongs in domain/ranking.py.
"""



from domain.ranking import *