"""
Ranking module — post-fusion ordering for the recommendations pipeline.

Receives a list of ScoredCandidate objects from hybrid.py and returns
a RankedList sorted by S_hybrid descending, trimmed to Top-N.

Optional diversity re-ranking:
  When EngineConfig.diversity_weight > 0, applies a Maximal Marginal
  Relevance (MMR) variant that penalises consecutive items from the
  same dataset category to improve result variety.

This module owns ALL post-fusion ordering logic.
hybrid.py must not sort or filter — it calls rank() from here.
"""
