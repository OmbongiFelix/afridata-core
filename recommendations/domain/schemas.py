"""
Shared data contracts for the recommendations domain layer.

All inter-module communication in domain/ uses these types.
Import from here — never import from individual engine files
to avoid circular dependencies.

Types:
  CandidateSet      — output of candidate_gen: user_id + list of item_ids
  ScoredCandidate   — one item with S_CF, S_CBF, and S_hybrid scores
  RankedList        — ordered list of ScoredCandidate up to Top-N
  EngineConfig      — runtime settings: alpha, top_n, diversity_weight
"""
