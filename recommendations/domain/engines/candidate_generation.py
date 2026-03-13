"""
Candidate generation engine for the recommendations pipeline.

Retrieves the set of dataset IDs that are eligible to be recommended
to a given user. Filters out items the user has already interacted
with so that collaborative.py and content_based.py only score
genuinely new candidates.

Responsibilities:
  1. Fetch all available dataset IDs via persistence.get_all_dataset_ids()
  2. Fetch the user's interaction history via persistence.get_user_interactions()
  3. Subtract seen items from the full pool
  4. Apply optional recency or popularity pre-filters to cap pool size
  5. Return a CandidateSet schema object
"""
