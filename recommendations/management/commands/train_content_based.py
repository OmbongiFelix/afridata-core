"""
Management command: python manage.py train_content_based

Fetches all Dataset records from persistence.py, concatenates title,
description, and tags into a text corpus, fits a TF-IDF vectoriser,
and saves the resulting item matrix via infrastructure/vector_store.py.

Options:
  --max-features  TF-IDF vocabulary size (default: 10000)
  --ngram-range   N-gram range, e.g. '1,2' for unigrams+bigrams (default: 1,1)
  --output        Override default matrix save path

Incremental updates are not supported — always a full rebuild.
Re-run whenever Dataset metadata is bulk-updated.
"""
