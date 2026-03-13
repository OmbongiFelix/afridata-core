"""
Management command: python manage.py train_collaborative

Fetches the full UserInteraction history from persistence.py,
builds a user-item interaction matrix, fits a Matrix Factorisation
model (ALS or SVD), and saves the trained weights via
infrastructure/model_store.py.

Options:
  --factors   Number of latent factors (default: 50)
  --epochs    Training iterations (default: 20)
  --output    Override default model save path
  --evaluate  Run evaluation metrics against a held-out test split

Run nightly via Celery beat or after any bulk interaction import.
"""
