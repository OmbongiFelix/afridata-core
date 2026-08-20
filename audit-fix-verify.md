GOAL
metadata and recommendations already have most of the pieces: metadata
is API-only (adapters/core/api), recommendations has domain/infrastructure/
api with content_based, collaborative, AND hybrid engines already built.
This is NOT a from-scratch "convert to API" job — it's: audit what's
broken or inconsistent, fix it, close any gaps so the following flow
works end-to-end over REST, and PROVE it works with real HTTP calls
against a running server (not just green unit tests):

  1. POST a structured dataset (csv/excel/sql) → ingested
  2. Metadata inferred for that dataset (semantic classifier pipeline)
  3. GET recommendations for that dataset — content-based, collaborative,
     and hybrid — using the inferred metadata

SCOPE
Apps in play: metadata, recommendations, config only.
Preserve existing architecture — domain/infrastructure/api in
recommendations, adapters/core/api in metadata. Don't collapse or
restructure these layers.

STEP 1 — AUDIT (Plan Mode, no code changes yet)
- Reconcile recommendations/views.py + urls.py (root level) against
  recommendations/api/*. Report what each actually serves and whether
  the root-level ones are legacy/dead code or intentionally separate
  (e.g. Celery/internal triggers vs public REST).
- Confirm config/settings.py DB config vs the committed db.sqlite3 —
  report any mismatch.
- Read and report what important_notice.py is for.
- Audit domain/engines/{content_based,collaborative,hybrid}.py against
  recommendations/tests/* — report any engine with failing or missing
  tests.
- Compare metadata/api and recommendations/api's actual urls.py against
  postman/specs and postman/collections — report drift between what's
  documented and what's actually wired up.
- Report the current auth story (none found in the tree — confirm and
  flag as a decision point, don't silently add one).
Stop after this and show me the plan before writing code.

STEP 2 — FIX (after I approve the plan)
Fix only what Step 1 identified as broken. Don't refactor working code.
Flag anything ambiguous instead of guessing.

STEP 3 — CLOSE THE GAPS
Wire up whatever's missing so the ingest → metadata → recommend flow is
fully reachable over REST:
- POST /datasets/ (or wherever metadata/api already routes this) —
  ingest a structured dataset
- GET /datasets/{id}/metadata/ — inferred metadata
- GET /recommendations/{dataset_id}/?strategy=content|collaborative|hybrid
  (or match whatever routing convention recommendations/api already uses)
Resolve the root-level views.py/urls.py duplication from Step 1 rather
than leaving both live.

STEP 4 — VERIFY IT ACTUALLY WORKS
This is not "tests pass." Do this:
- Run migrations against the real configured DB (uv run manage.py migrate)
- Start the dev server (uv run manage.py runserver)
- Read postman/collections and postman/environments to see what
  endpoints, methods, and payloads are expected
- Hit each of those endpoints with curl against the live server —
  report actual status codes and response bodies for each call, not a
  summary, I want to see what came back
- Then run the Django test suite (uv run pytest, or manage.py test —
  check pyproject.toml for the configured runner) for metadata/tests
  and recommendations/tests, report pass/fail per file
- Any endpoint hit manually in this step that isn't represented in the
  existing postman collection, add a request for it there so the
  collection stays a real end-to-end reference going forward

CONSTRAINTS
- Don't invent functionality beyond closing the gaps above
- Don't touch mpesa/community/home/accounts — they don't exist in this
  tree, ignore any instinct to create them
- Preserve model field names and existing migrations where possible;
  call out any unavoidable schema change before making it
- If auth is genuinely required to ship this, stop and ask rather than
  picking a scheme