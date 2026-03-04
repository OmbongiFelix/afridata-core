# Metadata Extraction Pipeline

An automated pipeline that ingests raw content and transforms it into enriched, searchable metadata. The pipeline follows a linear modular flow:

**Ingestion → Extraction → Enhancement → Storage**


## Directory Structure

### `adapters/`

Houses external interface logic.

| File | Responsibility |
|-------------------|---------------|
| `s3_connector.py` | Handles connections and data retrieval from AWS S3 |
| `web_scraper.py`  | Extracts raw HTML or data from target websites |


### `core/`

The orchestration and business logic of the pipeline.

#### `extractors/`

Content-specific logic to pull raw data from various file types.

| File | Responsibility |
|-----------------------|---------------|
| `text_extractor.py`   | Extracts text from PDFs, DOCX, or plain text files |
| `visual_extractor.py` | Processes images or video frames for visual data |

#### `enhancement/`
Refines extracted data into structured metadata.

| File | Responsibility |
|------|---------------|
| `nlp_processor.py` | Performs Named Entity Recognition (NER) and summarization |
| `tagger.py` | Automatically assigns categories or keyword tags to content |

#### Core Files

| File | Responsibility |
|------|---------------|
| `pipeline.py` | Main controller — sequences ingestion, extraction, and enhancement steps |
| `models.py` | Defines the database schema for storing final metadata objects |
| `views.py` | Provides API endpoints to trigger pipeline runs or retrieve stored metadata |

---

## Pipeline Flow

```
Ingestion         Extraction              Enhancement         Storage
─────────         ──────────              ───────────         ───────
s3_connector  →   text_extractor    →     nlp_processor  →   models.py
web_scraper       visual_extractor        tagger             views.py
                        ↑
                   pipeline.py (orchestrates all steps)
```