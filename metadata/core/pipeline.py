"""
Main pipeline orchestrator for the Metadata Extraction system.

This module is intentionally thin. It imports one class or function
from each stage module and chains them in order:

    1. Adapter       → ingest raw data into a DataFrame
    2. Profiler      → compute column-level statistics
    3. Extractor     → extract text / SQL schema metadata
    4. Classifier    → assign semantic types via ML model
    5. LLM Generator → enrich metadata using an LLM prompt
    6. SchemaBuilder → serialise final output to JSON Schema

Do NOT add business logic here. If a stage needs complex logic,
it belongs in that stage's own module.

Usage:
    result = MetadataPipeline(source='csv', path='data.csv').run()

"""
