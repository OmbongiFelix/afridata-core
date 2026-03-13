"""
Test suite for the Metadata Extraction Pipeline.

Organised by pipeline stage — each stage has its own TestCase class:

    TestCSVConnector         - adapter ingestion and error handling
    TestSQLConnector         - database extraction with SQLite fixture
    TestColumnProfiler       - profile accuracy on known DataFrames
    TestSemanticClassifier   - semantic type predictions
    TestLLMGenerator         - LLM call mocked with unittest.mock
    TestSchemaBuilder        - valid JSON Schema output
    TestPipelineIntegration  - end-to-end run with CSV fixture
    TestPipelineAPI          - DRF test client for all endpoints

Run with: python manage.py test metadata
"""

