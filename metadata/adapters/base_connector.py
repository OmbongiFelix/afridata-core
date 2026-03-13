"""
Base connector interface for all data source adapters.

Every adapter (CSV, Excel, SQL, S3) must subclass BaseConnector and
implement the `connect()` and `extract()` methods. This enforces a
consistent contract across all ingestion sources and allows the pipeline
orchestrator to swap connectors without changing downstream logic.

Usage:
    class CSVConnector(BaseConnector):
        def connect(self): ...
        def extract(self) -> pd.DataFrame: ...

"""

