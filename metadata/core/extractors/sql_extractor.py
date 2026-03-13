"""
Schema metadata extractor for SQL/database sources.

Uses SQLAlchemy's Inspector to query the database information schema
and attach relational metadata to each ColumnProfile:

    - Primary key flags
    - Foreign key references (target table + column)
    - NOT NULL constraints
    - Index membership
    - Native SQL type (VARCHAR(255), DECIMAL(10,2), etc.)

This metadata is unavailable from Pandas alone and significantly
improves the quality of semantic type classification.
"""
