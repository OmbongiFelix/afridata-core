"""
Adapter for ingesting relational database sources.

Uses SQLAlchemy for database-agnostic connectivity. Accepts a
connection string, table name, or raw SQL query. Streams large
result sets in chunks to avoid memory overflow.

Supported backends:
    - PostgreSQL  (psycopg2)
    - MySQL       (pymysql)
    - SQLite      (built-in)
    - Any SQLAlchemy-compatible dialect

"""
