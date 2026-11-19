"""Build exact-currency retail marts in a temporary DuckDB connection."""

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

SQL_PATH = Path(__file__).resolve().parents[1] / "sql" / "marts.sql"
REPORT_QUERIES = {
    "monthly": "SELECT * FROM monthly_mart ORDER BY month",
    "segments": """
        SELECT * FROM segment_mart ORDER BY CASE segment
            WHEN 'Recent high-value repeat' THEN 1
            WHEN 'Other recent' THEN 2
            WHEN 'Earlier repeat' THEN 3
            ELSE 4
        END
    """,
    "cohorts": "SELECT * FROM cohort_mart ORDER BY cohort_month, month_offset",
    "quality": "SELECT * FROM quality_mart ORDER BY classification",
    "summary": "SELECT * FROM summary_mart",
}


def build_marts(lines: pd.DataFrame, *, cutoff: date) -> dict[str, pd.DataFrame]:
    """Aggregate validated lines and exclude activity on or after the RFM cutoff."""
    if lines.empty:
        raise ValueError("At least one source invoice line is required")
    if cutoff.day != 1:
        raise ValueError("The cutoff must be the first day of a month")

    with duckdb.connect(":memory:") as connection:
        connection.register("source_lines", lines)
        connection.execute("CREATE TABLE analysis_config (cutoff DATE NOT NULL)")
        connection.execute("INSERT INTO analysis_config VALUES (?)", [cutoff])
        connection.execute(SQL_PATH.read_text(encoding="utf-8"))
        marts = {}
        for name, query in REPORT_QUERIES.items():
            cursor = connection.execute(query)
            columns = [column[0] for column in cursor.description]
            # DuckDB's direct DataFrame conversion coerces Decimal values to floats.
            marts[name] = pd.DataFrame.from_records(cursor.fetchall(), columns=columns)
    return marts
