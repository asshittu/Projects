"""Audit the observed paths and reconcile descriptive campaign-credit rules."""

from math import isclose
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
CREDIT_FIELDS = ("first_touch", "last_touch", "linear_touch")


def query_records(connection, sql: str, parameters=None) -> list[dict]:
    """Return named rows from an aggregate query."""
    result = connection.execute(sql, parameters)
    columns = [column[0] for column in result.description]
    return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]


def source_summary(connection) -> dict:
    """Check the subset of fields required to identify pre-conversion impressions."""
    summary, = query_records(connection, """
        SELECT count(*) AS impressions, count(DISTINCT user_id) AS users,
               count(DISTINCT campaign) AS campaigns,
               min(touch_time) AS first_impression_second, max(touch_time) AS last_impression_second,
               count(*) FILTER (WHERE has_conversion = 1) AS linked_impressions,
               min(conversion_time) FILTER (WHERE has_conversion = 1) AS first_conversion_second,
               max(conversion_time) FILTER (WHERE has_conversion = 1) AS last_conversion_second,
               count(*) FILTER (WHERE has_conversion = 1 AND touch_time >= conversion_time)
                   AS at_or_after_conversion_impressions,
               count(*) FILTER (
                   WHERE touch_time IS NULL OR user_id IS NULL OR campaign IS NULL
                      OR has_conversion IS NULL OR conversion_time IS NULL
                      OR conversion_id IS NULL OR attributed IS NULL
                      OR least(touch_time, user_id, campaign) < 0
                      OR has_conversion NOT IN (0, 1) OR attributed NOT IN (0, 1)
                      OR (has_conversion = 1 AND least(conversion_id, conversion_time) < 0)
                      OR (has_conversion = 0 AND (conversion_id <> -1 OR conversion_time <> -1))
               ) AS invalid_rows
        FROM impressions
    """)
    if not summary["impressions"] or summary["invalid_rows"]:
        raise ValueError("Criteo impression fields are empty or contain invalid values.")
    return summary


def analyze(connection, lookback_days: int) -> tuple[dict, list[dict], list[dict]]:
    """Build conversion-linked paths and allocate one credit per path by each rule."""
    if not 1 <= lookback_days <= 30:
        raise ValueError("Lookback must be an integer from 1 to 30 days.")
    source = source_summary(connection)
    connection.execute("CREATE TEMP TABLE settings AS SELECT ? AS lookback_seconds", [lookback_days * 86_400])
    connection.execute("""
        CREATE TEMP TABLE observation_bounds AS
        SELECT ? AS first_touch, ? AS last_touch
    """, [source["first_impression_second"], source["last_impression_second"]])
    connection.execute((PROJECT / "sql" / "attribution.sql").read_text(encoding="utf-8"))

    windows, = query_records(connection, """
        SELECT count(*) AS reported_conversion_ids,
               count(*) FILTER (WHERE status = 'ambiguous') AS ambiguous_conversion_ids,
               count(*) FILTER (WHERE status = 'left_boundary') AS insufficient_prehistory_ids,
               count(*) FILTER (WHERE status = 'right_boundary') AS beyond_observation_ids,
               count(*) FILTER (WHERE status = 'within_bounds') AS within_observation_ids
        FROM conversion_windows
    """)
    linked, campaigns = summarize_scope(connection, "source_linked")
    linked["paths"].update(windows)
    linked["paths"]["no_touch_in_lookback_ids"] = (
        windows["within_observation_ids"] - linked["paths"]["qualified_conversions"]
    )

    sensitivity, same_user_campaigns = summarize_scope(connection, "same_user")
    anchors, = query_records(connection, """
        SELECT count(*) AS source_linked_anchors,
               count(*) FILTER (WHERE ids_at_moment > 1) AS simultaneous_conversion_ids
        FROM same_user_anchors
    """)
    sensitivity["paths"].update(anchors)
    sensitivity["paths"]["anchors_without_eligible_touches"] = (
        anchors["source_linked_anchors"] - sensitivity["paths"]["qualified_conversions"]
    )

    return {
        "study_scope": "Observed Criteo display campaign impressions",
        "lookback_days": lookback_days,
        "source": source,
        **linked,
        "same_user_sensitivity": sensitivity,
    }, campaigns, same_user_campaigns


def summarize_scope(connection, scope: str) -> tuple[dict, list[dict]]:
    """Summarize the paths and verify that each rule's credits reconcile."""
    paths, = query_records(connection, """
        WITH conversion_paths AS (
            SELECT conversion_id, count(*) AS touches, count(DISTINCT campaign) AS campaigns,
                   max(tie_count) FILTER (WHERE touch_time = first_time) AS first_ties,
                   max(tie_count) FILTER (WHERE touch_time = last_time) AS last_ties
            FROM qualifying_touches WHERE scope = ? GROUP BY conversion_id
        )
        SELECT count(*) AS qualified_conversions,
               coalesce(sum(touches), 0) AS qualified_impressions,
               count(*) FILTER (WHERE touches > 1) AS multi_touch_conversions,
               count(*) FILTER (WHERE campaigns > 1) AS multi_campaign_conversions,
               count(*) FILTER (WHERE first_ties > 1 OR last_ties > 1) AS timestamp_tie_conversions
        FROM conversion_paths
    """, [scope])
    campaigns = query_records(connection, """
        SELECT campaign, touches, paths, first_touch, last_touch, linear_touch
        FROM campaign_credits
        WHERE scope = ?
        ORDER BY linear_touch DESC, campaign
    """, [scope])
    totals = {field: sum(row[field] for row in campaigns) for field in CREDIT_FIELDS}
    if any(not isclose(value, paths["qualified_conversions"], rel_tol=1e-10, abs_tol=1e-10)
           for value in totals.values()):
        raise RuntimeError("Credit totals must equal the number of eligible conversions.")
    for row in campaigns:
        for field in CREDIT_FIELDS:
            row[field] = round(row[field], 8)
    campaigns.sort(key=lambda row: (-row["linear_touch"], row["campaign"]))

    return {
        "paths": paths,
        "campaigns_receiving_credit": len(campaigns),
        "credit_totals": {field: round(value, 8) for field, value in totals.items()},
    }, campaigns
