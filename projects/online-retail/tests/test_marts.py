"""Check retail rollups and time windows on artificial invoice lines."""

import unittest
from datetime import date
from decimal import Decimal

import pandas as pd
from src.marts import build_marts


def fixture() -> pd.DataFrame:
    """Include duplicate lines, anonymous commerce, invalid signs, and a partial month."""
    records = [
        ("100", "P1", 2, "2011-01-01", 10.1234, 1),
        ("100", "P1", 2, "2011-01-01", 10.1234, 1),
        ("101", "P2", 1, "2011-01-03", 5, 2),
        ("102", "POST", 2, "2011-01-05", 10, None),
        ("104", "P3", 1, "2011-01-07", 3.1234, 2),
        ("C103", "P1", -1, "2011-01-20", 1.2345, 1),
        ("105", "M", 10, "2011-01-15", 50, 1),
        ("A106", "P1", 1, "2011-01-15", 50, 1),
        ("107", "P1", 1, "2011-01-02", -3, 1),
        ("108", "P1", 0, "2011-01-02", 5, 1),
        ("109", "P1", 1, "2011-01-02", 0, 1),
        ("C110", "P1", 2, "2011-01-02", 20, 1),
        ("111", "P1", -1, "2011-01-02", 10, 1),
        ("112", "P1", 3, "2011-02-01", 7, 3),
        ("C113", "P1", -1, "2011-02-01", 2, 4),
        ("C114", "P1", -1, "2011-02-02", 1.5, None),
        ("115", "P1", 1, "2011-03-02", 30, 1),
        ("115", "P2", 1, "2011-03-02", 0.0001, 1),
        ("116", "P1", 1, "2011-03-02", 7, 3),
        ("117", "P1", 1, "2011-03-01", 50, 5),
        ("118", "P1", 1, "2011-04-01", 5, 6),
        ("C119", "P1", -1, "2011-04-02", 4, 1),
        ("120", "P1", 1, "2011-04-09", 500, 1),
    ]
    data = pd.DataFrame.from_records(records, columns=[
        "invoice_no", "stock_code", "quantity", "invoice_date", "unit_price", "customer_id",
    ])
    data["invoice_date"] = pd.to_datetime(data["invoice_date"])
    data["customer_id"] = data["customer_id"].astype("Int64")
    data["description"] = "Test product"
    data["country"] = "United Kingdom"
    return data


class MartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.marts = build_marts(fixture(), cutoff=date(2011, 4, 1))

    def test_ledger_classification_is_exhaustive_and_keeps_repeated_lines(self):
        quality = self.marts["quality"].set_index("classification")
        self.assertEqual(quality["line_count"].to_dict(), {
            "credit": 4,
            "invalid_price": 1,
            "negative_non_credit": 1,
            "noncommercial_adjustment": 2,
            "positive_cancellation": 1,
            "sale": 12,
            "zero_quantity_or_price": 2,
        })
        summary = self.marts["summary"].iloc[0]
        self.assertEqual(summary["source_rows"], len(fixture()))
        self.assertEqual(summary["excluded_lines"], 7)
        self.assertEqual(summary["paid_invoices"], 10)

    def test_reconciles_exact_monthly_gbp_and_preserves_anonymous_postings(self):
        months = self.marts["monthly"].set_index("month")
        january = months.loc[date(2011, 1, 1)]
        self.assertEqual(january["gross_gbp"], Decimal("68.6170"))
        self.assertEqual(january["credits_gbp"], Decimal("1.2345"))
        self.assertEqual(january["net_gbp"], Decimal("67.3825"))
        self.assertIsInstance(january["net_gbp"], Decimal)
        self.assertEqual(january["paid_invoices"], 4)
        self.assertEqual(january["anonymous_gross_gbp"], Decimal("20.0000"))
        self.assertEqual(months.loc[date(2011, 2, 1), "anonymous_net_gbp"], Decimal("-1.5000"))
        for month in months.itertuples():
            self.assertEqual(month.gross_gbp - month.credits_gbp, month.net_gbp)
            self.assertEqual(month.anonymous_gross_gbp - month.anonymous_credits_gbp, month.anonymous_net_gbp)
        summary = self.marts["summary"].iloc[0]
        self.assertEqual(summary["net_gbp"], Decimal("672.8826"))
        self.assertEqual(months["net_gbp"].sum(), summary["net_gbp"])
        self.assertEqual(months["paid_invoices"].sum(), summary["paid_invoices"])

    def test_partial_end_month_keeps_values_and_rfm_uses_exclusive_cutoff(self):
        months = self.marts["monthly"].set_index("month")
        self.assertEqual(months["is_partial"].tolist(), [False, False, False, True])
        self.assertEqual(months.loc[date(2011, 4, 1), "observed_through"], date(2011, 4, 9))
        self.assertEqual(months.loc[date(2011, 4, 1), "net_gbp"], Decimal("501.0000"))

        summary = self.marts["summary"].iloc[0]
        self.assertEqual(summary["rfm_customers"], 4)
        self.assertEqual(summary["monetary_median_gbp"], Decimal("39.00000"))
        segments = self.marts["segments"].set_index("segment")
        self.assertEqual(segments["customer_count"].to_dict(), {
            "Recent high-value repeat": 1,
            "Other recent": 1,
            "Earlier repeat": 1,
            "Earlier single-invoice": 1,
        })
        self.assertEqual(segments.loc["Recent high-value repeat", "median_recency_days"], 30)
        self.assertEqual(segments.loc["Recent high-value repeat", "median_frequency"], 2)
        self.assertEqual(segments.loc["Recent high-value repeat", "observed_net_gbp"], Decimal("69.2592"))
        self.assertEqual(segments["observed_net_gbp"].sum(), summary["rfm_observed_net_gbp"])

    def test_mature_cohort_cells_include_zero_but_partial_month_is_censored(self):
        january = self.marts["cohorts"].query("cohort_month == @jan", local_dict={"jan": date(2011, 1, 1)})
        self.assertEqual(january["month_offset"].tolist(), [0, 1, 2])
        self.assertEqual(january["cohort_size"].tolist(), [2, 2, 2])
        self.assertEqual(january["active_customers"].tolist(), [2, 0, 1])
        self.assertEqual(january["repeat_share"].tolist(), [1.0, 0.0, 0.5])
        self.assertTrue((self.marts["cohorts"]["activity_month"] < date(2011, 4, 1)).all())
        self.assertEqual(sum(self.marts["cohorts"].query("month_offset == 0")["cohort_size"]), 4)
        for frame in self.marts.values():
            self.assertNotIn("customer_id", frame.columns)
            self.assertNotIn("invoice_no", frame.columns)

    def test_empty_data_or_non_month_cutoff_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "source invoice line"):
            build_marts(fixture().iloc[:0], cutoff=date(2011, 4, 1))
        with self.assertRaisesRegex(ValueError, "first day of a month"):
            build_marts(fixture(), cutoff=date(2011, 4, 2))

    def test_earlier_customer_cutoff_keeps_complete_months_in_ledger(self):
        cutoff = date(2011, 3, 1)
        marts = build_marts(fixture(), cutoff=cutoff)
        march = marts["monthly"].set_index("month").loc[cutoff]
        self.assertFalse(march["is_partial"])
        self.assertEqual(march["observed_through"], date(2011, 3, 31))
        self.assertEqual(march["net_gbp"], Decimal("87.0001"))
        self.assertTrue((marts["cohorts"]["activity_month"] < cutoff).all())
        self.assertEqual(marts["summary"].iloc[0]["rfm_customers"], 3)
