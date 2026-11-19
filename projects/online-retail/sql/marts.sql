-- Named local views form a line -> invoice -> calendar / customer aggregation DAG.
-- Keep the exact source line count until reporting its mutually exclusive classes.
CREATE VIEW ledger AS
WITH normalized AS (
    SELECT
        invoice_no,
        upper(invoice_no) LIKE 'C%' AS cancellation,
        upper(invoice_no) LIKE 'A%' AS administrative_invoice,
        upper(stock_code) AS stock_code,
        quantity,
        CAST(invoice_date AS DATE) AS invoice_day,
        CAST(date_trunc('month', invoice_date) AS DATE) AS invoice_month,
        CAST(unit_price AS DECIMAL(18, 4)) AS price_gbp,
        customer_id
    FROM source_lines
)
SELECT
    *,
    CAST(quantity * price_gbp AS DECIMAL(18, 4)) AS booked_gbp,
    CASE
        WHEN administrative_invoice OR stock_code IN (
            'AMAZONFEE', 'B', 'BANK CHARGES', 'CRUK', 'D', 'M', 'S'
        ) THEN 'noncommercial_adjustment'
        WHEN price_gbp < 0 THEN 'invalid_price'
        WHEN quantity = 0 OR price_gbp = 0 THEN 'zero_quantity_or_price'
        WHEN cancellation AND quantity > 0 THEN 'positive_cancellation'
        WHEN NOT cancellation AND quantity < 0 THEN 'negative_non_credit'
        WHEN cancellation THEN 'credit'
        ELSE 'sale'
    END AS classification
FROM normalized;

CREATE VIEW commercial_ledger AS
SELECT * FROM ledger
WHERE classification IN ('sale', 'credit');

-- One paid invoice is recognized on the date of its first eligible sale line.
-- Credit-only invoices remain in the view for reconciliation, without paid status.
CREATE VIEW invoice_mart AS
SELECT
    invoice_no,
    min(invoice_day) AS first_posted_date,
    min(invoice_day) FILTER (WHERE classification = 'sale') AS first_sale_date,
    COALESCE(sum(booked_gbp) FILTER (WHERE classification = 'sale'), 0) AS gross_gbp,
    COALESCE(-sum(booked_gbp) FILTER (WHERE classification = 'credit'), 0) AS credits_gbp,
    sum(booked_gbp) AS net_gbp
FROM commercial_ledger
GROUP BY invoice_no;

CREATE VIEW source_period AS
SELECT min(invoice_day) AS source_first_date, max(invoice_day) AS source_last_date
FROM ledger;

CREATE VIEW calendar_months AS
WITH coverage AS (
    SELECT
        CAST(period.month_start AS DATE) AS month,
        greatest(CAST(period.month_start AS DATE), source_first_date) AS observed_from,
        least(last_day(period.month_start), source_last_date) AS observed_through
    FROM source_period
    CROSS JOIN LATERAL generate_series(
        date_trunc('month', source_first_date),
        date_trunc('month', source_last_date),
        INTERVAL 1 MONTH
    ) AS period(month_start)
)
SELECT
    month,
    observed_from,
    observed_through,
    observed_from > month OR observed_through < last_day(month) AS is_partial
FROM coverage;

CREATE VIEW monthly_mart AS
WITH posting_totals AS (
    SELECT
        invoice_month AS month,
        sum(booked_gbp) FILTER (WHERE classification = 'sale') AS gross_gbp,
        -sum(booked_gbp) FILTER (WHERE classification = 'credit') AS credits_gbp,
        sum(booked_gbp) AS net_gbp,
        count(*) FILTER (WHERE classification = 'sale') AS sale_lines,
        count(*) FILTER (WHERE classification = 'credit') AS credit_lines,
        sum(booked_gbp) FILTER (
            WHERE customer_id IS NULL AND classification = 'sale'
        ) AS anonymous_gross_gbp,
        -sum(booked_gbp) FILTER (
            WHERE customer_id IS NULL AND classification = 'credit'
        ) AS anonymous_credits_gbp,
        sum(booked_gbp) FILTER (WHERE customer_id IS NULL) AS anonymous_net_gbp
    FROM commercial_ledger
    GROUP BY invoice_month
),
paid_invoice_counts AS (
    SELECT CAST(date_trunc('month', first_sale_date) AS DATE) AS month, count(*) AS paid_invoices
    FROM invoice_mart
    WHERE first_sale_date IS NOT NULL
    GROUP BY 1
)
SELECT
    calendar.month,
    calendar.observed_from,
    calendar.observed_through,
    calendar.is_partial,
    COALESCE(postings.gross_gbp, 0) AS gross_gbp,
    COALESCE(postings.credits_gbp, 0) AS credits_gbp,
    COALESCE(postings.net_gbp, 0) AS net_gbp,
    COALESCE(invoices.paid_invoices, 0) AS paid_invoices,
    COALESCE(postings.sale_lines, 0) AS sale_lines,
    COALESCE(postings.credit_lines, 0) AS credit_lines,
    COALESCE(postings.anonymous_gross_gbp, 0) AS anonymous_gross_gbp,
    COALESCE(postings.anonymous_credits_gbp, 0) AS anonymous_credits_gbp,
    COALESCE(postings.anonymous_net_gbp, 0) AS anonymous_net_gbp
FROM calendar_months AS calendar
LEFT JOIN posting_totals AS postings ON postings.month = calendar.month
LEFT JOIN paid_invoice_counts AS invoices ON invoices.month = calendar.month;

CREATE VIEW rfm_customer AS
SELECT
    customer_id,
    date_diff(
        'day',
        max(invoice_day) FILTER (WHERE classification = 'sale'),
        cutoff
    ) AS recency_days,
    count(DISTINCT invoice_no) FILTER (WHERE classification = 'sale') AS frequency,
    sum(booked_gbp) AS observed_net_gbp
FROM commercial_ledger
CROSS JOIN analysis_config
WHERE customer_id IS NOT NULL AND invoice_day < cutoff
GROUP BY customer_id, cutoff
HAVING count(*) FILTER (WHERE classification = 'sale') > 0;

CREATE VIEW rfm_rules AS
SELECT
    30 AS recency_max_days,
    2 AS minimum_repeat_invoices,
    -- The median of two 4-place currency values can require a fifth decimal.
    median(CAST(observed_net_gbp AS DECIMAL(28, 5))) AS monetary_median_gbp,
    count(*) AS rfm_customers,
    COALESCE(sum(observed_net_gbp), 0) AS rfm_observed_net_gbp
FROM rfm_customer;

CREATE VIEW segment_mart AS
WITH assigned AS (
    SELECT
        *,
        CASE
            WHEN recency_days <= recency_max_days
                AND frequency >= minimum_repeat_invoices
                AND observed_net_gbp >= monetary_median_gbp THEN 'Recent high-value repeat'
            WHEN recency_days <= recency_max_days THEN 'Other recent'
            WHEN frequency >= minimum_repeat_invoices THEN 'Earlier repeat'
            ELSE 'Earlier single-invoice'
        END AS segment
    FROM rfm_customer
    CROSS JOIN rfm_rules
)
SELECT
    segment,
    count(*) AS customer_count,
    count(*)::DOUBLE / max(rfm_customers) AS customer_share,
    median(recency_days) AS median_recency_days,
    median(frequency) AS median_frequency,
    sum(observed_net_gbp) AS observed_net_gbp,
    median(CAST(observed_net_gbp AS DECIMAL(28, 5))) AS median_observed_net_gbp
FROM assigned
GROUP BY segment;

CREATE VIEW cohort_mart AS
WITH customer_invoice AS (
    SELECT customer_id, invoice_no, min(invoice_day) AS paid_date
    FROM commercial_ledger
    CROSS JOIN analysis_config
    WHERE classification = 'sale' AND customer_id IS NOT NULL AND invoice_day < cutoff
    GROUP BY customer_id, invoice_no
),
activity AS (
    SELECT DISTINCT customer_id, CAST(date_trunc('month', paid_date) AS DATE) AS activity_month
    FROM customer_invoice
),
cohort_members AS (
    SELECT customer_id, min(activity_month) AS cohort_month
    FROM activity
    GROUP BY customer_id
),
cohort_sizes AS (
    SELECT cohort_month, count(*) AS cohort_size
    FROM cohort_members
    JOIN calendar_months AS calendar ON calendar.month = cohort_month AND NOT calendar.is_partial
    GROUP BY cohort_month
),
monthly_activity AS (
    SELECT cohort_month, activity_month, count(*) AS active_customers
    FROM activity
    JOIN cohort_members USING (customer_id)
    GROUP BY cohort_month, activity_month
)
SELECT
    sizes.cohort_month,
    date_diff('month', sizes.cohort_month, calendar.month) AS month_offset,
    calendar.month AS activity_month,
    sizes.cohort_size,
    COALESCE(activity.active_customers, 0) AS active_customers,
    COALESCE(activity.active_customers, 0)::DOUBLE / sizes.cohort_size AS repeat_share
FROM cohort_sizes AS sizes
CROSS JOIN analysis_config
JOIN calendar_months AS calendar
    ON calendar.month >= sizes.cohort_month AND calendar.month < cutoff AND NOT calendar.is_partial
LEFT JOIN monthly_activity AS activity
    ON activity.cohort_month = sizes.cohort_month AND activity.activity_month = calendar.month;

CREATE VIEW quality_mart AS
SELECT
    classification,
    count(*) AS line_count,
    count(*) FILTER (WHERE customer_id IS NULL) AS anonymous_line_count,
    sum(booked_gbp) AS signed_gbp
FROM ledger
GROUP BY classification;

CREATE VIEW summary_mart AS
WITH all_lines AS (
    SELECT
        count(*) AS source_rows,
        count(*) FILTER (WHERE classification NOT IN ('sale', 'credit')) AS excluded_lines,
        count(*) FILTER (WHERE customer_id IS NULL) AS anonymous_lines,
        count(*) FILTER (WHERE classification = 'sale' AND customer_id IS NULL) AS anonymous_sale_lines,
        count(*) FILTER (WHERE classification = 'credit' AND customer_id IS NULL) AS anonymous_credit_lines
    FROM ledger
),
financials AS (
    SELECT
        COALESCE(sum(sale_lines), 0) AS sale_lines,
        COALESCE(sum(credit_lines), 0) AS credit_lines,
        COALESCE(sum(paid_invoices), 0) AS paid_invoices,
        COALESCE(sum(gross_gbp), 0) AS gross_gbp,
        COALESCE(sum(credits_gbp), 0) AS credits_gbp,
        COALESCE(sum(net_gbp), 0) AS net_gbp,
        COALESCE(sum(anonymous_gross_gbp), 0) AS anonymous_gross_gbp,
        COALESCE(sum(anonymous_credits_gbp), 0) AS anonymous_credits_gbp,
        COALESCE(sum(anonymous_net_gbp), 0) AS anonymous_net_gbp
    FROM monthly_mart
)
SELECT *
FROM analysis_config
CROSS JOIN source_period
CROSS JOIN all_lines
CROSS JOIN financials
CROSS JOIN rfm_rules;
