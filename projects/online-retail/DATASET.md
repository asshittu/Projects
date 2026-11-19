# Online Retail data

The UCI Online Retail dataset (ID 352) contains invoice lines from a UK-based non-store retailer that sold gifts, including to wholesale customers. The source covers December 1, 2010 through December 9, 2011. Each row describes an item or other recorded line on an invoice. Count invoices by their invoice identifier when measuring order volume.

## Source, access, and attribution

- Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. [Official catalog](https://archive.ics.uci.edu/dataset/352/online+retail) · [DOI 10.24432/C5BW33](https://doi.org/10.24432/C5BW33)
- Chen, D., Sain, S. L., & Guo, K. (2012). *Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining.* Journal of Database Marketing & Customer Strategy Management, 19(3), 197–208. [Associated paper](https://doi.org/10.1057/dbm.2012.17)
- UCI licenses this release under Creative Commons Attribution 4.0 International. Reuse and adaptation require appropriate attribution, a license link, and an indication of changes. This project normalizes types and fields, classifies invoice lines, and publishes aggregate analytics. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Accessed November 18, 2026. The official public download contains `Online Retail.xlsx` without requiring a user account. The pipeline downloads the ZIP when needed. For offline runs, save it as `data/raw/online-retail.zip` within `projects/online-retail/`. Follow the [project README](README.md) for the run commands. [Official ZIP](https://archive.ics.uci.edu/static/public/352/online%2Bretail.zip)
- The downloaded ZIP had SHA-256 `f5385cbb54bbebf7196389109c6b0621faab0c304e3702548165e71c84aede8b`. The pipeline pins its workbook member to SHA-256 `43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d`.

## Column dictionary

| Source column | Pipeline field | Meaning and use |
| --- | --- | --- |
| `InvoiceNo` | `invoice_no` | Invoice identifier shared across its lines. A leading `C` denotes a cancellation in the source. |
| `StockCode` | `stock_code` | Item code. The file includes nonnumeric codes, so the pipeline keeps it as text. |
| `Description` | `description` | Item or line description. Blank descriptions remain missing. |
| `Quantity` | `quantity` | Signed integer quantity on the line. Positive and negative values contribute to classifying the transaction. |
| `InvoiceDate` | `invoice_date` | Recorded invoice date and time. The source supplies no explicit time zone. |
| `UnitPrice` | `unit_price` | Recorded unit price in pounds sterling (GBP). The pipeline preserves zero and negative amounts for auditing. |
| `CustomerID` | `customer_id` | Source customer identifier. Present IDs support customer-level metrics and blank IDs remain missing. |
| `Country` | `country` | Customer country as recorded by the retailer. |

## Data quality and transaction treatment

- The workbook contains 541,909 rows. `CustomerID` is blank on 135,080 rows and `Description` is blank on 1,454 rows. There are 5,268 exact repetitions beyond the first occurrence of each matching row. The pipeline preserves them because the source omits a unique invoice-line identifier. Identical visible values could represent distinct lines or repeated records, and the available fields cannot resolve which explanation applies. Counts and amounts reflect the retained rows.
- Commercial sales have positive quantity and unit price and a non-cancellation invoice. Credits have a `C` prefix, negative quantity, and positive unit price. The standard sale and credit totals exclude A-prefixed administrative invoices and the atypical stock codes AMAZONFEE, B, BANK CHARGES, CRUK, D, M, and S. These entries include fees, discounts, manual lines, samples, and other adjustments. Zero-price and other nonstandard lines are classified separately for audit. A signed line amount is `quantity × unit_price`. Aggregate net sales subtract recorded credits from commercial sales without pairing credits to specific original sale lines.
- Sale and credit summaries include eligible lines regardless of customer ID. Customer-level RFM and cohort calculations require an ID. Blank IDs cannot be grouped as one customer, so coverage differs between transaction trends and customer measures.
- The December 2011 period ends on December 9. The monthly ledger retains and identifies that partial month. The month-to-month chart uses complete months through November 2011, and the customer analysis uses an exclusive cutoff of December 1, 2011.

## Limits and privacy

- Recency is days since the last qualifying sale before the cutoff. Frequency counts distinct qualifying sales invoices, and monetary value uses recorded net line amounts. These features support segmentation within the observed period. Customer lifetime value would require longer history and more information about costs and future behavior.
- Cohorts group customers by their first observed purchasing month. Purchases before the workbook starts are unknown, so first observed month may differ from acquisition month. Activity in later months means another observed qualifying purchase. Interpret gaps as periods with no observed purchase and leave churn unclassified.
- Returns and cancellations have recorded dates that may fall in a different month from the original purchase. The reported net amount reflects the dated sales and credits observed in this extract. Booked revenue, margin, or cash flow would require accounting details outside the workbook.
- Customer identifiers are pseudonymous and transaction combinations may still identify or describe a customer. Keep the workbook and row-level marts local. Share aggregate trends, segment results, and cohort summaries.
