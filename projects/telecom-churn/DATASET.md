# Iranian Churn data

This project uses the UCI Iranian Churn dataset (ID 563), a sample of 3,150 customer records from one Iranian telecommunications company. UCI describes the predictors as aggregates from the first nine months of a 12-month observation period. `Churn` is the customer’s recorded state at the end of month 12. The intervening three months form a planning window for retention outreach.

## Source, access, and attribution

- *Iranian Churn* [Dataset]. (2020). UCI Machine Learning Repository. The record omits a named creator and the DOI registry lists the creator as unknown. [UCI catalog](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset) · [DOI 10.24432/C5JW3Z](https://doi.org/10.24432/C5JW3Z) · [DataCite record](https://api.datacite.org/dois/10.24432/C5JW3Z).
- UCI licenses the dataset under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Reuse and adaptation require appropriate credit, a license link, and an indication of changes. This project validates and selects model features, keeps matching predictor records in the same partition, and publishes aggregate results.
- Accessed November 18, 2026. The [official archive](https://archive.ics.uci.edu/static/public/563/iranian%2Bchurn%2Bdataset.zip) is available without an account. It contains `Customer Churn.csv`. Follow the [project README](README.md) to run the downloader, or save the archive to `data/raw/iranian-churn.zip` within `projects/telecom-churn/` for offline runs. Keep raw records in the ignored raw-data directory.
- The downloaded ZIP had SHA-256 `696c3a1812267980751f30d19193a1b430ac4ad76172bb089e664c96813ead66`. The pipeline verifies the CSV against SHA-256 `90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1`.

## Column dictionary

The dictionary follows the downloadable file. Some source headers contain two consecutive spaces. UCI’s prose mentions an anonymous customer ID. The released CSV has 13 feature columns and `Churn` and omits an identifier. The separately listed `Age` column and charge level 10 below reflect the released data.

| Source column | Meaning and coding | Use in the model |
| --- | --- | --- |
| `Call  Failure` | Count of failed calls | Included as a service-quality measure |
| `Complains` | Complaint indicator, `0` for no complaint and `1` for a complaint | Included as a binary service measure |
| `Subscription  Length` | Total months of tenure | Included as a numeric predictor |
| `Charge  Amount` | Ordinal charge level, observed `0` through `10` | Included as an ordered code. Monetary units are unspecified. |
| `Seconds of Use` | Total seconds of calls | Included as a numeric usage measure |
| `Frequency of use` | Total calls | Included as a numeric usage measure |
| `Frequency of SMS` | Total text messages | Included as a numeric usage measure |
| `Distinct Called Numbers` | Count of distinct called numbers | Included as a numeric usage measure |
| `Age Group` | Ordinal group, `1` for youngest through `5` for oldest | Excluded to limit direct use of demographics |
| `Tariff Plan` | `1` for pay-as-you-go and `2` for contractual | Included as a plan category |
| `Status` | Account status, `1` for active and `2` for non-active | Excluded because inactivity is close to the outcome and there is limited documentation on when or how status was set |
| `Age` | The observed values `15`, `25`, `30`, `45`, and `55` map one-to-one to `Age Group` levels `1` through `5` | Excluded alongside `Age Group` |
| `Customer Value` | Calculated customer value with an undocumented formula and refresh schedule | Excluded because its inputs and scoring-time availability cannot be confirmed |
| `Churn` | Month-12 outcome, `0` for non-churn and `1` for churn | Target |

The nine retained fields describe service history, account characteristics, or usage. The model also uses `average_call_seconds`, calculated by dividing `Seconds of Use` by `Frequency of use`, and `no_voice_calls`, set to `1` where `Frequency of use` is zero. For those rows the call average is undefined and the numeric preprocessing handles it as missing.

Before live use, recreate each feature from a verified snapshot as of the outreach decision. The published file provides the observation window and omits field-level timestamps. In the released data, 370 of 782 non-active customers churned (47.3%), compared with 125 of 2,368 active customers (5.3%). This association warrants scrutiny of when and how `Status` was set.

## Data quality and evaluation design

- The pinned CSV contains 3,150 records with zero blank cells. There are 495 churn labels (15.71%) and 2,655 non-churn labels. `Charge  Amount` includes seven rows coded `10`, one level above the catalog’s stated maximum. The project preserves those values and treats the column as ordinal. In 154 records, call count, duration, and distinct called numbers are all zero.
- Across the original fields and target, 300 rows repeat an earlier row exactly. The release omits an identifier that would let us distinguish duplicate records from different customers with the same values. The project retains all rows. Across the nine model fields, there are 2,785 distinct feature vectors, including 168 groups shared by 533 rows. Eighteen groups have both outcomes. Keeping matching predictors together protects evaluation against seeing the same profile during training and testing while preserving differing labels.
- UCI supplies a progression from observation to outcome within each record. The file omits calendar dates and verified ordering across accounts, so the evaluation uses a seeded stratified group split. It groups rows by the nine retained predictor values, assigns three of five folds to training, one to validation, and one to test, and keeps each group in a single partition. Preprocessing is fit on training data. Model and outreach threshold selection use validation data, and the test partition provides held-out evaluation.

## Limits and responsible use

- The file contains customer-level complaints, account information, and behavioral counts. Publish aggregate reports, avoid linking records to external identities, and verify consent, access controls, and retention policies before scoring customer data.
- Removing the two age fields limits their direct use. Plan and usage variables may still act as demographic or socioeconomic proxies. Before operational use, check precision, recall, and outreach selection rates across relevant groups and assess whether the retained fields are necessary for the decision.
- The sample comes from one telecom provider. Check model performance and calibration with later cohorts and different plans before applying it to another retention program. SaaS applications would require their own churn definition, observation window, and validation data.
- Use model scores to rank customers by estimated churn risk. Estimate how many customers would stay because of outreach through a separate prospective experiment with an appropriate comparison group.
