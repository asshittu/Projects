# Bank Marketing data

This project uses the 45,211 rows of `bank-full.csv` from the UCI Bank Marketing collection (ID 222). It records telephone marketing for term deposits at a Portuguese bank. The publisher lists the rows in date order from May 2008 through November 2010 in the `bank-names.txt` file packaged with the CSV. The outcome `y` records whether the contacted client subscribed to a term deposit. This historical sample represents the bank’s existing contact process, which could include repeated contacts to the same client.

## Source, access, and attribution

- Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. [Dataset record](https://archive.ics.uci.edu/dataset/222/bank+marketing) · [DOI 10.24432/C5K306](https://doi.org/10.24432/C5K306).
- The bundled `bank-names.txt` also requests attribution to Moro, S., Laureano, R., & Cortez, P. (2011). *Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology*. European Simulation and Modelling Conference, pp. 117–121. [Paper](https://hdl.handle.net/1822/14838).
- UCI lists the dataset under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). It allows reuse and adaptation with appropriate credit, a license link, and an indication of changes. This project selects and transforms features, splits the records in source order, and publishes derived aggregate results.
- Accessed November 18, 2026. The pipeline fetches the [official download](https://archive.ics.uci.edu/static/public/222/bank%2Bmarketing.zip) when its cache is empty, with no account required. For offline runs, save this outer ZIP as `data/raw/bank-marketing.zip` within `projects/bank-marketing/`. The pipeline reads `bank.zip` inside that archive, then `bank-full.csv`. Keep the archive in the ignored raw-data directory. `bank.csv` is a random subset, and the `bank-additional` files use a different feature schema.
- The downloaded outer ZIP had SHA-256 `e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4`. The pipeline verifies the semicolon-delimited `bank-full.csv` against SHA-256 `d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d`. Preserve the original file and row order for reproducing results.

## Column dictionary

The dictionary reflects `bank-full.csv` and its companion `bank-names.txt`. The UCI catalog also covers other variants of the dataset.

| Column | Type and meaning | Use in the pre-call model |
| --- | --- | --- |
| `age` | Integer, age in years | Excluded to limit use of demographic information |
| `job` | Occupation category. Values are `admin.`, `blue-collar`, `entrepreneur`, `housemaid`, `management`, `retired`, `self-employed`, `services`, `student`, `technician`, `unemployed`, `unknown`. | Included as a category |
| `marital` | `divorced`, `married`, or `single`. `divorced` also covers widowed clients. | Excluded to limit use of demographic information |
| `education` | `primary`, `secondary`, `tertiary`, or `unknown` | Included as a category |
| `default` | `yes` or `no`, whether the client has credit in default | Included as a category |
| `balance` | Integer, average yearly account balance in euros | Included as a signed numeric value |
| `housing` | `yes` or `no`, whether the client has a housing loan | Included as a category |
| `loan` | `yes` or `no`, whether the client has a personal loan | Included as a category |
| `contact` | Channel of the recorded campaign contact. Values are `cellular`, `telephone`, `unknown`. | Excluded because it describes the recorded current-campaign contact |
| `day` | Integer, recorded last contact’s day of the month (1–31) | Excluded because it describes the recorded current-campaign contact |
| `month` | Month of the recorded last contact, `jan` through `dec` | Excluded because it describes the recorded current-campaign contact |
| `duration` | Integer, recorded last contact’s duration in seconds | Excluded. Duration becomes known after the call and would introduce information unavailable at scoring time. |
| `campaign` | Integer, number of contacts during this campaign, including the recorded last contact | Excluded because it incorporates activity from the campaign being predicted |
| `pdays` | Integer, days since the last contact in a previous campaign. `-1` represents no previous contact. | Included via prior-contact indicator and elapsed days when defined |
| `previous` | Integer, number of contacts before the current campaign | Included as prior history |
| `poutcome` | Previous campaign outcome. Values are `failure`, `other`, `success`, `unknown`. | Included as prior history |
| `y` | `yes` or `no`, whether the client subscribed to a term deposit | Target, mapped to `1` or `0` respectively |

The model assumes the nine included raw inputs come from records available at scoring time, before the current campaign call. Production use would require checking feature snapshot timestamps and recreating prior-contact history as it stood at scoring. The pipeline preserves the literal `unknown` category as recorded. For `pdays=-1`, `contacted_before` is `0` and `prior_days` is missing. For a recorded elapsed time, `contacted_before` is `1` and `prior_days` retains the time in days. The numeric imputer learns from the training partition.

## Data quality and split

The checked CSV has 16 source predictors and one target. There are zero blank cells and zero exact duplicate rows across all 17 columns. `unknown` appears in `job` (288 rows), `education` (1,857), `contact` (13,020), and `poutcome` (36,959). All 36,954 rows with `pdays=-1` also have `previous=0` and `poutcome=unknown`, and a further five rows record prior contact with an unknown outcome. Account balances range from −€8,019 to €102,127. Preserve negative values because the column measures a signed balance. The loader validates the schema, coding, and prior-contact consistency and retains every source row.

The project uses contiguous partitions in publisher-provided date order. The first 60% of rows train the models, the next 20% support model and threshold selection, and the final 20% provide a held-out evaluation. Floor rounding applies at both cut points. The model pipelines fit preprocessing to the training rows only.

| Partition | Rows | Subscriptions | Subscription rate |
| --- | ---: | ---: | ---: |
| Training | 27,126 | 1,367 | 5.04% |
| Validation | 9,042 | 1,066 | 11.79% |
| Test | 9,043 | 2,856 | 31.58% |
| Full dataset | 45,211 | 5,289 | 11.70% |

The subscription rate changes substantially across partitions. Compare the models to the dummy baseline on the same held-out rows, and interpret precision, calibration, and threshold portability in light of the changing rates. The CSV omits years, full timestamps, and client identifiers. Source order supports a chronological holdout, while repeated clients may span partitions and complete calendar cutoffs or customer-level separation cannot be verified.

## Privacy, fairness, and applicability

- The source omits direct identifiers but records individual financial and demographic characteristics. Publish aggregate outputs and avoid linking records to external identities or exposing example customer profiles.
- Excluding age and marital status reduces direct use of those attributes. Occupation, education, and financial features can still proxy for demographic or socioeconomic differences. Evaluate group-level contact rates and errors, assess feature necessity, and check that outreach respects consent and contact preferences before any operational use.
- The data describe a historical sample from one bank and its existing outreach policy. They may underrepresent clients whom that policy rarely contacted. Recheck discrimination, calibration, and coverage on new, permissioned data before using scores to prioritize outreach among eligible, opted-in clients.
- Observed subscriptions after outreach support a prediction task. Estimating incremental subscriptions caused by outreach requires a separate randomized comparison. Keep marketing prioritization separate from eligibility, pricing, underwriting, and other decisions about access to financial products.
