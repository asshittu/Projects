# Marketing A/B Testing data

The public dataset records whether each person in a marketing campaign saw an advertisement or a public service announcement (PSA), and whether they converted. The primary analysis compares conversion rates between the recorded exposure groups at the user level.

## Source, access, and use

- Favio Vázquez (`faviovaz`). *Marketing A/B Testing* [Dataset], version 1, released October 20, 2021. Kaggle lists the contributor and license in its public metadata. [Dataset card](https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing) · [Kaggle API metadata](https://www.kaggle.com/api/v1/datasets/view/faviovaz/marketing-ab-testing)
- Kaggle lists the license as `CC0: Public Domain`. The analysis uses Kaggle’s documented public download access and publishes aggregate counts and charts. Kaggle’s active Terms of Use became effective June 22, 2025. The project credits the contributor and keeps the CSV and user identifiers in the local raw-data cache. [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) · [Kaggle Terms of Use](https://www.kaggle.com/terms) · [Public download access](https://www.kaggle.com/product-announcements/485439)
- Accessed November 18, 2026. Kaggle’s versioned download endpoint returned a ZIP without account credentials. For offline use, save that archive as `data/raw/marketing-ab-testing.zip` within `projects/marketing-ab-test/`. It contains `marketing_AB.csv`. Follow the [project README](README.md) to run the analysis. [Official version 1 archive](https://www.kaggle.com/api/v1/datasets/download/faviovaz/marketing-ab-testing?datasetVersionNumber=1)
- The ZIP had SHA-256 `669e02e4c032d8378f59b73a1dbc52e5d0c396286d787de62570cda0b1d977f0`. The loader verifies the CSV against SHA-256 `de6ce8def1a6559e9e5ab364fce07cfe2e84f02dd407c3586eea55743c7e17f2`.

## Column dictionary

| Source column | Meaning and coding | Use in the analysis |
| --- | --- | --- |
| Unnamed index | Exported row number, `0` through `588100` | File integrity check |
| `user id` | Integer user identifier, unique in this release | Unit of analysis and uniqueness check. Excluded from aggregate outputs. |
| `test group` | `ad` for users recorded as seeing an advertisement, `psa` for users recorded as seeing the PSA | Exposure grouping field |
| `converted` | Boolean outcome. The publisher describes `True` as a purchase and `False` as no purchase. | One binary conversion outcome per user |
| `total ads` | Publisher’s count of ads seen per person. Observed values range from `1` to `2065`. | Data quality or descriptive exposure summary only |
| `most ads day` | English weekday of peak recorded exposure. All seven weekdays appear. | Data quality or descriptive exposure summary only |
| `most ads hour` | Hour of peak recorded exposure, `0` through `23`. The time zone is unspecified. | Data quality or descriptive exposure summary only |

`total ads`, `most ads day`, and `most ads hour` are measured after exposure begins. Conditioning on them can change the comparison and introduce selection bias. The primary analysis uses `test group` and `converted`, with user identifiers used only to validate the unit of analysis.

## Data quality and allocation

The published CSV contains 588,101 rows and seven columns. All users have unique identifiers, the row index is sequential, and the file contains zero blank cells. Both exposure groups contain conversions and non-conversions.

| Recorded group | Users | Conversions | Share of records |
| --- | ---: | ---: | ---: |
| Advertisement (`ad`) | 564,577 | 14,423 | 96.00% |
| Public service announcement (`psa`) | 23,524 | 420 | 4.00% |

The publisher describes a large ad group and a small PSA group. The project treats 96% and 4% as observed shares because the planned assignment ratio is unspecified. A formal sample-ratio mismatch test needs an independently specified allocation. The file also omits the randomization procedure, assignment logs, and pre-exposure covariates that could support a balance check.

## Interpretation and privacy

- The release labels observed exposure and every record has at least one logged impression. Users who were assigned but received zero impressions may be absent. Assignment-based intention-to-treat analysis would require the original assignment population and logs.
- The observed conversion gap measures a difference between recorded groups. A causal interpretation would require credible randomized assignment, complete outcome measurement, and comparable inclusion of users across arms. The file omits dates, campaign details, and the conversion window, so generalizability to other audiences and campaigns remains unknown.
- The source gives purchase indicators and omits revenue, media cost, and margin. The observed gap can inform planning for a better-documented follow-up experiment. An ROI or profit estimate would require financial and campaign cost data.
- Keep user IDs and row-level exposure history in the local raw-data cache. Publish counts, rates, and uncertainty intervals at group level, and avoid linking identifiers to other data.
