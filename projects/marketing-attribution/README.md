# Marketing attribution for display campaigns

Which campaigns receive conversion credit under first-touch, last-touch, and linear rules? This study applies those rules to anonymized Criteo display impressions. It tests whether campaign rankings are sensitive to which observations are considered part of the conversion path. The intended decision is whether the path definition is reliable enough for routine attribution reporting. [Data card](DATASET.md) · [Criteo source](https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/)

## Results

The pipeline scanned 16,468,027 impressions. It required each reported conversion ID to identify one user and one conversion time. It kept anchors with seven days of observed prehistory and a conversion time inside the impression observation period. An eligible touch had to occur strictly before conversion and within those seven days.

| Path definition | Conversions with touches | Impressions counted | Paths with multiple campaigns |
| --- | ---: | ---: | ---: |
| Source-linked impressions | 241,450 | 381,898 | 0 |
| Same-user sensitivity | 241,039 | 623,011 | 41,337 (17.15%) |

The three rules gave identical campaign totals for source-linked impressions. All eligible linked paths stayed within one campaign, even though 65,145 included multiple impressions. The agreement follows from the path definition.

The sensitivity analysis starts with the source-linked anchors and adds eligible impressions for the same anonymized user. It keeps touches strictly after the user's most recent earlier coherent conversion, when available. This removed every eligible touch for 411 anchors. The 94 anchors with equal-time conversion IDs are kept as separate reported IDs and can share touches. The resulting paths may include ads from unrelated advertisers.

In the same-user analysis, campaign `10341182` received the most first-touch credit (11,420.5), while campaign `9100693` led the last-touch and linear rules (12,284.0 and 11,761.0 credits). Campaign `2869134` had the largest absolute first-to-last difference, from 5,815 to 6,790 credits. Each method assigns exactly one total credit per eligible path, including split credits at tied timestamps. [Metrics](reports/metrics.json) · [Campaign table](reports/campaign-credit.csv) · [Sensitivity table](reports/campaign-credit-same-user.csv)

Use the source-linked report when comparing documented conversion links. Keep the same-user results as a sensitivity check and investigate linking, measurement coverage, and advertiser boundaries before relying on ranks that change. A randomized holdout or other credible causal design would be needed to measure incremental impact. Named channels, complete paths, and actual campaign costs are outside the release's scope.

## Reproduce the analysis

The pipeline uses Python 3.12 and DuckDB. Its first run downloads the pinned 653 MB gzip into `data/raw/`, verifies the SHA-256, and scans the compressed source. The query reads approximately 2.4 GB of uncompressed text without saving it to disk.

```sh
cd projects/marketing-attribution
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.run
python -m unittest discover -s tests -v
```

For a previously downloaded copy, use `python -m src.run --source-file /path/to/criteo_attribution_dataset.tsv.gz`. `--lookback-days` accepts 1 through 30 days. Use `--reports-dir` for alternative runs so the default seven-day reports stay available for comparison.

The [SQL transformations](sql/attribution.sql) validate conversion coherence, reconstruct timestamp-bounded paths, count equal-time ties, and sum campaign credits. The suite uses in-memory fixtures and requires no download. It checks clipped lookbacks, ambiguous IDs, exclusion of touches at conversion, repeated campaigns, timestamp ties, reconciliation, and input validation.

The [source-linked chart](reports/campaign-credit.svg) and [same-user chart](reports/campaign-credit-same-user.svg) each show the seven campaigns with the highest linear credit in that path definition. Only aggregate data are exported. The report adaptations are shared under the source’s CC BY-NC-SA 4.0 terms. [Report attribution](reports/ATTRIBUTION.md)
