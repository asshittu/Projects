# Criteo attribution data

Criteo AI Lab released this anonymized, subsampled dataset from approximately 30 days of real display-ad traffic. Each row represents an impression. An impression can carry a conversion identifier and conversion timestamp, which the project uses to associate pre-conversion impressions with a reported conversion. Campaign identifiers represent campaigns observed by Criteo. [Criteo data description](https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/)

## Source, access, and attribution

- Criteo AI Lab (2017). *Criteo Attribution Modeling for Bidding Dataset*. Cite Diemert, E., Meynet, J., Lefortier, D., & Galland, P. (2017), *Attribution Modeling Increases Efficiency of Bidding in Display Advertising*, AdKDD & TargetAd. [Criteo release and terms](https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/) · [Paper](https://arxiv.org/abs/1707.06409)
- The source is licensed CC BY-NC-SA 4.0. Reuse is limited to noncommercial purposes. Shared adaptations of the data follow the same terms and credit the source. This educational analysis selects and aggregates impressions. The output metrics, campaign table and figure are shared under that license. The raw data stay local. [License](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- Accessed November 19, 2026, through the public Criteo-hosted Hugging Face release. The pipeline pins revision `904188a63cbad78bee43cd26ff5ee4ac77903986`, checks SHA-256 `94ac7a465564349bc7ba008602211d5990a3c53cc133abc0aadef61ea2391a98`, and caches the 653,015,824-byte gzip in `data/raw/`. The uncompressed TSV is approximately 2.4 GB. [Dataset mirror](https://huggingface.co/datasets/criteo/criteo-attribution-dataset) · [Pinned gzip](https://huggingface.co/datasets/criteo/criteo-attribution-dataset/resolve/904188a63cbad78bee43cd26ff5ee4ac77903986/criteo_attribution_dataset.tsv.gz)

## Column dictionary

| Source column | Meaning and use |
| --- | --- |
| `timestamp` | Relative impression time in seconds. Orders touches and defines the observation limits. |
| `uid` | Anonymized user identifier. Used to check that a conversion identifier maps to one user. |
| `campaign` | Anonymized campaign identifier. Reporting unit. |
| `conversion` | `1` when the release associates an impression with a conversion in the next 30 days, `0` otherwise. |
| `conversion_timestamp` | Relative conversion timestamp. `-1` denotes no observed linked conversion. |
| `conversion_id` | Reported conversion identifier. `-1` denotes no observed linked conversion. Different users or times sharing an ID are flagged and excluded. |
| `attribution` | Indicator that the advertiser attributed the conversion to Criteo. Retained for schema validation. |
| `click` | Whether the displayed impression was clicked. |
| `click_pos` | Click position before a conversion. Zero denotes first click. |
| `click_nb` | Number of clicks before the conversion. |
| `cost` | Transformed impression cost. |
| `cpo` | Transformed cost per order for attributed conversions. |
| `time_since_last_click` | Seconds since the user's previous click. |
| `cat1` through `cat9` | Anonymized contextual categories with undisclosed meanings. |

## Data quality and scope

- The pinned file contains 16,468,027 impression rows. Of 435,810 reported conversion identifiers, 2,910 map to multiple users or conversion times and are excluded. The publisher's conversion count in the release description is approximate. The analysis reports its own measured counts and eligibility at each step.
- Impression timestamps span seconds 0 through 2,671,199. Some appended conversion timestamps fall after the last impression. The default analysis selects conversion times with a full seven-day lookback inside the observed impression interval, then keeps source-linked touches from the lookback start up to the instant before conversion. Keeping the full interval inside the release limits boundary effects. Subsampling and unobserved traffic can still omit touches.
- A separate sensitivity analysis anchors on those same source-linked conversions and includes any observed impression for that user inside the lookback, after their previous coherent conversion if one exists. Equal-time conversion IDs cannot be ordered and may share impressions. Other campaigns may belong to unrelated advertisers. This wider scope tests how the path definition changes reported credit without treating same-user proximity as a conversion link.
- No unique impression identifier is supplied. Each eligible source row contributes one touch, including repeat campaign exposures. First and last models split their credit among impressions tied at that exact timestamp. The linear model apportions credit evenly across eligible impressions.
- The published reports aggregate by campaign and omit user, conversion, and event-level timestamps. Keep the raw archive and any row-level intermediates private. The transformed costs and unknown campaign identities limit spend and ROI interpretation. The observed conversion paths support descriptive credit rules within Criteo display traffic.
