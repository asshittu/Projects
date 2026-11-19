# Sentiment Labelled Sentences

The UCI collection (ID 331) contains positive and negative review sentences from Yelp, IMDb, and Amazon. This project uses the Yelp subset for sentence-level restaurant sentiment. The authors selected clearly positive and negative examples from larger review collections, aiming to leave out neutral sentences. Each domain starts with 500 examples of each label.

## Source, access, and attribution

- Kotzias, D. (2015). *Sentiment Labelled Sentences* [Dataset]. UCI Machine Learning Repository. [UCI catalog](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences) · [DOI 10.24432/C57604](https://doi.org/10.24432/C57604)
- The archive also requests citation of Kotzias, D., Denil, M., de Freitas, N., & Smyth, P. (2015). *From Group to Individual Labels Using Deep Features*. Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. [Associated paper](https://doi.org/10.1145/2783258.2783380)
- UCI lists the dataset under Creative Commons Attribution 4.0 International. This project selects the Yelp subset, removes repeated sentences for evaluation, and attributes reproduced excerpts to the dataset. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- The official ZIP was retrieved November 18, 2026 without an account. For offline use, save it as `data/raw/sentiment-labelled-sentences.zip` within `projects/restaurant-sentiment/`. The loader selects `sentiment labelled sentences/yelp_labelled.txt`. Follow the [project README](README.md) for run commands. [Official archive](https://archive.ics.uci.edu/static/public/331/sentiment%2Blabelled%2Bsentences.zip)
- The downloaded ZIP had SHA-256 `afc26626d710899948693e1a61405dce197f57ffa719fa1130d346b4cc095343`. The Yelp member is pinned to SHA-256 `c76468b7b5c6e56a0804d728345c5f84aa2142ddb214420f61cc9cfd4c00d2ea`.

## Fields and quality checks

The file uses tab-separated lines. The sentence and its binary label form the two fields. The released Yelp member has 1,000 valid pairs, with 500 labels of each class.

| Field | Meaning | Use |
| --- | --- | --- |
| `text` | Review sentence or sentence fragment from the Yelp subset | Input to text feature extraction |
| `label` | Dataset polarity code, `1` for positive and `0` for negative | Classification target |

Examples from the Yelp member include “The fries were great too.” (positive) and “The service was meh.” (negative). These excerpts reproduce the source text. [Kotzias, *Sentiment Labelled Sentences* (2015)](https://doi.org/10.24432/C57604) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

To identify duplicate sentences before splitting, normalize each comparison key using Unicode NFKC, collapse consecutive whitespace to one space, strip surrounding whitespace, and apply `casefold()`. Keep the first original representative of each key and reject conflicting labels for the same key. The Yelp file contains four repeated rows under this rule and zero conflicting label groups. The prepared set has 996 sentences, including 499 positive and 497 negative. Splitting before fitting TF-IDF keeps the held-out vocabulary and document frequencies out of training.

## Limits and privacy

- The balanced sample emphasizes clearly labeled sentiment. Its class proportions and metrics describe this benchmark, and real feedback can include neutral, mixed, or subtle sentiment at different proportions.
- The labels capture sentence polarity. They leave out topics, resolution urgency, and full review context. Negation, sarcasm, or mixed views can be difficult to interpret in an isolated fragment.
- The release omits reviewer IDs, restaurant IDs, collection dates, and review-level grouping. Exact-text deduplication reduces repeated-sentence overlap. It cannot verify that held-out sentences come from separate reviewers, restaurants, or reviews.
- Validate any use with recent, representative, permissioned feedback in the intended setting. Keep human review for ambiguous cases, and use aggregated summaries for reporting. Public examples should stay brief, omit identifying detail, and preserve source and license attribution.
