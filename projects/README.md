# Phase 2 portfolio projects

| Project | Status | Source and reports |
| --- | --- | --- |
| 01. Bank Marketing Conversion Prediction | Complete | [Bank Marketing](bank-marketing/) |
| 02. Telecom Churn & Retention Modeling | Complete | [Telecom Churn](telecom-churn/) |
| 03. Marketing A/B Testing | Complete | [Marketing A/B Testing](marketing-ab-test/) |
| 04. GA4 E-commerce Funnel & Channel Analysis | Awaiting BigQuery access | [Required dataset](https://developers.google.com/analytics/bigquery/web-ecommerce-demo-dataset) |
| 05. Online Retail Analytics & Customer Segmentation | Complete | [Online Retail](online-retail/) |
| 06. Restaurant Review Sentiment Classification | Complete | [Restaurant Sentiment](restaurant-sentiment/) |
| 07. Restaurant Visitor Forecasting | Awaiting Kaggle competition access | [Required competition rules](https://www.kaggle.com/competitions/recruit-restaurant-visitor-forecasting/rules) |

## Data access required

- **GA4.** The project needs an authorized BigQuery project and access to the official `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*` tables for November 1, 2020 through January 31, 2021. The official metadata request returned HTTP 401 in the workspace. A connected BigQuery sandbox or an authorized export from the required tables would unblock the analysis. [Google dataset guide](https://developers.google.com/analytics/bigquery/web-ecommerce-demo-dataset) · [BigQuery sandbox](https://cloud.google.com/bigquery/docs/sandbox)
- **Restaurant Visitor Forecasting.** Kaggle requires authenticated access under the accepted Recruit Restaurant Visitor Forecasting rules. These rules govern who can access, use, and share the data. The project can run once the workspace has access through an eligible account under those terms. [Competition rules](https://www.kaggle.com/competitions/recruit-restaurant-visitor-forecasting/rules)

The completed cases are served by GitHub Pages under `/Projects/`. Each case links to its reproducible source, measured outputs, and data attribution. [Live portfolio](https://asshittu.github.io/Projects/#projects)
