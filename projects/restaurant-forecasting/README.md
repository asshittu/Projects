# Restaurant Visitor Forecasting

How accurately can an operations team forecast daily visitor counts 39 days ahead when it only knows booking and visitor history up through the forecast date? I compared a same-weekday baseline with gradient boosting on three historical windows of the Recruit Restaurant Visitor Forecasting competition data. The completed run used an eligible account’s private Kaggle notebook. [Kaggle competition data](https://www.kaggle.com/competitions/recruit-restaurant-visitor-forecasting/data) · [Competition rules](https://www.kaggle.com/competitions/recruit-restaurant-visitor-forecasting/rules)

## Results

| Historical set | Observed store-days | Seasonal naïve MAE | Boosting MAE | Seasonal naïve RMSLE | Boosting RMSLE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation 1, Dec 27, 2016–Feb 3, 2017 | 24,958 | 11.92 | 7.71 | 0.798 | 0.601 |
| Validation 2, Feb 4–Mar 14, 2017 | 27,597 | 8.67 | 6.96 | 0.678 | 0.530 |
| Pooled validation | 52,555 | 10.21 | 7.31 | 0.738 | 0.565 |
| Holdout, Mar 15–Apr 22, 2017 | 28,031 | 9.68 | 7.45 | 0.688 | 0.542 |

Boosting had the lower pooled validation RMSLE and reduced holdout MAE by 23.0% and holdout RMSLE by 21.2% relative to the baseline. MAE is expressed in visitors. RMSLE compares `log1p` visitor counts, which accommodates zero counts and reduces the influence of large absolute misses.

The aggregate export from the private run was checked for chronological folds, sample counts, pooled errors and model selection. Both plots were regenerated from the exported metrics. [Full results](reports/metrics.json) · [MAE by window](reports/figures/mae-by-window.svg) · [RMSLE by window](reports/figures/rmsle-by-window.svg)

## Method

The loader validates required columns, dates, nonnegative counts, store-date uniqueness and calendar coverage. It drops reservations recorded after their visit time. Evaluation uses observed AIR visitor counts and leaves unrecorded store-days outside the analysis. The three equal-length windows contain 80,586 observed store-days in total, ranging from 24,958 to 28,031 per window.

At each origin, the baseline repeats the latest same-weekday count for each store, falling back to the store or overall mean. The tree model adds lead time, calendar, store history, recent averages and counts from AIR bookings recorded by the cutoff. Training replays prior forecast snapshots using only counts observed by that fold's origin. Model selection uses pooled RMSLE across the first two windows, then the same protocol is refit for the final holdout.

## Capacity scenario

Assuming 18 visitors per shift, adding 10% headroom to the selected model’s forecasts would increase total modeled shift slots from 47,141 to 50,042. The number of observed holdout store-days exceeding that planned capacity falls from 4,933 (17.60%) to 3,903 (13.92%). The planner rounds up to whole shift slots each day.

Test any headroom policy against real service targets, role coverage, costs and demand by time of day. This backtest covers historical AIR store-days with observed counts. It cannot measure absent or closed days, staffing savings or outcomes under a live scheduling policy.

## Run in a private Kaggle notebook

1. Open [restaurant-forecasting.ipynb](restaurant-forecasting.ipynb) on GitHub and use **Download raw file**.
2. Import the downloaded notebook into Kaggle and keep its visibility **Private**.
3. Use **Add Input** to attach the official Recruit Restaurant Visitor Forecasting competition data. Run all cells. The loader accepts loose CSVs, individual CSV ZIP files, and a single competition ZIP. It can also use the official KaggleHub download in an authorized Kaggle notebook if it cannot find the mounted files.
4. Download `restaurant-forecasting-results.zip` from Kaggle’s working output. This archive contains aggregate metrics and two charts. Share that archive only if the competition rules permit sharing those outputs. Keep the source files and credentials private.

The evaluation uses only the AIR records. It infers the forecast horizon from the sample’s dates and evaluates nonoverlapping historical windows with actual visitor counts. The first two windows support model selection and the latest is a holdout. Both models use the same observed store-days. Missing store-days stay missing. The seasonal baseline repeats the latest observed same weekday, with store and overall fallbacks. The tree model uses earlier observations, calendar features and reservations booked by the applicable forecast cutoff. The reporting archive contains no store identifiers, reservation rows or individual predictions. Forecast errors measure visitor counts and any staffing conversion uses stated scenario assumptions.

## Test the code locally

The test suite uses artificial fixtures and runs without Kaggle credentials or competition data.

```sh
cd projects/restaurant-forecasting
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python build_notebook.py
```
