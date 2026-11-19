# Restaurant Visitor Forecasting

This private evaluation uses the [Recruit Restaurant Visitor Forecasting competition data](https://www.kaggle.com/competitions/recruit-restaurant-visitor-forecasting/data) in an eligible Kaggle account. Review the competition rules before using the data. The source archive and notebook execution stay in your private workspace. [Competition rules](https://www.kaggle.com/competitions/recruit-restaurant-visitor-forecasting/rules)

## Run in a private Kaggle notebook

1. Open [restaurant-forecasting.ipynb](restaurant-forecasting.ipynb) on GitHub and use **Download raw file**.
2. Import the downloaded notebook into Kaggle and keep its visibility **Private**.
3. Use **Add Input** to attach the official Recruit Restaurant Visitor Forecasting competition data. Run all cells. The final cell also supports the official KaggleHub download in the authorized Kaggle notebook if the files are unavailable under the mounted inputs.
4. Download `restaurant-forecasting-results.zip` from Kaggle’s working output. This archive contains aggregate metrics and two charts. Share that archive only if the competition rules permit sharing those outputs. Keep the source files and credentials private.

The evaluation uses only the AIR records. It infers the forecast horizon from the sample’s dates and evaluates nonoverlapping historical windows with actual visitor counts. The first two windows support model selection and the latest is a holdout. Both models use the same observed store-days. Missing store-days stay missing. The seasonal baseline repeats the latest observed same weekday, with store and overall fallbacks. The tree model uses earlier observations, calendar features and reservations booked by the applicable forecast cutoff. The reporting archive contains no store identifiers, reservation rows or individual predictions. Forecast errors measure visitor counts and any staffing conversion uses stated scenario assumptions.

## Test the code locally

The test suite uses artificial fixtures. The forecast has yet to run on the competition records in an authorized notebook.

```sh
cd projects/restaurant-forecasting
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python build_notebook.py
```
