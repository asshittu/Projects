# Recruit Restaurant Visitor Forecasting data

This notebook uses the official Kaggle competition files inside an eligible account’s private workspace. Access and reuse follow the accepted competition rules. The public repository contains code and artificial test fixtures. The evaluation exports aggregate counts and errors only. [Kaggle data](https://www.kaggle.com/competitions/recruit-restaurant-visitor-forecasting/data) · [Competition rules](https://www.kaggle.com/competitions/recruit-restaurant-visitor-forecasting/rules)

| Input | Fields used | Purpose |
| --- | --- | --- |
| `air_visit_data.csv` | `air_store_id`, `visit_date`, `visitors` | Observed AIR store-day counts used for fitting and backtesting |
| `date_info.csv` | `calendar_date`, `holiday_flg` | Published holiday indicators for target dates. Weekday and month come from the target date. |
| `air_reserve.csv` | `air_store_id`, `visit_datetime`, `reserve_datetime`, `reserve_visitors` | Count AIR bookings and booked visitors by store and target date, using bookings recorded by the forecast origin |
| `sample_submission.csv` | Date suffix in `id` | Infer the length of the daily horizon |

Use the source timestamps as supplied. At each origin, history ends at the end of that date and reservations must be recorded before the next midnight. Historical training examples replay earlier forecast origins and use only targets observed by the current training cutoff. The holdout scores cover historical observed AIR store-days. Days without a recorded count are outside the evaluation.

The sample file supplies the horizon for the backtest. The notebook publishes no competition predictions and creates only aggregate historical results. The staffing sensitivity compares modeled capacity under an illustrative 18 visitors per shift and optional 10% headroom. Staffing, sales, and cost outcomes require separate operational data.
