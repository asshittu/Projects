"""Assemble the standalone notebook from the tested Python modules."""

import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent


def cell(kind, identifier, source):
    """Build a clean notebook cell with a stable identifier."""
    result = {
        "cell_type": kind,
        "id": identifier,
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


intro = """
# Restaurant Visitor Forecasting

Use this notebook in a private Kaggle workspace with access under the [competition rules](https://www.kaggle.com/competitions/recruit-restaurant-visitor-forecasting/rules). Attach the official competition data using **Add Input**, then run all cells. The loader can read the supplied ZIP files directly. The final cell creates an archive with aggregate error metrics and charts. Keep the competition data and notebook run private.
"""

run = '''
import os

# Optional local overrides accept a directory or ZIP in an eligible workspace.
input_override = os.getenv("RESTAURANT_FORECAST_INPUT_DIR")
output_override = os.getenv("RESTAURANT_FORECAST_OUTPUT_DIR")
input_root = Path(input_override or "/kaggle/input")

try:
    visits, calendar, reservations, horizon_days = load_competition_data(input_root)
except FileNotFoundError:
    if input_override or not Path("/kaggle/working").is_dir():
        raise
    import kagglehub

    downloaded = kagglehub.competition_download("recruit-restaurant-visitor-forecasting")
    visits, calendar, reservations, horizon_days = load_competition_data(downloaded)

forecast_results = evaluate_forecasts(visits, calendar, reservations, horizon_days=horizon_days)
output_directory = Path(output_override or "/kaggle/working")
results_archive = write_report(forecast_results, output_directory)

for fold in forecast_results["folds"]:
    print(fold["name"], fold["models"])
print("Aggregate results archive:", results_archive)
'''

cells = [cell("markdown", "introduction", intro)]
for name in ("data", "forecast", "report"):
    source = (PROJECT / "src" / f"{name}.py").read_text(encoding="utf-8")
    cells.append(cell("code", f"module-{name}", source))
cells.append(cell("code", "run-private-evaluation", run))
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "file_extension": ".py", "mimetype": "text/x-python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(PROJECT / "restaurant-forecasting.ipynb").write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8",
)
