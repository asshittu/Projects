"""Regenerate the simulated experiment's aggregate results and figures."""

import argparse
import json
from pathlib import Path

from .analysis import analyze_trial
from .figures import write_figures
from .simulation import (
    ASSIGNMENT_PROBABILITY,
    DEFAULT_SEED,
    DEFAULT_WORKSPACES,
    PLANTED_RATES,
    generate_trial,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workspaces", type=int, default=DEFAULT_WORKSPACES)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    metrics = analyze_trial(generate_trial(seed=args.seed, workspaces=args.workspaces))
    metrics["generator"] = {
        "seed": args.seed,
        "workspaces": args.workspaces,
        "guided_assignment_probability": ASSIGNMENT_PROBABILITY,
        "planted_rates": PLANTED_RATES,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    write_figures(metrics, args.output_dir / "figures")
    print(f"Wrote simulated aggregate metrics and figures under {args.output_dir}")


if __name__ == "__main__":
    main()
