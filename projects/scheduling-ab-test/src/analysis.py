"""Analyze both assigned arms on an intention-to-treat basis."""

from collections.abc import Sequence
from math import erfc, hypot, sqrt
from statistics import NormalDist

from .simulation import ASSIGNMENT_PROBABILITY, WorkspaceOutcome

CONFIDENCE_LEVEL = 0.95
CRITICAL_Z = NormalDist().inv_cdf(0.5 + CONFIDENCE_LEVEL / 2)
SRM_ALPHA = 0.01
OUTCOMES = {
    "publish": ("published_within_7d", 0.02),
    "support": ("support_request_within_7d", 0.01),
}
ARMS = ("control", "guided")


def wilson_interval(events: int, n: int) -> dict[str, float]:
    """Estimate a two-sided Wilson score interval for one event rate."""
    rate = events / n
    z_squared = CRITICAL_Z ** 2
    denominator = 1 + z_squared / n
    center = (rate + z_squared / (2 * n)) / denominator
    radius = CRITICAL_Z * sqrt(rate * (1 - rate) / n + z_squared / (4 * n ** 2)) / denominator
    return {"lower": max(0.0, center - radius), "upper": min(1.0, center + radius)}


def difference_interval(guided: dict, control: dict) -> dict[str, float]:
    """Estimate Newcombe's hybrid Wilson interval for guided minus control."""
    guided_rate, control_rate = guided["rate"], control["rate"]
    difference = guided_rate - control_rate
    return {
        "lower": difference - hypot(guided_rate - guided["interval"]["lower"], control["interval"]["upper"] - control_rate),
        "upper": difference + hypot(guided["interval"]["upper"] - guided_rate, control_rate - control["interval"]["lower"]),
    }


def primary_test(arms: dict) -> dict[str, float | str | None]:
    """Run the two-sided pooled two-proportion z-test for publishing."""
    guided, control = arms["guided"], arms["control"]
    pooled = (guided["events"] + control["events"]) / (guided["n"] + control["n"])
    smallest_cell = min(arm["n"] * min(pooled, 1 - pooled) for arm in arms.values())
    statistic = None
    p_value = None
    if smallest_cell >= 5:
        error = sqrt(pooled * (1 - pooled) * (1 / guided["n"] + 1 / control["n"]))
        statistic = (guided["rate"] - control["rate"]) / error
        p_value = erfc(abs(statistic) / sqrt(2))
    return {
        "method": "Two-sided pooled two-proportion z-test",
        "minimum_expected_cell_count": smallest_cell,
        "z_statistic": statistic,
        "p_value": p_value,
    }


def allocation_check(counts: dict[str, int]) -> dict:
    """Compare the observed allocation with the known Bernoulli assignment mix."""
    n = sum(counts.values())
    expected = {"control": n * (1 - ASSIGNMENT_PROBABILITY), "guided": n * ASSIGNMENT_PROBABILITY}
    chi_square = sum((counts[arm] - expected[arm]) ** 2 / expected[arm] for arm in ARMS)
    p_value = erfc(sqrt(chi_square / 2)) if min(expected.values()) >= 5 else None
    return {
        "method": "Pearson chi-squared goodness-of-fit (1 df)",
        "observed": counts,
        "expected": expected,
        "chi_square": chi_square,
        "p_value": p_value,
        "alpha": SRM_ALPHA,
        "flagged": p_value < SRM_ALPHA if p_value is not None else None,
    }


def analyze_trial(rows: Sequence[WorkspaceOutcome]) -> dict:
    """Keep each eligible workspace in its originally assigned arm."""
    rows_by_arm = {arm: [] for arm in ARMS}
    seen = set()
    for row in rows:
        if row.arm not in rows_by_arm:
            raise ValueError("Each workspace must be assigned to control or guided")
        if row.workspace_id in seen:
            raise ValueError("Each workspace must occur exactly once")
        if type(row.published_within_7d) is not bool or type(row.support_request_within_7d) is not bool:
            raise ValueError("Both seven-day outcomes must be observed Booleans")
        seen.add(row.workspace_id)
        rows_by_arm[row.arm].append(row)

    counts = {arm: len(rows_by_arm[arm]) for arm in ARMS}
    if not all(counts.values()):
        raise ValueError("Both assigned arms must contain an eligible workspace")

    outcomes = {}
    for name, (outcome_field, threshold) in OUTCOMES.items():
        arms = {}
        for arm in ARMS:
            events = sum(getattr(row, outcome_field) for row in rows_by_arm[arm])
            n = counts[arm]
            arms[arm] = {"n": n, "events": events, "rate": events / n, "interval": wilson_interval(events, n)}
        guided, control = arms["guided"], arms["control"]
        difference = guided["rate"] - control["rate"]
        outcomes[name] = {
            "arms": arms,
            "contrast": {
                "difference": difference,
                "interval": difference_interval(guided, control),
                "relative_lift": difference / control["rate"] if control["events"] else None,
            },
            "threshold": threshold,
        }

    outcomes["publish"]["test"] = primary_test(outcomes["publish"]["arms"])
    allocation = allocation_check(counts)
    allocation_passes = allocation["flagged"] is False
    publish_passes = outcomes["publish"]["contrast"]["interval"]["lower"] > outcomes["publish"]["threshold"]
    support_passes = outcomes["support"]["contrast"]["interval"]["upper"] < outcomes["support"]["threshold"]
    return {
        "source": "synthetic",
        "unit": "workspace",
        "analysis": "intention-to-treat by original assignment",
        "n": sum(counts.values()),
        "confidence_level": CONFIDENCE_LEVEL,
        "allocation": allocation,
        "outcomes": outcomes,
        "decision": {
            "publish_lower_bound_above_threshold": publish_passes,
            "support_upper_bound_below_tolerance": support_passes,
            "allocation_check_passed": allocation_passes,
            "passes_simulated_criteria": publish_passes and support_passes and allocation_passes,
        },
    }
