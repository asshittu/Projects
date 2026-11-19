"""Estimate the unadjusted conversion-rate contrast between ad and PSA users."""

from math import erfc, hypot, sqrt
from statistics import NormalDist

import pandas as pd
from pandas.api.types import is_bool_dtype

CONFIDENCE_LEVEL = 0.95
PRACTICAL_HURDLE = 0.005
REPORTING_POPULATION = 10_000
CRITICAL_Z = NormalDist().inv_cdf(0.5 + CONFIDENCE_LEVEL / 2)


def wilson_interval(converted: int, n: int) -> dict[str, float]:
    """Calculate a two-sided Wilson score interval for one conversion rate."""
    rate = converted / n
    z_squared = CRITICAL_Z ** 2
    denominator = 1 + z_squared / n
    center = (rate + z_squared / (2 * n)) / denominator
    radius = CRITICAL_Z * sqrt(rate * (1 - rate) / n + z_squared / (4 * n ** 2)) / denominator
    return {"lower": max(0.0, center - radius), "upper": min(1.0, center + radius)}


def newcombe_interval(ad: dict, psa: dict) -> dict[str, float]:
    """Calculate Newcombe's uncorrected hybrid Wilson interval for ad minus PSA."""
    ad_rate = ad["conversion_rate"]
    psa_rate = psa["conversion_rate"]
    difference = ad_rate - psa_rate
    ad_ci = ad["wilson_95_ci"]
    psa_ci = psa["wilson_95_ci"]
    return {
        "lower": difference - hypot(ad_rate - ad_ci["lower"], psa_ci["upper"] - psa_rate),
        "upper": difference + hypot(ad_ci["upper"] - ad_rate, psa_rate - psa_ci["lower"]),
    }


def pooled_z_test(ad: dict, psa: dict) -> tuple[float | None, float | None]:
    """Test equal proportions with a pooled standard error and two-sided p-value."""
    pooled_rate = (ad["converted"] + psa["converted"]) / (ad["n"] + psa["n"])
    if pooled_rate in (0.0, 1.0):
        return None, None

    pooled_error = sqrt(pooled_rate * (1 - pooled_rate) * (1 / ad["n"] + 1 / psa["n"]))
    statistic = (ad["conversion_rate"] - psa["conversion_rate"]) / pooled_error
    return statistic, erfc(abs(statistic) / sqrt(2))


def run_analysis(data: pd.DataFrame) -> dict:
    """Analyze one primary comparison using validated groups and binary outcomes."""
    if not {"test_group", "converted"} <= set(data.columns):
        raise ValueError("Analysis requires test_group and converted columns")
    groups = data["test_group"]
    converted = data["converted"]
    if not groups.isin(["ad", "psa"]).all() or set(groups) != {"ad", "psa"}:
        raise ValueError("Analysis requires nonempty ad and psa groups")
    if not is_bool_dtype(converted) or converted.isna().any():
        raise ValueError("Conversion outcomes must be nonmissing Booleans")

    comparison = {}
    for treatment in ("ad", "psa"):
        outcomes = converted[groups.eq(treatment)]
        n = len(outcomes)
        successes = int(outcomes.sum())
        comparison[treatment] = {
            "n": n,
            "converted": successes,
            "conversion_rate": successes / n,
            "observed_share": n / len(data),
            "wilson_95_ci": wilson_interval(successes, n),
        }

    ad, psa = comparison["ad"], comparison["psa"]
    difference = ad["conversion_rate"] - psa["conversion_rate"]
    interval = newcombe_interval(ad, psa)
    statistic, p_value = pooled_z_test(ad, psa)
    if interval["upper"] < PRACTICAL_HURDLE:
        interval_relation = "entirely_below"
    elif interval["lower"] >= PRACTICAL_HURDLE:
        interval_relation = "entirely_at_or_above"
    else:
        interval_relation = "spans_hurdle"

    return {
        "confidence_level": CONFIDENCE_LEVEL,
        "groups": comparison,
        "contrast": {
            "treatment": "ad",
            "comparator": "psa",
            "absolute_difference": difference,
            "newcombe_95_ci": interval,
            "relative_lift": difference / psa["conversion_rate"] if psa["converted"] else None,
            "pooled_z_statistic": statistic,
            "two_sided_p_value": p_value,
            "additional_conversions_per_10000": difference * REPORTING_POPULATION,
            "additional_conversions_95_ci": {
                limit: rate * REPORTING_POPULATION for limit, rate in interval.items()
            },
        },
        "practical_hurdle": {
            "minimum_absolute_difference": PRACTICAL_HURDLE,
            "point_estimate_meets_hurdle": difference >= PRACTICAL_HURDLE,
            "interval_relation": interval_relation,
            "basis": "Illustrative planning threshold, with no campaign costs supplied",
        },
        "design": {
            "primary_contrasts": 1,
            "assignment_procedure_documented": False,
            "planned_allocation_known": False,
            "sample_ratio_mismatch_tested": False,
            "allocation_note": "Observed group shares are measured. Testing for a sample ratio mismatch requires the intended allocation.",
            "outcome": "converted",
            "excluded_post_exposure_fields": ["total_ads", "most_ads_day", "most_ads_hour"],
            "test_assumption": "The reported pooled z test and intervals treat the user records as independent.",
        },
    }
