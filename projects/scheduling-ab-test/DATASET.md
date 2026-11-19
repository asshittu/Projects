# Scheduling A/B Test data

The study uses entirely synthetic records. Each row represents one fictitious workspace and its first eligible manager. The local generator creates all records from the documented assumptions and a fixed seed of 42. It uses no external source data and requires no credentials. [Generator](src/simulation.py) · [Reproduction steps](README.md#reproduce-the-simulation)

## Assignment and outcomes

For each of 12,000 workspaces, draw a uniform random value from `random.Random(42)` and assign `guided` if it is below 0.50. Use the next two independent draws to generate whether the workspace publishes a schedule and whether it requests support within seven days, in that order. The probability of each outcome depends on the assigned arm.

| Assigned experience | Assignment probability | Planted publishing probability | Planted support request probability |
| --- | ---: | ---: | ---: |
| Standard editor (`control`) | 0.50 | 0.55 | 0.030 |
| Guided publishing flow (`guided`) | 0.50 | 0.60 | 0.032 |

The seven-day window is encoded in the outcome definitions. The generator produces binary indicators and does not simulate individual event times. The outcomes are independent conditional on assignment. A workspace may publish, request support, do both, or do neither.

## Fields

| Field | Meaning |
| --- | --- |
| `workspace_id` | Unique synthetic integer identifier, 0 through 11,999 in the default run. |
| `arm` | `control` or `guided`, assigned once per workspace. |
| `published_within_7d` | Boolean first-schedule publishing outcome and the primary metric. |
| `support_request_within_7d` | Boolean support request outcome and the guardrail. |

## Inclusion and reporting

The generator creates a complete outcome window for every assigned workspace. The intention-to-treat analysis keeps every workspace in its assigned arm. A false outcome indicates that the corresponding event did not occur in the window. The measured outcomes and assignment counts are synthetic sample values. The 50/50 expectation and the planted event probabilities are generator inputs.

The analysis checks that each identifier appears once, both arms are present, and both outcomes are Booleans. Only aggregate counts, rates, intervals, and charts are written to reports. The fictitious identifiers are held in memory during the run. [Analysis source](src/analysis.py)
