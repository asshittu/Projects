"""Generate a reproducible trial with no customer or employer records."""

from dataclasses import dataclass
from random import Random

DEFAULT_SEED = 42
DEFAULT_WORKSPACES = 12_000
ASSIGNMENT_PROBABILITY = 0.50
PLANTED_RATES = {
    "publish": {"control": 0.55, "guided": 0.60},
    "support": {"control": 0.030, "guided": 0.032},
}


@dataclass(frozen=True)
class WorkspaceOutcome:
    """One eligible synthetic workspace and its complete seven-day outcomes."""

    workspace_id: int
    arm: str
    published_within_7d: bool
    support_request_within_7d: bool


def generate_trial(
    *, seed: int = DEFAULT_SEED, workspaces: int = DEFAULT_WORKSPACES,
) -> list[WorkspaceOutcome]:
    """Draw independent assignments and conditional binary outcomes."""
    if workspaces < 2:
        raise ValueError("The simulation requires at least two eligible workspaces")

    rng = Random(seed)
    rows = []
    for workspace_id in range(workspaces):
        arm = "guided" if rng.random() < ASSIGNMENT_PROBABILITY else "control"
        rows.append(WorkspaceOutcome(
            workspace_id=workspace_id,
            arm=arm,
            published_within_7d=rng.random() < PLANTED_RATES["publish"][arm],
            support_request_within_7d=rng.random() < PLANTED_RATES["support"][arm],
        ))
    return rows
