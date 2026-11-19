"""Check randomization, intention-to-treat denominators, and interval calculations."""

import json
import unittest

from src.analysis import analyze_trial
from src.simulation import DEFAULT_WORKSPACES, WorkspaceOutcome, generate_trial


class SchedulingTrialTests(unittest.TestCase):
    def test_seeded_generator_reproduces_every_workspace_and_assigned_outcome(self):
        rows = generate_trial()
        self.assertEqual(rows, generate_trial())
        self.assertNotEqual(rows, generate_trial(seed=12))
        self.assertEqual(len(rows), DEFAULT_WORKSPACES)
        self.assertEqual(len({row.workspace_id for row in rows}), DEFAULT_WORKSPACES)
        result = analyze_trial(rows)
        self.assertEqual(result["source"], "synthetic")
        self.assertEqual(result["allocation"]["observed"], {"control": 6028, "guided": 5972})
        self.assertAlmostEqual(result["allocation"]["p_value"], 0.6092056132701683)
        self.assertTrue(result["decision"]["passes_simulated_criteria"])
        self.assertNotIn("workspace_id", json.dumps(result, allow_nan=False))

    def test_nonpublishers_stay_in_both_outcome_denominators(self):
        rows = [
            WorkspaceOutcome(1, "control", False, True),
            WorkspaceOutcome(2, "control", True, False),
            WorkspaceOutcome(3, "guided", True, False),
            WorkspaceOutcome(4, "guided", False, False),
            WorkspaceOutcome(5, "guided", False, True),
        ]
        results = analyze_trial(rows)
        for endpoint in ("publish", "support"):
            arms = results["outcomes"][endpoint]["arms"]
            self.assertEqual((arms["control"]["events"], arms["control"]["n"]), (1, 2))
            self.assertEqual((arms["guided"]["events"], arms["guided"]["n"]), (1, 3))
        self.assertIsNone(results["allocation"]["p_value"])
        self.assertIsNone(results["outcomes"]["publish"]["test"]["p_value"])
        self.assertFalse(results["decision"]["passes_simulated_criteria"])

    def test_wilson_and_newcombe_intervals_match_fixed_counts(self):
        rows = (
            [WorkspaceOutcome(i, "control", i < 5, False) for i in range(50)]
            + [WorkspaceOutcome(50 + i, "guided", i < 25, False) for i in range(100)]
        )
        result = analyze_trial(rows)
        publish = result["outcomes"]["publish"]
        self.assertAlmostEqual(publish["arms"]["guided"]["interval"]["lower"], 0.17545211362287674)
        self.assertAlmostEqual(publish["arms"]["control"]["interval"]["upper"], 0.21360231437479654)
        self.assertAlmostEqual(publish["contrast"]["difference"], 0.15)
        self.assertAlmostEqual(publish["contrast"]["interval"]["lower"], 0.01412184430672242)
        self.assertAlmostEqual(publish["contrast"]["interval"]["upper"], 0.25886823844333856)
        self.assertAlmostEqual(publish["test"]["p_value"], 0.03038282197657753)
        self.assertTrue(result["allocation"]["flagged"])
        self.assertFalse(result["decision"]["publish_lower_bound_above_threshold"])
        self.assertIsNone(result["outcomes"]["support"]["contrast"]["relative_lift"])

    def test_duplicate_missing_and_unassigned_records_are_rejected(self):
        control = WorkspaceOutcome(1, "control", False, False)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            analyze_trial([control, WorkspaceOutcome(1, "guided", True, True)])
        with self.assertRaisesRegex(ValueError, "assigned to control or guided"):
            analyze_trial([control, WorkspaceOutcome(2, "other", True, True)])
        with self.assertRaisesRegex(ValueError, "Both seven-day outcomes"):
            analyze_trial([control, WorkspaceOutcome(2, "guided", None, True)])
        with self.assertRaisesRegex(ValueError, "Both assigned arms"):
            analyze_trial([control])
