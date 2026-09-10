"""
Unit tests for the rule-based expert system.

Run with:
    python3 -m unittest test_expert_system.py -v
"""

import unittest

from expert_system import InferenceEngine, KnowledgeBase, Rule


def make_sample_kb() -> KnowledgeBase:
    rules = [
        Rule(name="R1", conditions={"fever", "cough"},
             conclusion="suspect_flu", certainty_factor=0.8),
        Rule(name="R2", conditions={"suspect_flu", "body_ache"},
             conclusion="recommend_rest_and_fluids", certainty_factor=0.9),
        Rule(name="R3", conditions={"sneezing", "runny_nose", "itchy_eyes"},
             conclusion="suspect_allergy", certainty_factor=0.75),
    ]
    return KnowledgeBase(rules=rules)


class TestKnowledgeBase(unittest.TestCase):
    def test_add_new_fact(self):
        kb = make_sample_kb()
        kb.add_fact("fever", 1.0)
        self.assertIn("fever", kb.facts)
        self.assertEqual(kb.certainty_of("fever"), 1.0)

    def test_certainty_combination_increases_confidence(self):
        """Two independent pieces of supporting evidence should raise, not
        overwrite, the combined certainty (MYCIN-style combination)."""
        kb = make_sample_kb()
        kb.add_fact("suspect_flu", 0.5)
        kb.add_fact("suspect_flu", 0.5)
        # CF_combined = 0.5 + 0.5*(1-0.5) = 0.75
        self.assertAlmostEqual(kb.certainty_of("suspect_flu"), 0.75)

    def test_unknown_fact_has_zero_certainty(self):
        kb = make_sample_kb()
        self.assertEqual(kb.certainty_of("never_added"), 0.0)


class TestInferenceEngine(unittest.TestCase):
    def test_single_rule_fires(self):
        kb = make_sample_kb()
        kb.add_fact("fever", 1.0)
        kb.add_fact("cough", 1.0)
        engine = InferenceEngine(kb)
        fired = engine.run()

        self.assertIn("R1", fired)
        self.assertIn("suspect_flu", kb.facts)

    def test_multi_step_chaining(self):
        """suspect_flu (from R1) should feed into R2's condition and
        produce recommend_rest_and_fluids -- this proves rules chain."""
        kb = make_sample_kb()
        kb.add_fact("fever", 1.0)
        kb.add_fact("cough", 1.0)
        kb.add_fact("body_ache", 1.0)
        engine = InferenceEngine(kb)
        fired = engine.run()

        self.assertIn("R1", fired)
        self.assertIn("R2", fired)
        self.assertIn("recommend_rest_and_fluids", kb.facts)

    def test_rule_does_not_fire_without_all_conditions(self):
        kb = make_sample_kb()
        kb.add_fact("fever", 1.0)  # missing 'cough'
        engine = InferenceEngine(kb)
        fired = engine.run()

        self.assertNotIn("R1", fired)
        self.assertNotIn("suspect_flu", kb.facts)

    def test_unrelated_rule_group_is_independent(self):
        kb = make_sample_kb()
        kb.add_fact("sneezing", 1.0)
        kb.add_fact("runny_nose", 1.0)
        kb.add_fact("itchy_eyes", 1.0)
        engine = InferenceEngine(kb)
        fired = engine.run()

        self.assertIn("R3", fired)
        self.assertNotIn("R1", fired)
        self.assertIn("suspect_allergy", kb.facts)

    def test_conclusion_certainty_is_weakened_by_weakest_condition(self):
        """A fuzzy-AND: conclusion certainty should never exceed the
        weakest supporting condition's certainty (times the rule's own CF)."""
        kb = make_sample_kb()
        kb.add_fact("fever", 1.0)
        kb.add_fact("cough", 0.4)  # weak evidence
        engine = InferenceEngine(kb)
        engine.run()

        # rule CF (0.8) * min(1.0, 0.4) = 0.32
        self.assertAlmostEqual(kb.certainty_of("suspect_flu"), 0.32)

    def test_stops_without_infinite_loop_when_no_new_facts(self):
        kb = make_sample_kb()
        kb.add_fact("fever", 1.0)
        kb.add_fact("cough", 1.0)
        engine = InferenceEngine(kb)
        engine.run()
        fired_count_first_run = len(engine.fired_rules)

        # Running again with no new facts should not fire anything new
        engine.run()
        self.assertEqual(len(engine.fired_rules), fired_count_first_run)


if __name__ == "__main__":
    unittest.main()
