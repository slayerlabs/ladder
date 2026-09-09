import math
import unittest
from dataclasses import replace
from ladder.tasks import FAMILIES, digest, generate, suite, verify
from ladder.evaluation import compare, evaluate, score_item, wilson


class LadderTests(unittest.TestCase):
    def test_all_generators_ground_truth_and_unique_choices(self):
        for family in FAMILIES:
            for level in range(1, 13):
                for index in range(100):
                    verify(generate(family, level, index))

    def test_determinism_balance_and_no_duplicates(self):
        items = suite(32)
        self.assertEqual(digest(items), digest(suite(32)))
        self.assertNotEqual(digest(items), digest(suite(32, split="dev")))
        fingerprints = [(i.prompt, tuple(sorted(i.choices))) for i in items]
        self.assertEqual(len(set(fingerprints)), len(items))
        for family in FAMILIES:
            for level in (1, 2, 3):
                answers = [i.answer for i in items if i.family == family and i.difficulty == level]
                self.assertEqual([answers.count(i) for i in range(4)], [8] * 4)

    def test_documented_full_profile(self):
        self.assertEqual(len(suite(256, levels=(1, 2, 3, 4, 5, 6))), 10752)

    def test_oracle_and_uniform_control(self):
        items = suite(32)
        oracle = evaluate(items, "oracle")
        self.assertEqual(oracle["highest_cleared_tier"], "D")
        control = evaluate(items, "chance")
        self.assertIsNone(control["highest_cleared_tier"])
        accuracy = sum(r["correct"] for r in control["items"]) / len(items)
        self.assertLess(abs(accuracy - .25), .06)
        for cell in control["cells"].values():
            self.assertAlmostEqual(cell["choice_nll"], math.log(4))
            self.assertEqual(cell["tie_rate"], 1)

    def test_stable_nll_and_normalization(self):
        item = generate("copy", 1, 0)
        scores = [-10000.] * 4
        scores[item.answer] = -9990.
        result = score_item(item, scores)
        self.assertEqual(result["correct"], 1)
        self.assertAlmostEqual(result["choice_nll"], math.log(1 + 3 * math.exp(-10)), places=10)
        for invalid in ([0.], [float('nan')] * 4, [float('-inf')] * 4, [1.] * 4):
            with self.assertRaises(ValueError):
                score_item(item, invalid)

    def test_paired_comparison_and_mismatch(self):
        items = suite(4, levels=(1,))
        control = evaluate(items)
        diff = compare(control, control, draws=100)
        for cell in diff["cells"].values():
            self.assertEqual(cell["accuracy_delta"], 0)
            self.assertEqual(cell["paired_bootstrap_ci95"], [0, 0])
        with self.assertRaises(ValueError):
            compare(control, evaluate(suite(4, levels=(2,))))

    def test_small_sample_does_not_clear(self):
        self.assertLess(wilson(4, 4)[0], .75)
        self.assertIsNone(evaluate(suite(4, levels=(1,)), "oracle")["highest_cleared_tier"])

    def test_bad_configuration_and_corrupted_gold(self):
        for count in (-4, 0, 3, 5):
            with self.assertRaises(ValueError):
                suite(count)
        with self.assertRaises(ValueError):
            suite(4, levels=(1, 1))
        item = generate("lookup", 1, 0)
        with self.assertRaises(ValueError):
            verify(replace(item, answer=(item.answer + 1) % 4))


if __name__ == '__main__':
    unittest.main()
