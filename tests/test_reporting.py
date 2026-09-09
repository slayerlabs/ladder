import copy
import unittest
from scripts.report_micro_comparison import validate


class ReportingTests(unittest.TestCase):
    def report(self):
        return {'suite_sha256': 'suite', 'protocol_sha256': 'protocol', 'tier': 'micro',
                'coverage': {'pairs': 1, 'pairs_with_region': 1},
                'pair_items': [{'id': 'one', 'language': 'pl', 'paradigm': 'SV-P',
                                'sentence_prob': .8, 'region_prob': .7}]}

    def test_comparison_checks_protocol_and_pair_identity(self):
        left = self.report()
        validate(left, copy.deepcopy(left))
        for key, value in [('suite_sha256', 'other'), ('protocol_sha256', 'other')]:
            right = copy.deepcopy(left)
            right[key] = value
            with self.assertRaises(ValueError):
                validate(left, right)
        right = copy.deepcopy(left)
        right['pair_items'][0]['id'] = 'other'
        with self.assertRaises(ValueError):
            validate(left, right)

    def test_comparison_rejects_missing_regions_and_nonfinite_scores(self):
        left = self.report()
        for field, value in [('region_prob', None), ('sentence_prob', float('nan'))]:
            right = copy.deepcopy(left)
            right['pair_items'][0][field] = value
            with self.assertRaises(ValueError):
                validate(left, right)


if __name__ == '__main__':
    unittest.main()
