import json
import math
from pathlib import Path
import tempfile
import unittest
from ladder.scoring import bpb, pair_probability
from ladder.analysis import calibrate, rank_agreement, decision, summarize_runs, spearman
from ladder.gating import scale_policy
from ladder.pairs import critical_region, reviewed_pairs
from ladder.corpus import freeze, load_frozen, SLICES, shingles, jaccard


def experiment():
    runs = []
    for rung in (8, 25, 120):
        for m, mix in enumerate(('control', 'mix1', 'mix2', 'mix3')):
            for seed in range(3):
                values = {'bpb/out_of_mix': 4 - m * .2 + seed * .01,
                          'pair/pl/SV-P/sentence_prob': .5 + m * .04 + seed * .001}
                runs.append({'rung': rung, 'mix': mix, 'seed': seed, 'tier': 'fast',
                             'decision_eligible': True, 'suite_sha256': 'suite', 'protocol_sha256': 'fp32',
                             'training_config_sha256': f'{rung}/{mix}',
                             'checkpoints': [{'step': step, 'metrics': values} for step in (100, 200, 300)]})
    return runs


class ContinuousTests(unittest.TestCase):
    def test_probability_extremes_and_unicode_bpb(self):
        self.assertEqual(pair_probability(-10000, -10000), .5)
        self.assertEqual(pair_probability(-1, -10000), 1)
        self.assertEqual(pair_probability(-10000, -1), 0)
        self.assertAlmostEqual(bpb(math.log(2) * len('żółć'.encode()), len('żółć'.encode())), 1)

    def test_critical_region_includes_space_and_full_word(self):
        g, b = 'Oni piszą list.', 'Oni pisze list.'
        region = critical_region(g, b)
        self.assertEqual(g[slice(*region['good'])], ' piszą')
        self.assertEqual(b[slice(*region['bad'])], ' pisze')
        self.assertIsNone(critical_region('Oni piszą.', 'Ona pisze.'))

    def test_calibration_last_three_seeds_and_adoption(self):
        runs = experiment()
        runs[0]['checkpoints'].insert(0, {'step': 0, 'metrics': {k: 999 for k in runs[0]['checkpoints'][0]['metrics']}})
        calibration = calibrate(runs, 'control')
        self.assertAlmostEqual(calibration['rungs']['8']['bpb/out_of_mix']['sigma_seed'], .01)
        self.assertTrue(calibration['rungs']['8']['bpb/out_of_mix']['dev_composite_eligible'])
        self.assertEqual(decision(runs, 'control', 'mix1', 8, ['pair/pl/SV-P/sentence_prob'])['decision'], 'adopt')
        for run in runs:
            if run['mix'] == 'mix1':
                for checkpoint in run['checkpoints']:
                    checkpoint['metrics']['pair/pl/SV-P/sentence_prob'] = .1
        self.assertEqual(decision(runs, 'control', 'mix1', 8, ['pair/pl/SV-P/sentence_prob'])['decision'], 'investigate')

    def test_rank_agreement_and_ties(self):
        self.assertAlmostEqual(spearman([1, 2, 2, 4], [4, 2, 2, 1]), -1)
        self.assertIsNone(spearman([1, 1], [1, 2]))
        results = rank_agreement(experiment())
        entry = results['metrics']['bpb/out_of_mix']['8']
        self.assertAlmostEqual(entry['rho'], 1)
        self.assertAlmostEqual(entry['exact_two_sided_permutation_p'], 2 / 24)

    def test_incomplete_or_incompatible_runs_rejected(self):
        for field, value in [('tier', 'micro'), ('decision_eligible', False), ('suite_sha256', 'different')]:
            runs = experiment()
            runs[0][field] = value
            with self.assertRaises(ValueError):
                summarize_runs(runs)
        with self.assertRaises(ValueError):
            calibrate([r for r in experiment() if r['seed'] != 2], 'control')

    def test_control_only_leaves_snr_unresolved(self):
        results = calibrate([r for r in experiment() if r['mix'] == 'control'], 'control')
        metric = results['rungs']['8']['bpb/out_of_mix']
        self.assertIsNone(metric['snr'])
        self.assertIsNone(metric['dev_composite_eligible'])

    def test_collector_refuses_accuracy_as_decision_metric(self):
        from ladder.collection import collect
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'experiment.json'
            path.write_text(json.dumps({'metrics': ['pair/pl/SV-P/accuracy'], 'runs': []}))
            with self.assertRaisesRegex(ValueError, 'decision metrics'):
                collect(path)

    def test_policy_is_prior_and_mc_gate(self):
        for rung in (8, 25, 50):
            self.assertEqual(scale_policy(rung)['metrics']['mc_accuracy']['status'], 'off')
        self.assertEqual(scale_policy(120)['metrics']['ewok_knowledge']['status'], 'canary')

    def test_corpus_freeze_dedup_hash_and_source_exclusion(self):
        class CharacterTokenizer:
            def encode(self, text, **kwargs): return list(text)
            def decode(self, ids, **kwargs): return ''.join(ids)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = {'slices': {s: {'sources': [s + '.example'], 'out_of_mix': i < 2} for i, s in enumerate(SLICES)},
                   'micro_slices': list(SLICES[:2])}
            (root / 'sources.json').write_text(json.dumps(cfg))
            docs = [{'id': s, 'slice': s, 'source': s + '.example',
                     'text': ' '.join(s + str(i) for i in range(30))} for s in SLICES]
            (root / 'candidates.jsonl').write_text(''.join(json.dumps(d) + '\n' for d in docs))
            train = {'id': 'train1', 'source': 'train.example', 'text': 'different training material with no shared content'}
            (root / 'train.jsonl').write_text(json.dumps(train) + '\n')
            freeze(root / 'sources.json', [root / 'train.jsonl'], root / 'candidates.jsonl', root / 'frozen',
                   CharacterTokenizer(), {'fixture': True}, tokens_per_slice=20, micro_per_slice=10)
            manifest, frozen = load_frozen(root / 'frozen')
            self.assertEqual(sum(d['reference_tokens'] for d in frozen), 160)
            self.assertEqual(sum(d['micro_reference_tokens'] for d in frozen), 20)
            (root / 'frozen/documents.jsonl').write_text('tampered')
            with self.assertRaises(ValueError):
                load_frozen(root / 'frozen')
            train['source'] = docs[0]['source']
            (root / 'train.jsonl').write_text(json.dumps(train))
            with self.assertRaisesRegex(ValueError, 'Out-of-mix'):
                freeze(root / 'sources.json', [root / 'train.jsonl'], root / 'candidates.jsonl', root / 'bad',
                       CharacterTokenizer(), {}, 20, 10)
            train = {**docs[2], 'source': 'train.example'}
            (root / 'train.jsonl').write_text(json.dumps(train))
            with self.assertRaisesRegex(ValueError, 'Insufficient'):
                freeze(root / 'sources.json', [root / 'train.jsonl'], root / 'candidates.jsonl', root / 'near',
                       CharacterTokenizer(), {}, 20, 10)
        text = ' '.join(str(i) for i in range(40))
        self.assertEqual(jaccard(shingles(text), shingles(text.upper())), 1)

    def test_explicit_unreviewed_micro_stays_incomplete(self):
        from ladder.benchmark import evaluate_benchmark
        class FakeScorer:
            metadata = {'dtype': 'float32', 'attention': 'fixture', 'deterministic_algorithms': True,
                        'batch_shape': [8, 512], 'stride': 256}
            def score(self, requests):
                return [{'nll_nats': 1., 'bytes': len(r.text.encode()), 'tokens': 2} for r in requests]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'pairs.jsonl'
            path.write_text(json.dumps({'id': 'one', 'paradigm': 'agreement', 'language': 'pl',
                'axis': 'agreement_pairs', 'good': 'Oni piszą.', 'bad': 'Oni pisze.', 'region': None}) + '\n')
            with self.assertRaisesRegex(ValueError, 'allow-incomplete'):
                evaluate_benchmark(FakeScorer(), None, pl_path=path, allow_unreviewed=True)
            report = evaluate_benchmark(FakeScorer(), None, pl_path=path,
                                       allow_incomplete=True, allow_unreviewed=True)
            self.assertFalse(report['decision_eligible'])
            self.assertFalse(report['complete'])
            self.assertEqual(report['coverage']['validation_reference_tokens'], 0)
            self.assertEqual(report['coverage']['pairs'], 1)
            self.assertEqual(report['corpus'], {})
            self.assertEqual(report['pair_items'][0]['sentence_prob'], .5)

    def test_koliber_adapter_rejects_unreviewed_revision(self):
        from ladder.adapters import koliber_snapshot, KOLIBER_REPO
        with self.assertRaisesRegex(ValueError, 'exact pinned revision'):
            koliber_snapshot(KOLIBER_REPO, 'main')

    def test_review_pending_is_not_admitted(self):
        from ladder.corpus import sha_file
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'pairs.jsonl'
            path.write_text(json.dumps({'id': 'p', 'paradigm': 'agreement'}) + '\n')
            review = Path(temp) / 'review.json'
            review.write_text(json.dumps({'candidate_sha256': sha_file(path), 'paradigms': {'agreement': {'status': 'pending'}}}))
            with self.assertRaisesRegex(ValueError, 'native review'):
                reviewed_pairs(path, review)


if __name__ == '__main__':
    unittest.main()
