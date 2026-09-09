"""Tier execution on frozen corpus and reviewed pairs."""
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from .corpus import load_frozen, sha_file, jsonl
from .pairs import reviewed_pairs
from .scoring import Request, bpb, score_pairs
from .gating import scale_policy


def stable_take(items, count):
    groups = defaultdict(list)
    for item in sorted(items, key=lambda p: hashlib.sha256(p['id'].encode()).hexdigest()):
        groups[item['paradigm']].append(item)
    result, offset = [], 0
    while len(result) < count:
        added = False
        for key in sorted(groups):
            if offset < len(groups[key]) and len(result) < count:
                result.append(groups[key][offset])
                added = True
        if not added:
            break
        offset += 1
    return result


def evaluate_benchmark(scorer, corpus_path, pl_path=None, review_path=None, en_path=None,
                       tier='micro', rung=8, mc_path=None, allow_incomplete=False, allow_unreviewed=False):
    if tier not in ('micro', 'fast', 'full'):
        raise ValueError('Unknown tier')
    policy = scale_policy(rung)
    missing = []
    if allow_unreviewed and not allow_incomplete:
        raise ValueError('Unreviewed pairs require --allow-incomplete; never decision eligible')
    if corpus_path:
        manifest, documents = load_frozen(corpus_path)
    elif allow_incomplete:
        manifest, documents = {'documents_sha256': None}, []
        missing.append('Frozen validation corpus unavailable; BPB not measured')
    else:
        raise ValueError('Frozen corpus required unless --allow-incomplete is explicit')
    if allow_unreviewed and pl_path:
        pl = list(jsonl(pl_path))
        missing.append('Native Polish review pending; unreviewed candidates used')
        if review_path:
            review = json.loads(Path(review_path).read_text())
            if review['candidate_sha256'] != sha_file(pl_path):
                raise ValueError('Review does not match candidate file')
            rejected = {i for p in review['paradigms'].values() for i in p.get('rejected_ids', [])}
            pl = [p for p in pl if p['id'] not in rejected]
    else:
        pl = reviewed_pairs(pl_path, review_path) if pl_path and review_path else []
    en = list(jsonl(en_path)) if en_path else []
    if tier == 'micro':
        documents = [{**d, 'text': d['micro_text'], 'reference_tokens': d['micro_reference_tokens']}
                     for d in documents if d['micro_text']]
        selected_pairs = stable_take([p for p in pl if p['axis'] == 'agreement_pairs'], 2000)
        if len(selected_pairs) != 2000:
            missing.append('2,000 reviewed PL agreement pairs')
        expected_tokens = 200000
    else:
        expected_tokens = 1200000
        selected_pl, selected_en = stable_take(pl, 8000), stable_take(en, 4000)
        if len(selected_pl) != 8000:
            missing.append('8,000 reviewed PL pairs')
        if len(selected_en) != 4000:
            missing.append('4,000 EN pairs')
        selected_pairs = selected_pl + selected_en
        if tier == 'full':
            selected_pairs = []
            for language, pairs, n_paradigms, per_paradigm in [('pl', pl, 25, 1000), ('en', en, 67, 128)]:
                groups = defaultdict(list)
                for pair in pairs:
                    groups[pair['paradigm']].append(pair)
                if len(groups) != n_paradigms or any(len(v) < per_paradigm for v in groups.values()):
                    missing.append(f'{language}: {n_paradigms} paradigms × {per_paradigm} pairs')
                for values in groups.values():
                    selected_pairs.extend(stable_take(values, per_paradigm))
            missing.extend(['EWoK adapter', 'PolEval/KLEJ adapter', 'generation sanity adapter'])
    if sum(d['reference_tokens'] for d in documents) != expected_tokens:
        missing.append(f'{expected_tokens:,} reference validation tokens')
    mc = list(jsonl(mc_path)) if mc_path and tier != 'micro' and rung >= 50 else []
    if tier != 'micro' and rung >= 50 and len({d['task'] for d in mc}) != 3:
        missing.append('three MC tasks')
    if missing and not allow_incomplete:
        raise ValueError('Incomplete tier: ' + '; '.join(missing))
    if len({p['id'] for p in selected_pairs}) != len(selected_pairs):
        raise ValueError('Duplicate pair IDs across selected languages')
    suite = {'corpus': manifest['documents_sha256'],
             'corpus_manifest': sha_file(Path(corpus_path) / 'manifest.json') if corpus_path else None,
             'pl': sha_file(pl_path) if pl_path else None, 'review': sha_file(review_path) if review_path else None,
             'en': sha_file(en_path) if en_path else None, 'mc': sha_file(mc_path) if mc_path else None,
             'selection_version': 1, 'allow_unreviewed': allow_unreviewed}
    suite_hash = hashlib.sha256(json.dumps(suite, sort_keys=True).encode()).hexdigest()
    protocol = {k: scorer.metadata[k] for k in ('dtype', 'attention', 'deterministic_algorithms', 'batch_shape', 'stride')}
    protocol.update(version=2, scoring='sum-target-logprob; exact original UTF8 bytes; BOS reset per document',
                    pair_primary='sentence_prob; region_prob reported separately', mc_primary='correct-choice BPB')
    scored_docs = scorer.score([Request(d['text']) for d in documents])
    groups = defaultdict(list)
    for doc, result in zip(documents, scored_docs):
        groups[doc['slice']].append(result)
        if doc['out_of_mix']:
            groups['out_of_mix'].append(result)
    metrics = {}
    corpus_results = {}
    for key, values in groups.items():
        nll = math.fsum(v['nll_nats'] for v in values)
        size = sum(v['bytes'] for v in values)
        metrics[f'bpb/{key}'] = bpb(nll, size)
        corpus_results[key] = {'nll_nats': nll, 'bytes': size, 'bpb': metrics[f'bpb/{key}']}
    pair_rows = score_pairs(scorer, selected_pairs) if selected_pairs else []
    pair_groups = defaultdict(list)
    for row in pair_rows:
        pair_groups[f"{row['language']}/{row['paradigm']}"] .append(row)
    pair_results = {}
    for key, values in pair_groups.items():
        metrics[f'pair/{key}/sentence_prob'] = statistics.mean(v['sentence_prob'] for v in values)
        region = [v for v in values if v['region_prob'] is not None]
        if region:
            metrics[f'pair/{key}/region_prob'] = statistics.mean(v['region_prob'] for v in region)
        pair_results[key] = {'n': len(values), 'region_n': len(region),
                             'sentence_prob': metrics[f'pair/{key}/sentence_prob'],
                             'sentence_accuracy': statistics.mean(v['sentence_accuracy'] for v in values),
                             'sentence_log_margin': statistics.mean(v['sentence_log_margin'] for v in values),
                             'region_prob': statistics.mean(v['region_prob'] for v in region) if region else None}
    mc_rows = []
    if mc:
        requests = []
        for doc in mc:
            if not doc['correct']:
                raise ValueError('Empty correct continuation')
            text = doc['prefix'] + doc['correct']
            requests.append(Request(text, (len(doc['prefix']), len(text))))
        mc_scores = scorer.score(requests)
        mc_groups = defaultdict(list)
        for doc, value in zip(mc, mc_scores):
            mc_rows.append({'id': doc['id'], 'task': doc['task'], **value})
            mc_groups[doc['task']].append(value)
        for task, values in mc_groups.items():
            metrics[f'mc_bpb/{task}'] = bpb(math.fsum(v['nll_nats'] for v in values), sum(v['bytes'] for v in values))
    return {'schema_version': 2, 'suite_sha256': suite_hash, 'suite_inputs': suite,
            'protocol_sha256': hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
            'protocol': protocol, 'model': scorer.metadata, 'tier': tier, 'rung': rung,
            'decision_eligible': tier == 'fast' and not missing, 'complete': not missing, 'missing': missing,
            'coverage': {'pairs': len(selected_pairs), 'pairs_with_region': sum(p.get('region') is not None for p in selected_pairs),
                         'validation_documents': len(documents), 'validation_reference_tokens': sum(d['reference_tokens'] for d in documents)},
            'scale_policy': policy, 'metrics': metrics, 'corpus': corpus_results,
            'pairs': pair_results, 'pair_items': pair_rows, 'mc_items': mc_rows,
            'notes': ['Scale statuses are priors until seed separation is measured.',
                      'Micro never settles decisions; partial tiers are never decision eligible.',
                      'Any reported timing applies only to the measured coverage and recorded hardware.']}
