"""Publish a checked, paired micro comparison; refuses mismatched evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

START = '<!-- micro-results:start -->'
END = '<!-- micro-results:end -->'


def load(path):
    return json.loads(Path(path).read_text())


def validate(left, right):
    for key in ('suite_sha256', 'protocol_sha256', 'tier', 'coverage'):
        if left[key] != right[key]:
            raise ValueError(f'Mismatched {key}')
    if left['tier'] != 'micro':
        raise ValueError('Only micro reports are supported')
    for report in (left, right):
        rows = report['pair_items']
        if len(rows) != report['coverage']['pairs'] or len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Invalid or duplicate pair coverage')
        if sum(r['region_prob'] is not None for r in rows) != report['coverage']['pairs_with_region']:
            raise ValueError('Invalid region count')
    if [r['id'] for r in left['pair_items']] != [r['id'] for r in right['pair_items']]:
        raise ValueError('Pair IDs/order differ')
    for a, b in zip(left['pair_items'], right['pair_items']):
        if (a['language'], a['paradigm']) != (b['language'], b['paradigm']):
            raise ValueError('Pair annotations differ')
        if (a['region_prob'] is None) != (b['region_prob'] is None):
            raise ValueError('Region coverage differs')
        for row in (a, b):
            if not math.isfinite(row['sentence_prob']) or not 0 <= row['sentence_prob'] <= 1:
                raise ValueError('Invalid probability')
            if row['region_prob'] is not None and (not math.isfinite(row['region_prob']) or not 0 <= row['region_prob'] <= 1):
                raise ValueError('Invalid region probability')
    if not left['pair_items']:
        raise ValueError('Empty pair suite')


def aggregate(report):
    rows = report['pair_items']
    region = [r for r in rows if r['region_prob'] is not None]
    return {'model': report['model']['model'], 'revision': report['model']['requested_revision'],
            'parameters': report['model']['parameters'], 'pairs': len(rows), 'region_pairs': len(region),
            'sentence_prob': statistics.mean(r['sentence_prob'] for r in rows),
            'region_prob': statistics.mean(r['region_prob'] for r in region) if region else None,
            'sentence_accuracy': statistics.mean(r['sentence_accuracy'] for r in rows),
            'evaluation_seconds': report['timing_seconds']['evaluation'],
            'total_seconds': report['timing_seconds']['total'], 'paradigms': report['pairs']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--koliber', required=True)
    parser.add_argument('--pollock', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--readme', required=True)
    args = parser.parse_args()
    left, right = load(args.koliber), load(args.pollock)
    validate(left, right)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    summaries = {'koliber': aggregate(left), 'pollock': aggregate(right)}
    difference = {metric: summaries['pollock'][metric] - summaries['koliber'][metric]
                  for metric in ('sentence_prob', 'region_prob', 'sentence_accuracy')}
    comparison = {'status': 'provisional, unreviewed Polish agreement only; no BPB',
                  'suite_sha256': left['suite_sha256'], 'protocol_sha256': left['protocol_sha256'],
                  'coverage': left['coverage'], 'models': summaries,
                  'pollock_minus_koliber': difference, 'decision_eligible': False,
                  'notes': ['No seed variance, significance, or general model superiority is inferred.',
                            'Pollock is English-trained according to its model card; this dataset is Polish.',
                            'Native review and training-overlap audit are pending.']}
    for name, report in [('koliber', left), ('pollock', right), ('comparison', comparison)]:
        (output / f'{name}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob('*.json'))}
    (output / 'sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    rel = output.as_posix()
    lines = [START, '## Measured Polish micro — provisional', '',
             'Measured on an Apple M4 Max (40-core GPU, 64 GB), using MPS, fp32, math SDPA, fixed batch 8 × 512, and no sampling.', '',
             '| Model | Parameters | Sentence pair probability ↑ | Critical-region probability ↑ | Sentence accuracy (diagnostic) | Evaluation time |',
             '|---|---:|---:|---:|---:|---:|']
    for label, key in [('Koliber v1.1 Base Preview', 'koliber'), ('Pollock 1.4', 'pollock')]:
        r = summaries[key]
        lines.append(f"| [{label}](https://huggingface.co/{r['model']}/tree/{r['revision']}) | {r['parameters']:,} | {r['sentence_prob']:.4f} | {r['region_prob']:.4f} | {r['sentence_accuracy']:.2%} | {r['evaluation_seconds']:.1f} s |")
    lines += ['', f"Both models scored the **same {left['coverage']['pairs']:,} pair IDs**; critical-region results cover the same **{left['coverage']['pairs_with_region']:,} pairs**. Probabilities are averaged over items, not inferred from accuracy. Runtime excludes model loading and output serialization.", '',
              '**Scope:** provisional Polish subject–verb agreement candidates from MultiBLiMP. Native review and training-overlap checks are pending. The frozen 200k-token validation corpus is unavailable, so **BPB was not measured** and this is not a complete micro tier. Pollock’s model card lists English training, while Koliber is primarily Polish; this is not a matched-language or overall-capability ranking. No seed-calibrated adoption claim is made.', '',
              f"[Comparison and reproduction details]({rel}/README.md) · [Machine-readable comparison]({rel}/comparison.json) · [Koliber item scores]({rel}/koliber.json) · [Pollock item scores]({rel}/pollock.json)", END]
    readme = Path(args.readme)
    text = readme.read_text()
    section = '\n'.join(lines)
    if START in text:
        before, rest = text.split(START, 1)
        _, after = rest.split(END, 1)
        text = before + section + after
    else:
        marker = '## Current implementation'
        if marker not in text:
            raise ValueError('README insertion marker missing')
        text = text.replace(marker, section + '\n\n' + marker, 1)
    text = text.replace('**Not yet measured:** real corpus scores, seed noise, rung admission, runtimes, or rank agreement.',
                        '**Not yet measured:** real corpus scores, seed noise, rung admission, complete-tier runtimes, or rank agreement. Provisional pair-only results and runtimes are reported above.')
    readme.write_text(text)
    details = ['# Provisional Polish micro comparison', '',
               'This folder publishes actual model outputs, not timing projections or synthetic oracle controls.', '',
               'The top-level README contains the overall results. Differences below are Pollock minus Koliber; they are descriptive and are not divided by unmeasured seed noise.', '',
               '| Paradigm | Pairs | Koliber sentence probability | Pollock sentence probability | Difference |',
               '|---|---:|---:|---:|---:|']
    names = {'pl/SV-#': 'Subject–verb number', 'pl/SV-G': 'Subject–verb gender', 'pl/SV-P': 'Subject–verb person'}
    for key in left['pairs']:
        a, b = left['pairs'][key], right['pairs'][key]
        details.append(f"| {names.get(key, key)} | {a['n']} | {a['sentence_prob']:.4f} | {b['sentence_prob']:.4f} | {b['sentence_prob'] - a['sentence_prob']:+.4f} |")
    details += ['', '## Evidence', '',
                '- `koliber.json` and `pollock.json`: complete reports including item IDs, probabilities, raw log probabilities, metadata, missing coverage and measured times.',
                '- `comparison.json`: aggregate metrics and paired differences; `sha256.json`: hashes of the three evidence files.',
                '- Identical suite and protocol hashes, identical ordered pair IDs and matching critical-region availability were checked before publication.',
                '- Models use their own tokenizers and document-start tokens; those are recorded in each report. No shared tokenizer is substituted.',
                '- The Koliber custom adapter validates source/config hashes at a pinned revision and forces math SDPA. Standard GPT-2 Pollock uses the same math backend.',
                '- CPU accumulation follows MPS-to-CPU transfer before float64 conversion; forward computation stays fp32.', '',
                '## Reproduce', '',
                'From the repository root, prepare the pinned public source and deterministic review pool. Native review remains pending; the flags below explicitly enable provisional diagnostics only.', '',
                '```sh', 'uv sync --extra dev', 'mkdir -p data/raw',
                'curl -fL https://huggingface.co/datasets/jumelet/multiblimp/resolve/de923efa8d2483d6b13364ee68e65308e990a991/pol/data.tsv -o data/raw/multiblimp-pol.tsv',
                '.venv/bin/ladder import-multiblimp --tsv data/raw/multiblimp-pol.tsv --revision de923efa8d2483d6b13364ee68e65308e990a991 --output data/multiblimp-pl-v0', '```', '',
                'If the prepared pool already exists, retain it and verify its hashes rather than overwriting it.', '']
    for key, report in [('koliber', left), ('pollock', right)]:
        adapter = ' --adapter koliber' if key == 'koliber' else ''
        details += ['```sh', f".venv/bin/ladder eval --model {report['model']['model']} --revision {report['model']['requested_revision']}{adapter} --attention sdpa_math \\",
                    '  --pl-pairs data/multiblimp-pl-v0/candidates.jsonl --review data/multiblimp-pl-v0/review.json \\',
                    '  --tier micro --rung 120 --device mps --batch-size 8 --context 512 \\',
                    f'  --allow-incomplete --allow-unreviewed --output runs/{key}-micro-provisional.json', '```', '']
    details += ['Both runs use reporting rung 120; actual parameter counts remain explicit. Commands refuse existing outputs.', '',
                'No BPB, native validation, seed calibration, or training-overlap assessment is supplied by this experiment. A matched English probe is needed before drawing conclusions about Pollock’s intended language domain.']
    (output / 'README.md').write_text('\n'.join(details) + '\n')
    print(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    main()
