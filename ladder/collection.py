"""Collect actual checkpoint reports into the analysis input without hand-copying scores."""
import json
from pathlib import Path


def collect(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    selected = manifest['metrics']
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('Select unique continuous metric IDs explicitly')
    if any(not key.startswith(('bpb/', 'pair/', 'mc_bpb/')) or
           (key.startswith('pair/') and not key.endswith(('/sentence_prob', '/region_prob'))) for key in selected):
        raise ValueError('Only BPB, pair probabilities, and correct-choice BPB are decision metrics')
    result = []
    for entry in manifest['runs']:
        checkpoints = []
        signature = None
        for checkpoint in entry['checkpoints']:
            report_path = path.parent / checkpoint['report']
            report = json.loads(report_path.read_text())
            current = (report['suite_sha256'], report['protocol_sha256'], report['tier'], report['decision_eligible'])
            if signature is not None and current != signature:
                raise ValueError('Checkpoint suite/protocol/tier differs within a run')
            signature = current
            if report['rung'] != entry['rung']:
                raise ValueError('Checkpoint report rung differs from experiment manifest')
            if not set(selected) <= set(report['metrics']):
                raise ValueError(f'Missing selected metrics in {report_path}')
            checkpoints.append({'step': checkpoint['step'], 'report': str(report_path),
                                'metrics': {key: report['metrics'][key] for key in selected}})
        if signature is None:
            raise ValueError('Empty checkpoint list')
        result.append({**{key: entry[key] for key in ('rung', 'mix', 'seed', 'training_config_sha256')},
                       'suite_sha256': signature[0], 'protocol_sha256': signature[1],
                       'tier': signature[2], 'decision_eligible': signature[3], 'checkpoints': checkpoints})
    return result
