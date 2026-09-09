"""Import review candidates; admission requires a separately supplied native review."""
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from .corpus import jsonl, sha_file


def critical_region(good, bad):
    g, b = list(re.finditer(r'\S+', good)), list(re.finditer(r'\S+', bad))
    if len(g) != len(b):
        return None
    changed = [i for i, (x, y) in enumerate(zip(g, b)) if x.group() != y.group()]
    if len(changed) != 1:
        return None
    i = changed[0]
    start = g[i - 1].end() if i else 0
    bad_start = b[i - 1].end() if i else 0
    if good[:start] != bad[:bad_start]:
        return None
    return {"good": [start, g[i].end()], "bad": [bad_start, b[i].end()]}


def import_multiblimp(path, revision, output):
    out = Path(output)
    if out.exists():
        raise ValueError("Review pool already exists; create a new version")
    pairs, seen = [], set()
    with open(path) as f:
        for index, row in enumerate(csv.DictReader(f, delimiter='\t')):
            if not row['phenomenon'].startswith('SV-') or row['lang'] != 'pol':
                continue
            good, bad = row['sen'], row['wrong_sen']
            if good == bad or (good, bad) in seen:
                continue
            seen.add((good, bad))
            region = critical_region(good, bad)
            # Agreement target must have its controller in the left context.
            if int(float(row['child_idx'])) >= int(float(row['head_idx'])):
                region = None
            pairs.append({"id": f"multiblimp-pol/{revision}/{index}", "good": good, "bad": bad,
                          "language": "pl", "paradigm": row['phenomenon'], "axis": "agreement_pairs",
                          "region": region, "origin": "jumelet/multiblimp", "revision": revision,
                          "source_metadata": row['metadata'], "native_review": "pending"})
    out.mkdir(parents=True)
    target = out / 'candidates.jsonl'
    target.write_text(''.join(json.dumps(p, ensure_ascii=False) + '\n' for p in pairs))
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair['paradigm']].append(pair)
    review = {"candidate_sha256": sha_file(target), "paradigms": {}}
    for paradigm, values in sorted(groups.items()):
        selected = sorted(values, key=lambda p: hashlib.sha256(p['id'].encode()).hexdigest())[:100]
        review['paradigms'][paradigm] = {"status": "pending", "reviewer": None, "native_polish": False,
                                       "sample_ids": [p['id'] for p in selected], "reviewed_ids": [], "rejected_ids": []}
    (out / 'review.json').write_text(json.dumps(review, indent=2) + '\n')
    sample_ids = {i for p in review['paradigms'].values() for i in p['sample_ids']}
    (out / 'review-sample.jsonl').write_text(''.join(json.dumps(p, ensure_ascii=False) + '\n' for p in pairs if p['id'] in sample_ids))
    manifest = {"source": "https://huggingface.co/datasets/jumelet/multiblimp", "revision": revision,
                "raw_sha256": sha_file(path), "candidate_sha256": sha_file(target), "pairs": len(pairs),
                "paradigms": {p: len(v) for p, v in groups.items()}, "status": "native review required; not admitted"}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def reviewed_pairs(path, review_path):
    review = json.loads(Path(review_path).read_text())
    if review['candidate_sha256'] != sha_file(path):
        raise ValueError("Review does not match candidate file")
    pairs = list(jsonl(path))
    ids = {p['id']: p for p in pairs}
    if len(ids) != len(pairs):
        raise ValueError("Duplicate pair IDs")
    allowed, rejected = set(), set()
    for paradigm, entry in review['paradigms'].items():
        if entry['status'] != 'approved':
            continue
        checked = set(entry['reviewed_ids'])
        expected = set(entry['sample_ids'])
        if (not entry['reviewer'] or not entry['native_polish'] or len(expected) < 100 or
            not expected <= checked or any(i not in ids or ids[i]['paradigm'] != paradigm for i in checked)):
            raise ValueError(f"Incomplete native review for {paradigm}")
        if not set(entry['rejected_ids']) <= checked:
            raise ValueError("Rejected IDs must have been reviewed")
        allowed.add(paradigm)
        rejected.update(entry['rejected_ids'])
    result = [p for p in pairs if p['paradigm'] in allowed and p['id'] not in rejected]
    if not result:
        raise ValueError("No paradigms have passed native review")
    return result
