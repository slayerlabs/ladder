"""Freeze source-attributed validation documents after train-pool MinHash screening."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

SLICES = ("pl_web", "pl_prose", "pl_wiki", "pl_dialogue", "pl_technical_legal", "en_web", "en_prose", "code")


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def shingles(text):
    words = unicodedata.normalize("NFKC", text).casefold().split()
    if not words:
        raise ValueError("Empty document")
    if len(words) < 13:
        return {" ".join(words).encode()}
    return {" ".join(words[i:i + 13]).encode() for i in range(len(words) - 12)}


def jaccard(a, b):
    return len(a & b) / len(a | b)


def prefix(text, budget, tokenizer):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= budget:
        return text, len(ids)
    cut = tokenizer.decode(ids[:budget], clean_up_tokenization_spaces=False)
    if not text.startswith(cut) or len(tokenizer.encode(cut, add_special_tokens=False)) != budget:
        raise ValueError("Reference token boundary is not a lossless text prefix; change source document order")
    return cut, budget


def freeze(config_path, train_paths, candidates_path, output, tokenizer, reference_metadata,
           tokens_per_slice=150000, micro_per_slice=100000):
    from datasketch import MinHash, MinHashLSH
    from importlib.metadata import version
    config = json.loads(Path(config_path).read_text())
    if set(config["slices"]) != set(SLICES):
        raise ValueError("Configure all eight named slices")
    out_slices = [s for s, c in config["slices"].items() if c["out_of_mix"]]
    if len(out_slices) < 2:
        raise ValueError("At least two entire slices must be out of every candidate mix")
    micro_slices = config["micro_slices"]
    if len(micro_slices) != 2 or len(set(micro_slices)) != 2 or not set(micro_slices) <= set(SLICES):
        raise ValueError("Choose two distinct micro slices")
    if tokens_per_slice < micro_per_slice or micro_per_slice <= 0:
        raise ValueError("Invalid slice budgets")
    out = Path(output)
    if out.exists():
        raise ValueError("Frozen output already exists; create a new version")
    lsh = MinHashLSH(threshold=.8, num_perm=128)
    candidates, sets, rejected, ids = [], {}, {}, set()
    for doc in sorted(jsonl(candidates_path), key=lambda d: d["id"]):
        if doc["id"] in ids or not isinstance(doc["text"], str) or not doc["text"]:
            raise ValueError("Duplicate ID or invalid candidate text")
        ids.add(doc["id"])
        if doc["source"] not in config["slices"][doc["slice"]]["sources"]:
            raise ValueError("Candidate source absent from slice source allowlist")
        grams = shingles(doc["text"])
        sketch = MinHash(num_perm=128, seed=1)
        sketch.update_batch(sorted(grams))
        hits = sorted(lsh.query(sketch))
        if any(jaccard(grams, sets[k]) >= .8 for k in hits):
            rejected[doc["id"]] = "validation near-duplicate"
            continue
        lsh.insert(doc["id"], sketch)
        sets[doc["id"]] = grams
        candidates.append(doc)
    forbidden_sources = {source for s in out_slices for source in config["slices"][s]["sources"]}
    train_documents = 0
    for path in train_paths:
        for doc in jsonl(path):
            train_documents += 1
            if doc["source"] in forbidden_sources:
                raise ValueError(f'Out-of-mix source occurs in full training pool: {doc["source"]}')
            if doc["id"] in sets:
                rejected[doc["id"]] = "training document ID"
            grams = shingles(doc["text"])
            sketch = MinHash(num_perm=128, seed=1)
            sketch.update_batch(sorted(grams))
            for key in sorted(lsh.query(sketch)):
                if jaccard(grams, sets[key]) >= .8:
                    rejected[key] = "training near-duplicate"
    if not train_documents:
        raise ValueError("Full training pool must not be empty")
    frozen, totals, micro_totals = [], {}, {}
    for slice_name in SLICES:
        total = micro_total = 0
        for doc in candidates:
            if doc["slice"] != slice_name or doc["id"] in rejected or total == tokens_per_slice:
                continue
            text, n = prefix(doc["text"], tokens_per_slice - total, tokenizer)
            micro_text, mn = (prefix(text, micro_per_slice - micro_total, tokenizer)
                              if slice_name in micro_slices and micro_total < micro_per_slice else (None, 0))
            frozen.append({**doc, "text": text, "full_document_sha256": hashlib.sha256(doc["text"].encode()).hexdigest(),
                           "reference_tokens": n, "micro_text": micro_text, "micro_reference_tokens": mn,
                           "out_of_mix": config["slices"][slice_name]["out_of_mix"]})
            total += n
            micro_total += mn
        if total != tokens_per_slice or (slice_name in micro_slices and micro_total != micro_per_slice):
            raise ValueError(f"Insufficient clean documents for {slice_name}: {total} tokens")
        totals[slice_name], micro_totals[slice_name] = total, micro_total
    out.mkdir(parents=True)
    docs_path = out / "documents.jsonl"
    docs_path.write_text(''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in frozen))
    manifest = {"schema_version": 2, "documents_sha256": sha_file(docs_path), "reference_tokenizer": reference_metadata,
                "slice_tokens": totals, "micro_slice_tokens": micro_totals, "source_policy": config,
                "train_inputs": [{"path": str(p), "sha256": sha_file(p)} for p in train_paths],
                "candidate_input_sha256": sha_file(candidates_path), "train_documents_screened": train_documents,
                "dedup": {"datasketch_version": version("datasketch"), "word_ngram": 13, "threshold": .8, "num_perm": 128, "seed": 1,
                          "candidate_search": "MinHashLSH", "verification": "exact Jaccard of retrieved candidates",
                          "limitation": "LSH can miss near-duplicates; no exhaustive recall guarantee"},
                "rejected": rejected}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def load_frozen(path):
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text())
    if sha_file(root / "documents.jsonl") != manifest["documents_sha256"]:
        raise ValueError("Frozen corpus hash mismatch")
    return manifest, list(jsonl(root / "documents.jsonl"))
