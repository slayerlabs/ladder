# Tiny LLM benchmark ladder

A continuous evaluation pipeline for **8M → 25M → 50M → 120M**, with three separate axes: **metrics**, **evaluation cadence**, and **empirical scale admission**. The ladder is validated by cross-rung agreement on data-mix rankings.

The default CLI now runs standalone Transformers/PyTorch scorers; **no `lm-eval-harness` dependency** is needed. The original synthetic suite remains available as `ladder-synthetic`, documented in [the legacy notes](docs/legacy-synthetic.md). Its accuracy tiers are not training-decision criteria.

## Current implementation

- Frozen validation documents: reference-token budgets, exact byte preservation, source allowlists, two out-of-mix slices, document-ID exclusion, 13-word-gram MinHash screening against the supplied full training pool, and exact Jaccard verification at 0.8 for retrieved matches.
- Deterministic fp32 scoring with eager attention, fixed padded batch shapes, document resets, and bounded sliding context. No decoding or sampling during development evaluation.
- Corpus BPB, full-sentence pair probability, critical-region pair probability, and correct-choice MC BPB. Pair accuracy and log margin are diagnostics only.
- Micro/fast/full coverage checks. Incomplete tiers are explicitly marked and cannot settle decisions.
- Last-three-checkpoint averaging; seed SD, SNR and empirical admission; the out-of-mix BPB/no-pair-regression decision rule.
- Spearman rank agreement between 8M/25M and 120M, with exact permutation p-values for the same 4–6 candidate mixes.

**Not yet measured:** real corpus scores, seed noise, rung admission, runtimes, or rank agreement. The full training pool, held-out sources, and trained checkpoints have not been supplied. PL paradigms beyond agreement, the EN dataset, natural MC datasets, and release-only adapters are not built yet.

## Install and verify

```sh
uv sync --extra dev
.venv/bin/python -m unittest discover -s tests -v
HF_HUB_OFFLINE=1 .venv/bin/python tests/scoring_smoke.py
.venv/bin/ladder --help
```

The offline scorer test uses a locally created random model solely for numerical validation. It checks manual NLL agreement, repeatability, Unicode bytes, long-window coverage, region scoring, fixed shapes, and fp32 parameters. It is not learned-capability evidence.

## 1. Freeze validation slices

Use [configs/sources.example.json](configs/sources.example.json) as a source-policy template, replacing every placeholder. The example assigns prose slices as out-of-mix; choose the actual excluded sources before freezing.

Candidate JSONL rows:

```json
{"id":"source/document-id","source":"pl_web-source-id","slice":"pl_web","text":"Original complete document text."}
```

Training JSONL rows require `id`, `source`, and `text`. Pass **all documents in every candidate training pool**, before mixing or sampling. File hashes and the total number of screened training documents are recorded. The tool cannot verify that omitted training files exist elsewhere.

```sh
.venv/bin/ladder freeze-val \
  --sources configs/sources.json \
  --train /data/train/pool-000.jsonl /data/train/pool-001.jsonl \
  --candidates /data/validation-candidates.jsonl \
  --reference-tokenizer /models/frozen-reference-tokenizer \
  --output data/val-v1
```

This writes `documents.jsonl` and `manifest.json`, with **150k reference tokens per slice**, and nested 100k-token subsets from each of the two specified micro slices. Deduplication examines complete documents before trimming selected validation prefixes. The full source-document hash is retained. Scoring checks the frozen content hash and refuses mutation. Commands refuse to overwrite existing outputs.

MinHash LSH is approximate: exact verification removes false positive matches but cannot recover missed candidates. The manifest states this limitation. It does not claim perfect contamination removal. Source IDs must represent real publisher/domain identities consistently across all input files.

## 2. Review BLiMP-PL agreement v0

A pinned [MultiBLiMP Polish source](https://huggingface.co/datasets/jumelet/multiblimp/tree/de923efa8d2483d6b13364ee68e65308e990a991/pol) has been downloaded locally and prepared:

- `data/multiblimp-pl-v0/candidates.jsonl`: **3,197 unique pairs**, from 3,272 source rows.
- `data/multiblimp-pl-v0/review-sample.jsonl`: **300 pairs**, 100 per subject–verb number/person/gender paradigm.
- `data/multiblimp-pl-v0/review.json`: pending native review; no approval is fabricated.

The data directory is ignored by Git. Recreate the pool from a local TSV with:

```sh
.venv/bin/ladder import-multiblimp --tsv data/raw/multiblimp-pol.tsv \
  --revision de923efa8d2483d6b13364ee68e65308e990a991 \
  --output data/multiblimp-pl-v0-new
```

A native reviewer records their identity, `native_polish: true`, all reviewed sample IDs, rejected IDs, and a paradigm-level `status: approved` after reviewing and accepting the pool. The candidate hash binds review to the data. Rejected pairs are excluded. The scorer refuses unapproved paradigms. Approval of a sample is not a guarantee that every unreviewed pair is clean.

Critical regions include preceding whitespace and every subword of the changed whitespace-delimited word. Imported regions are omitted when the controller occurs after the agreement target, since prefix-only scoring then lacks the relevant evidence. Sentence probability is still available. Region boundaries and attached punctuation are part of native review; token-boundary crossings fail explicitly.

## 3. Evaluate checkpoints

```sh
.venv/bin/ladder eval --model /models/8m/checkpoint-10000 \
  --corpus data/val-v1 \
  --pl-pairs data/multiblimp-pl-v0/candidates.jsonl \
  --review data/multiblimp-pl-v0/review.json \
  --tier micro --rung 8 --device cpu --batch-size 8 --context 512 \
  --output runs/8m/control/seed0/micro-step10000.json
```

For fast, supply reviewed 8k PL pairs and `--en-pairs` containing 4k selected EN pairs, in the same pair schema (`id`, `language`, `paradigm`, `axis`, `good`, `bad`, optional `region`). Selection is stable and stratified by paradigm. At 50M and above, also supply `--mc` with three tasks, each row having `id`, `task`, `prefix`, and `correct`. Include the continuation's leading space in `correct`. MC scoring evaluates only the gold continuation and never uses argmax as a decision metric.

`--allow-incomplete` enables engineering checks on partial inputs but always makes the report ineligible for decisions. The full tier deliberately reports missing EWoK, PolEval/KLEJ and generation adapters rather than claiming release coverage. Runtime estimates in the specification are targets, not measurements.

## 4. Calibrate and validate the ladder

[configs/experiment.example.json](configs/experiment.example.json) lists the six initial control runs (three seeds at 8M and three at 25M) and three checkpoint reports per run. Point it at actual **fast** reports and include hashes of non-seed training configurations. Explicitly select the same continuous metrics for every comparison; do not select accuracy.

```sh
.venv/bin/ladder collect --manifest configs/experiment.json --output runs/collected.json
.venv/bin/ladder calibrate --runs runs/collected.json --control-mix control --output runs/noise.json
```

With control seeds only, `sigma_seed` is available; SNR and composite membership remain unresolved until multiple mixes are present. Once candidate runs are added, calibration computes across-mix SD, SNR, separation greater than twice seed SD, and eligibility at SNR ≥ 1.5. This describes eligibility; it does not invent a weighted composite.

```sh
.venv/bin/ladder decide --runs runs/collected-mixes.json \
  --control-mix control --candidate-mix candidate --rung 25 \
  --pair-metrics pair/pl/SV-P/sentence_prob pair/pl/SV-G/sentence_prob \
  --output runs/decision.json

.venv/bin/ladder rank-agreement --runs runs/all-rungs-mixes.json \
  --output runs/rank-agreement.json
```

All analysis uses each seed's mean over its last three distinct checkpoint steps. `decide` requires three seeds per compared mix, >2σ improvement in pooled out-of-mix BPB, and no regression on any explicitly selected pair probability. Select the admitted pair metrics before observing candidate results.

Rank agreement requires the same 4–6 mixes at 8M, 25M, and 120M. Publish per-metric ρ even when it is zero, negative, or undefined because of ties. Four to six mixes provide limited statistical resolution; there is no automatic “high enough” cutoff.

See [the protocol](docs/protocol.md) for the metric, cadence and scale-gating tables, and [configs/ladder.json](configs/ladder.json) for the machine-readable specification.
