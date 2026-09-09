# Provisional Polish micro comparison

This folder publishes actual model outputs, not timing projections or synthetic oracle controls.

The top-level README contains the overall results. Differences below are Pollock minus Koliber; they are descriptive and are not divided by unmeasured seed noise.

| Paradigm | Pairs | Koliber sentence probability | Pollock sentence probability | Difference |
|---|---:|---:|---:|---:|
| Subject–verb number | 667 | 0.8948 | 0.5235 | -0.3714 |
| Subject–verb gender | 667 | 0.8828 | 0.5074 | -0.3754 |
| Subject–verb person | 666 | 0.9473 | 0.7436 | -0.2038 |

## Evidence

- `koliber.json` and `pollock.json`: complete reports including item IDs, probabilities, raw log probabilities, metadata, missing coverage and measured times.
- `comparison.json`: aggregate metrics and paired differences; `sha256.json`: hashes of the three evidence files.
- Identical suite and protocol hashes, identical ordered pair IDs and matching critical-region availability were checked before publication.
- Models use their own tokenizers and document-start tokens; those are recorded in each report. No shared tokenizer is substituted.
- The Koliber custom adapter validates source/config hashes at a pinned revision and forces math SDPA. Standard GPT-2 Pollock uses the same math backend.
- CPU accumulation follows MPS-to-CPU transfer before float64 conversion; forward computation stays fp32.

## Reproduce

From the repository root, prepare the pinned public source and deterministic review pool. Native review remains pending; the flags below explicitly enable provisional diagnostics only.

```sh
uv sync --extra dev
mkdir -p data/raw
curl -fL https://huggingface.co/datasets/jumelet/multiblimp/resolve/de923efa8d2483d6b13364ee68e65308e990a991/pol/data.tsv -o data/raw/multiblimp-pol.tsv
.venv/bin/ladder import-multiblimp --tsv data/raw/multiblimp-pol.tsv --revision de923efa8d2483d6b13364ee68e65308e990a991 --output data/multiblimp-pl-v0
```

If the prepared pool already exists, retain it and verify its hashes rather than overwriting it.

```sh
.venv/bin/ladder eval --model OrisTeam/Koliber-v1.1-Base-Preview --revision 10bfff5fec23a43cd798b80645f95673a234f39a --adapter koliber --attention sdpa_math \
  --pl-pairs data/multiblimp-pl-v0/candidates.jsonl --review data/multiblimp-pl-v0/review.json \
  --tier micro --rung 120 --device mps --batch-size 8 --context 512 \
  --allow-incomplete --allow-unreviewed --output runs/koliber-micro-provisional.json
```

```sh
.venv/bin/ladder eval --model SlayerLab/pollock-mini-lm-125m --revision 0d22afece64fc5a28f1a32e3eac7a14bc563e089 --attention sdpa_math \
  --pl-pairs data/multiblimp-pl-v0/candidates.jsonl --review data/multiblimp-pl-v0/review.json \
  --tier micro --rung 120 --device mps --batch-size 8 --context 512 \
  --allow-incomplete --allow-unreviewed --output runs/pollock-micro-provisional.json
```

Both runs use reporting rung 120; actual parameter counts remain explicit. Commands refuse existing outputs.

No BPB, native validation, seed calibration, or training-overlap assessment is supplied by this experiment. A matched English probe is needed before drawing conclusions about Pollock’s intended language domain.
