> Historical synthetic scaffold; not the decision benchmark.

# Tiny LLM benchmark ladder

A runnable evaluation scaffold for an 8M → 16M → 32M → 64M → 128M → 256M model family. It generates deterministic, verified synthetic probes, scores answer continuations through `lm-evaluation-harness`, and reports accuracy, uncertainty, losses, and paired checkpoint changes.

**This is an uncalibrated first version.** No model-size-to-tier relationship has been measured. The included chance and oracle runs validate the machinery; they are not model results. English, zero-shot base-model completion is the initial protocol.

## Run it

The synthetic suite and controls require Python 3.11+ and no dependencies:

```sh
python3 -m ladder.cli generate --output runs/suite.json
python3 -m ladder.cli run --backend chance --output runs/chance.json
python3 -m ladder.cli run --backend oracle --output runs/oracle.json
python3 -m ladder.cli compare runs/chance.json runs/oracle.json --output runs/comparison.json
python3 -m unittest discover -s tests -v
```

Each run writes machine-readable JSON plus a readable Markdown report. JSON retains item-level scores and a hash of the entire generated suite. Generate the suite with identical arguments to inspect prompts, answers, and verification evidence.

For a Hugging Face causal LM checkpoint:

```sh
uv venv --python 3.12
uv sync --extra legacy-hf
# Alternatively: uv pip install -e '.[legacy-hf]'
.venv/bin/ladder-synthetic run --backend hf --model /absolute/path/to/checkpoint \
  --device cpu --batch-size 8 --parameters 8000000 --training-tokens 160000000 \
  --output runs/8m-step10000.json
```

Use `--device cuda:0` or `--device mps` when supported by your environment. Remote HF model IDs are accepted; pin `--revision` to a commit for reproducibility. The adapter records dependency versions, actual model parameter count, tokenizer hash, and the resolved model revision when available. Local checkpoints should be immutable and recorded in your experiment manifest; their weights are not hashed automatically. Remote custom code is disabled.

Compare checkpoints evaluated on the **same suite and protocol**:

```sh
.venv/bin/ladder-synthetic compare runs/8m-step10000.json runs/16m-step10000.json \
  --output runs/8m-to-16m.json
```

The comparison reports per-cell accuracy changes, paired bootstrap intervals, changed item counts, and choice-loss changes. It does not fit scaling laws or infer phase transitions from a single contrast.

## Initial tasks

| Tier | Family | Controllable difficulty | Interpretation |
|---|---|---|---|
| 0 | Copy | Sequence length | Exact copying preference |
| 0 | Progression | Sequence length and step size | Simple numeric pattern completion |
| A | Lookup | Number of key/value facts | Contextual retrieval; not world knowledge |
| B | Arithmetic | Number of addition terms | Short arithmetic |
| B | Sorting | List length | Ordering with near-miss distractors |
| C | Multihop | Number of links to follow | Compositional retrieval |
| D | State tracking | Number of register operations | Static execution proxy; not agent behavior |

The default tiny suite has **672 items**: seven families × three levels × 32 examples, or 2,688 candidate likelihood requests. The larger profile is `--count 256 --levels 1 2 3 4 5 6` (10,752 items). Levels 1–12 are supported; this version is deliberately finite. Difficulty need not increase monotonically, especially when later `set` operations erase earlier state.

All items have four unique candidates, an executable answer check, balanced answer positions per cell, and duplicate rejection within a suite. Candidate texts are scored directly after the prompt; option letters and candidates are not listed in the prompt. The primary score is summed continuation log probability, with per-UTF-8-byte normalized accuracy as a separately named diagnostic. A leading space is included consistently. Model tokenization and boundary handling are delegated to the harness; contexts exceeding the model limit produce an error instead of silent truncation.

## Reading the results

- **Accuracy + 95% Wilson interval:** per family and difficulty. Random baseline is 25%. Ties use deterministic item-seeded random selection; tie rate is visible.
- **Choice NLL:** negative log probability of the correct candidate after renormalizing the four sequence scores. Uniform baseline is `ln(4) ≈ 1.386` nats. This is not next-token language-model loss or perplexity.
- **Gold completion NLL / bits per byte:** probability assigned to the correct continuation, independent of candidate renormalization. These are task-completion diagnostics, not held-out corpus BPB.
- **Tier accuracy:** equal-weight average of its cells. There is no universal overall score.
- **Highest cleared tier:** all cells in that and every earlier tier have a lower 95% accuracy bound at or above `--threshold` (default 0.75). All tiers are evaluated regardless of clearance. This is a conservative descriptive rule, not a familywise statistical guarantee.
- **Signal flags:** chance-compatible, above-chance, below-chance, or ceiling. Flags indicate item-level uncertainty, not measured discrimination between sizes.

## Turn this into a calibrated benchmark

See [the experiment protocol](docs/protocol.md) for natural-language anchors, split handling, calibration, compute accounting, and limitations. The first useful experiment is running your actual checkpoints with the same tokenizer, training mixture, prompt protocol, suite seed, and levels. Keep model-size labels out of the task assignment until neighboring checkpoints demonstrate reliable separation.

The project reuses the [LM Evaluation Harness likelihood interface](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md). [tinyBenchmarks](https://github.com/felipemaiapolo/tinyBenchmarks) is a candidate source of calibrated natural-task subsets; its published estimators should not be assumed accurate for a new 8M–256M population without validation.

An optional offline integration check uses a locally created random GPT-2 fixture:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python tests/hf_smoke.py
```

It exercises actual forward passes and context-overflow rejection; its random weights do not measure learned capability. `uv.lock` pins the dependency graph for reproducible installs with `uv sync --extra legacy-hf`.
