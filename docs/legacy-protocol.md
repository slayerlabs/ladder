# Calibration and experiment protocol

## What to keep from the proposed ladder

Keep progressive difficulty, per-axis results, short prompts, frequent checkpoint evaluation, and continuous loss diagnostics. Treat the proposed size bands as planning hypotheses. Architecture, tokenizer, training tokens, data mixture, language, and post-training can all change where a model starts succeeding.

Run every model across all selected cells. “Advance only after clearing a tier” is useful for an optional runtime budget policy, but conflicts with measuring the whole capability curve. This implementation evaluates all cells and derives clearance afterwards. If adaptive evaluation is added, mark skipped cells as missing rather than failed, and periodically run the full suite.

## Pilot matrix

Use the available 8M, 16M, 32M, 64M, 128M, and 256M checkpoints; do not train them merely to populate a chart. Record total and non-embedding parameters, training tokens, data mixture/version, language, seed, architecture, context, tokenizer, checkpoint hash, precision, and training/evaluation compute. The CLI accepts total parameters and training tokens and records backend details; the rest belongs in an experiment manifest.

Measure several checkpoints per size and, when affordable, independent training seeds. Neighboring sizes should share the evaluation items. Compare both matched training-token budgets and matched-compute budgets as separate experiments. A fixed data-to-parameter ratio is another experiment, not equivalent to either comparison.

Use the tiny profile to identify promising cells. Expand their sample sizes before concluding that a neighbor improved. With 32 items, intervals are wide; a point estimate near 75% does not demonstrate clearance. Shared templates also induce dependence that item-level Wilson and bootstrap intervals do not capture. Use template-held-out variants and training-seed replication before claiming phase transitions.

## Development, holdout, and contamination

`--split dev` and `--split eval` use separate deterministic seed namespaces. This is convenience, not a private test set: generators, templates, seeds, and answer checks are public. Independent seeds do not guarantee semantic disjointness across suites. Audit prompt/content overlaps before making a frozen release; never train on generated evaluation items or tune prompts against final evaluation results. For contamination-resistant testing, reserve templates and generator rules as well as seeds, and maintain a separately controlled final set.

The current suite rejects exact prompt-plus-candidate-set duplicates internally. It does not make different permutations of a reasoning problem independent, or eliminate semantic overlap across difficulty levels. Keep performance claims conditional on these templates until broader validation exists.

Before selecting a tiny subset, use a development pool of representative small models. Choose items or cells that distinguish adjacent checkpoints; evaluate that selection on held-out models and checkpoints. Freeze item IDs, generator version, protocol, and selection weights. Repeatedly choosing the best-looking tasks on the final test scores creates selection bias.

## Natural-language anchors to add

The initial implementation intentionally starts with verified synthetic tasks. It does **not** yet contain factual knowledge, corpus LM loss, full grammaticality evaluation, free-generation exact match, or interactive agents. Run established harness tasks alongside it, preserving their native protocols and result files:

| Axis | Candidate anchor | Metric / qualification |
|---|---|---|
| Language modeling | Licensed held-out text in the training language | Total negative log likelihood divided by total UTF-8 bytes and ln(2); disjoint from training |
| Syntax | BLiMP minimal pairs | Pair accuracy and log-likelihood margin; full sentence scoring, not an instruction test |
| Science / commonsense | SciQ, ARC-Easy, PIQA | Native multiple-choice scoring, candidate-count-specific chance baseline |
| Completion | HellaSwag, WinoGrande | Diagnostic until above chance; preserve native normalization and context construction |
| Reasoning | Controlled word problems, then GSM8K / ARC-Challenge | Separate candidate scoring from exact-answer generation |
| Stress | MMLU or other held-out hard tasks | Optional controls; no assumed 256M activation boundary |

Use installed `lm-eval` task discovery/help to confirm current task names and run the same version across the model family. Freeze dataset revisions and licensed snapshots; record split, source item IDs, and any filters. A filtered dataset must receive a new name and cannot claim the original benchmark's full score. “Easy” should be established on a development population or by controllable structure, not by filtering test items based on the target model's answers.

The [tinyBenchmarks paper](https://arxiv.org/abs/2402.14992) and [reference implementation](https://github.com/felipemaiapolo/tinyBenchmarks) describe reduced evaluation and estimators. A hundred selected examples are not automatically a representative unweighted subset. Use the associated estimator/weights and validate transfer to the small-model family before reporting estimated full-benchmark performance. No tinyBenchmarks estimator is included here.

## Prompting and interpretation

Version zero-shot, few-shot, and CoT as separate protocols. Tiny base models may fail to follow instructions while learning useful language statistics; hence the importance of corpus loss, minimal pairs, and natural continuation anchors. The first implementation uses English zero-shot completion only. Do not combine Polish and English into a single unqualified capability score.

Raw sequence scoring can prefer shorter answers. Byte-normalized scoring changes the decision rule and is only a diagnostic; it does not remove all tokenization and length biases. Candidate NLL depends on the exact distractors. Fit scaling curves to fixed prompts, candidates, and normalizations. The oracle verifies generator answers, not a measured human ceiling.

## Agentic extension

State tracking is a static proxy. A true agent tier needs a deterministic environment, a tool schema, action parsing, explicit tool/step budgets, execution traces, and terminal success checks. Start with one action, then sequential actions, branching, and recovery. Separate tool-format validity from task success. Evaluate agentic post-training separately from base-model capacity. This version neither executes tools nor awards agent capability from its D tier.

## Scaling laws

First collect genuine held-out losses and enough N/D variation to identify a model. A single fixed-ratio ladder does not independently identify parameter and data exponents. Fit a candidate relation such as `L(N,D) = E + A*N^(-alpha) + B*D^(-beta)` on development models with positive constraints; compare against simpler baselines and validate on withheld sizes. Task-choice loss need not obey this relation.

Then fit and validate the loss-to-accuracy mapping with sampling uncertainty and chance/ceiling behavior. Report out-of-sample prediction errors, confidence intervals, and the actual compute ledger. Neither one-percent-compute predictions nor phase transitions are guaranteed. A noisy accuracy jump alone is insufficient; inspect losses, intervals, prompt effects, multiple comparisons, and repeatability.
