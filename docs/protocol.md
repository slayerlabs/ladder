# Continuous ladder protocol

## Metrics: three continuous measurements

1. **Corpus BPB:** `sum(nll_nats) / (ln(2) * sum(original_utf8_bytes))`, separately for every slice and pooled for out-of-mix slices. Count actual scored bytes, not padded tokens or BOS markers. A frozen reference tokenizer defines dataset token budgets only. Evaluation uses each model's own tokenizer. The unit is tokenizer-independent; language, corpus composition, tokenization lossiness, and effective context still affect interpretation. PL and EN scores share a unit, not necessarily an intrinsic difficulty level.
2. **Pair probability:** `sigmoid(logp_good - logp_bad)`, then arithmetic mean over pairs. Full sentence and critical region are separate series (`sentence_prob`, `region_prob`) with region coverage counts. Store raw log probabilities, log margin and accuracy as diagnostics. Never use pair accuracy to decide a change. Ties contribute probability 0.5 and diagnostic accuracy 0.5, with no random tie-breaking.
3. **Correct-choice BPB:** summed NLL of the gold MC continuations divided by their summed UTF-8 bytes and `ln(2)`. Score all subwords, including leading whitespace. Do not renormalize over distractors or use the argmax as the primary measure.

The scorer uses fp32 weights/logits, eager attention, eval/inference mode, deterministic PyTorch algorithms, TF32 disabled, fixed padded batch shapes, and no generation. Float64 summation accumulates fp32 token NLLs. Every document starts with BOS (or EOS fallback); long documents use overlapping windows and score each target token exactly once. Record context, stride, batch shape, tokenizer hash, model revision, runtime versions and hardware. Exact repeatability was tested on the local CPU fixture, not across all hardware or software versions. See [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html).

Tokenization must round-trip exact text. Span scoring fails when a token crosses the declared region boundary. This prevents silently counting NLL for a larger region than its byte denominator. Boundary-failing examples need review or a frozen, explicitly versioned scoring policy; do not silently skip them.

## Frozen validation data

Eight slices, each 150k **reference** tokens:

| Language | Slices |
|---|---|
| PL | web; prose/books; wiki; dialogue/forum; technical/legal |
| EN | web; prose |
| Code | one slice |

The two out-of-mix slices must be sourced entirely outside all candidate training pools. Source identities are explicit, consistent publisher/domain IDs. The entire training pool is screened, not just the sampled training mix. Compare mixes primarily on these slices; fixed-data hyperparameter/architecture comparisons can also use in-distribution slices.

The builder indexes validation candidates with 128-permutation, fixed-seed MinHash over normalized 13-word grams, streams every training document, and verifies retrieved matches with exact Jaccard ≥0.8. It also excludes matching document IDs and screens validation duplicates before selecting budgets. Short documents use their whole normalized word sequence as a shingle. This is an approximate near-dedup pipeline: [MinHash LSH has false negatives](https://ekzhu.com/datasketch/lsh.html), even with exact verification of retrieved candidates. Production freeze should include a recall audit or stronger candidate retrieval if that residual contamination risk is unacceptable.

The builder records train-file hashes, candidate hash, full-document hashes, reference tokenizer hash, sources and exclusions. It refuses to overwrite a frozen version; scoring verifies the content hash. Filesystem immutability or signed manifests remain an operational responsibility. Future mixes must respect excluded sources and rerun contamination screening against newly added training pools.

## Pair data and admission

PL v0 is the pinned MultiBLiMP Polish agreement pool. Its 3,197 unique pairs span subject–verb number, person and gender; all require native review before admission. It is insufficient for the complete fast tier's 8k PL pairs. [MultiBLiMP's reference repository](https://github.com/jumelet/multiblimp) supplies provenance; a newly selected/filtered pool is not the original benchmark score.

PL v1 target: 25 paradigms ×1k, generated from PL-UD and Morfeusz2, with at least 100 pairs reviewed by a native speaker per paradigm. Priority: prepositional case, genitive of negation, masculine-personal plural, adjective–noun agreement, aspect/tense, verb government, numeral constructions. These generators are not implemented yet. Review must reject ambiguous grammatical alternatives and false violations; otherwise apparent plateaus can be data artifacts.

EN full target: 67 BLiMP paradigms ×128, stratified and frozen. Fast chooses a stable stratified 4k subset. No EN data has been imported yet. Preserve provenance and original item IDs; label subset results separately from published full-benchmark accuracy.

Critical regions use a shared prefix and include preceding whitespace. MultiBLiMP imports additionally require the agreement controller to precede the changed target; otherwise region probability is unavailable and sentence probability remains valid. The current importer selects the changed whitespace-delimited word, so attached punctuation must be inspected during review.

## Cadence

| Tier | Contents | Target cost | When | Decisions |
|---|---|---|---|---|
| micro | 200k reference val tokens from two frozen slices; 2k PL agreement pairs | <1 min | every checkpoint | never |
| fast | all eight slices /1.2M reference tokens; 8k PL +4k EN pairs; three MC tasks where enabled | 5–10 min at 120M | end of every run, every seed | yes, if complete |
| full | fast plus all PL paradigms, EN 67×128, EWoK, PolEval/KLEJ, generation sanity | ~1h | release only | release inspection; dev decisions still use fast |

Costs are targets awaiting measurement. MC execution starts at 50M in this build order; its lower-rung policy remains a prior. Partial runs are engineering diagnostics, never decision eligible. Full-tier adapters are outstanding and the CLI refuses to claim full coverage.

## Scale gating: prior, not results

| Measurement | 8M | 25M | 50M | 120M |
|---|---|---|---|---|
| Corpus BPB, every slice | primary | primary | primary | primary |
| Agreement pair probability | active | active | active | active |
| Case/negation/gender pair probability | floor | active | active | active |
| Long-distance/island pair probability | floor | floor | weak | active |
| MC correct-choice BPB | floor | weak | active | active |
| MC accuracy | off | off | off | reportable diagnostic |
| EWoK/knowledge | off | off | off | canary only |

The agreement row includes the initial subject–verb gender agreement subset. The case/negation/gender row refers to the expanded morphosyntactic probes. Assign every new paradigm's axis explicitly rather than inferring its gate from its name.

Admission needs an observed same-rung contrast exceeding **2σ_seed**. Re-evaluate whenever a rung is added. The CLI emits the prior policy alongside measured results; calibration emits evidence-based eligibility separately. Floor/weak labels do not prove that actual scores are at chance. No EWoK canary contributes to a dev decision.

## Noise and adoption

Before mix search, run three seeds at 8M and three at 25M with identical data/hyperparameters apart from training seed. Use each seed's mean over its last three checkpoint scores. Compute sample SD across independent seeds; do not treat three checkpoints from one run as three seeds.

Across candidate mixes, compute SD of mix means divided by seed SD. Exclude SNR <1.5 from the development composite; a benchmark also needs a >2σ contrast for admission. With only a control mix, SNR is unresolved. Zero observed SD is flagged, not interpreted as proof of noiseless evaluation. The implementation returns eligible metric IDs but does not invent composite weights or normalize incomparable units into a scalar.

Recalibrate at 120M. Smaller seed variance at larger scale is a hypothesis to check, not an automatic reason to reuse a supposedly conservative 25M estimate.

The decision rule uses pooled out-of-mix BPB: adopt only if improvement exceeds 2σ_seed and **none** of the preselected admitted pair probabilities regress. `decide` requires three seeds for both control and candidate and only accepts complete fast runs with matching suite/protocol hashes. This implements the specified heuristic; 2σ_seed is not a formal confidence interval for a difference of means or a multiple-comparison correction.

## What makes it a ladder

Run the same four to six mixes at 8M, 25M and 120M. Average checkpoint means across training seeds. Compute Spearman ρ per continuous metric for 8M→120M and 25M→120M. Ties use average ranks; a constant ranking gives undefined ρ. The implementation enumerates all 24–720 permutations for a two-sided permutation p-value, with no Monte Carlo sampling.

Report every metric's ρ and the underlying mix means, even if correlation is zero or negative. There is no preselected “high ρ” cutoff or guaranteed extrapolation beyond the tested mixes. With only four to six mixes, statistical resolution is limited. A rank check on this experiment does not establish universal small-to-large transfer, and no unverified publication-novelty claim is made.

## Remaining external work

Provide full train-pool and validation-source files plus the reference tokenizer; freeze and inspect real slices. Complete native review of the prepared PL v0 pool. Supply the six initial control training runs and complete fast-tier pair coverage. The software is tested, but these data collection, human review and GPU experiments have not occurred in this session.
