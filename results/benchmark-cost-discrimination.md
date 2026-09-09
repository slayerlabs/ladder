# Benchmark cost versus discrimination

This is a planning comparison for the next paired Koliber/Pollock evaluation. “Forward-request proxy” counts approximate candidate likelihood evaluations: two variants for BLiMP, one continuation for LAMBADA, and four answer choices for multiple-choice tasks. It is useful for relative cost, not a runtime guarantee; sequence lengths, padding, batching, backend and precision change elapsed time.

The only measured discrimination below is the provisional Polish micro result. For a benchmark to receive a numeric discrimination factor, evaluate both models (and preferably multiple training seeds) on identical frozen items. Define the seed-calibrated factor as `DF = |mean(model A − model B)| / sigma_seed`, where the numerator uses the benchmark’s continuous primary metric and `sigma_seed` is the seed standard deviation of the model difference. A single-model score, accuracy gap, or sample count is not a DF.

| Benchmark | Primary metric | Samples | Forward-request proxy | Time on this Mac | Current discrimination evidence | Use |
|---|---|---:|---:|---:|---|---|
| Polish agreement micro | pair probability | 2,000 (1,171 regions) | 6,342 | Koliber 111.3 s; Pollock 100.4 s | observed Δ = 0.3169; DF unavailable (no seed σ) | smoke alarm; provisional |
| BLiMP | accuracy in Pollock reference | 67,000 | ~134,000 | not measured; roughly 35–45 min proxy | none; Koliber run required | high sample cost, syntax signal |
| LAMBADA OpenAI | accuracy | 5,153 | ~5,153 | not measured; roughly 1–3 min proxy | none; Koliber run required | cheap completion probe |
| HellaSwag | `acc_norm` | 10,042 | ~40,168 | not measured; roughly 7–12 min proxy | none; Koliber run required | MC continuation; check floor effects |
| PIQA | `acc_norm` | 1,838 | ~3,676 | not measured; roughly 1–2 min proxy | none; Koliber run required | cheap commonsense probe |
| SciQ | `acc_norm` | 1,000 | ~4,000 | not measured; roughly 1–2 min proxy | none; Koliber run required | small science probe |
| ARC-Easy | `acc_norm` | 2,376 | ~9,504 | not measured; roughly 2–4 min proxy | none; Koliber run required | likely above-floor only |
| ARC-Challenge | `acc_norm` | 1,172 | ~4,688 | not measured; roughly 1–3 min proxy | none; Koliber run required | hard canary; floor risk |

Pollock’s official artifact reports the seven English tasks on complete splits using `lm-evaluation-harness 0.4.12`, zero-shot, batch size 8, BF16 and maximum length 1024. Those are reference results, not paired discrimination measurements. See [`benchmarks/english.json`](https://huggingface.co/SlayerLab/pollock-mini-lm-125m/blob/main/benchmarks/english.json).

The time proxies scale the measured 2,000-pair MPS run by forward-request proxy. They assume the same M4 Max GPU, batch shape, model size and implementation. The current pair run took 111.3 seconds for Koliber and 100.4 seconds for Pollock; model loading added about 1–3 seconds. The official harness may differ because its padding and request ordering differ.

For the ladder, rank benchmarks by measured DF after three-seed calibration, not by Pollock’s score. Retain metrics with DF ≥ 2 for an adjacent-rung contrast; prune development-composite metrics with SNR < 1.5. Keep BPB and pair probability continuous even when accuracy is saturated or at chance. Do not claim that a benchmark discriminates Koliber from another model until Koliber is evaluated under the same protocol.

The complete English battery is approximately 201,189 candidate likelihoods by this proxy, roughly 32 times the current 6,342-request pair micro workload. The Polish micro result remains provisional because the 200k-token corpus is absent, native pair review is pending, and no training-seed variance has been measured.
