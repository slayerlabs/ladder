# Benchmark ladder: oracle

Highest cleared tier: **D**

Control run; not model capability evidence.

| Cell | n | Accuracy (95% CI) | Choice NLL | Signal |
|---|---:|---:|---:|---|
| 0/copy/d1 | 32 | 100.0% (89.3%–100.0%) | 0.000 | above-chance |
| 0/copy/d2 | 32 | 100.0% (89.3%–100.0%) | 0.000 | above-chance |
| 0/copy/d3 | 32 | 100.0% (89.3%–100.0%) | 0.000 | above-chance |
| 0/progression/d1 | 32 | 100.0% (89.3%–100.0%) | 0.000 | above-chance |
| 0/progression/d2 | 32 | 100.0% (89.3%–100.0%) | 0.000 | above-chance |
| 0/progression/d3 | 32 | 100.0% (89.3%–100.0%) | 0.000 | above-chance |
| A/lookup/d1 | 32 | 100.0% (89.3%–100.0%) | 0.000 | above-chance |
| A/lookup/d2 | 32 | 100.0% (89.3%–100.0%) | 0.000 | above-chance |
| A/lookup/d3 | 32 | 100.0% (89.3%–100.0%) | 0.000 | above-chance |
| B/arithmetic/d1 | 32 | 100.0% (89.3%–100.0%) | 0.000 | above-chance |
| B/arithmetic/d2 | 32 | 100.0% (89.3%–100.0%) | 0.000 | above-chance |
| B/arithmetic/d3 | 32 | 100.0% (89.3%–100.0%) | 0.000 | above-chance |
| B/sorting/d1 | 32 | 100.0% (89.3%–100.0%) | 0.000 | above-chance |
| B/sorting/d2 | 32 | 100.0% (89.3%–100.0%) | 0.000 | above-chance |
| B/sorting/d3 | 32 | 100.0% (89.3%–100.0%) | 0.000 | above-chance |
| C/multihop/d1 | 32 | 100.0% (89.3%–100.0%) | 0.000 | above-chance |
| C/multihop/d2 | 32 | 100.0% (89.3%–100.0%) | 0.000 | above-chance |
| C/multihop/d3 | 32 | 100.0% (89.3%–100.0%) | 0.000 | above-chance |
| D/state_tracking/d1 | 32 | 100.0% (89.3%–100.0%) | 0.000 | above-chance |
| D/state_tracking/d2 | 32 | 100.0% (89.3%–100.0%) | 0.000 | above-chance |
| D/state_tracking/d3 | 32 | 100.0% (89.3%–100.0%) | 0.000 | above-chance |

Random baseline: 25%; uniform-choice NLL: 1.386 nats.

Every cell in this and all earlier tiers has Wilson lower 95% bound >= threshold

- Uncalibrated synthetic probes; no size-to-tier guarantee.
- D is static state tracking, not interactive agent capability.
- Choice NLL is conditional on four candidates, not corpus LM loss.
- Wilson intervals are item-level, not adjusted for multiple comparisons or template dependence.
