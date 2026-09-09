# Benchmark ladder: chance

Highest cleared tier: **none**

Control run; not model capability evidence.

| Cell | n | Accuracy (95% CI) | Choice NLL | Signal |
|---|---:|---:|---:|---|
| 0/copy/d1 | 32 | 18.8% (8.9%–35.3%) | 1.386 | chance-compatible |
| 0/copy/d2 | 32 | 21.9% (11.0%–38.8%) | 1.386 | chance-compatible |
| 0/copy/d3 | 32 | 28.1% (15.6%–45.4%) | 1.386 | chance-compatible |
| 0/progression/d1 | 32 | 25.0% (13.3%–42.1%) | 1.386 | chance-compatible |
| 0/progression/d2 | 32 | 15.6% (6.9%–31.8%) | 1.386 | chance-compatible |
| 0/progression/d3 | 32 | 21.9% (11.0%–38.8%) | 1.386 | chance-compatible |
| A/lookup/d1 | 32 | 25.0% (13.3%–42.1%) | 1.386 | chance-compatible |
| A/lookup/d2 | 32 | 25.0% (13.3%–42.1%) | 1.386 | chance-compatible |
| A/lookup/d3 | 32 | 15.6% (6.9%–31.8%) | 1.386 | chance-compatible |
| B/arithmetic/d1 | 32 | 28.1% (15.6%–45.4%) | 1.386 | chance-compatible |
| B/arithmetic/d2 | 32 | 25.0% (13.3%–42.1%) | 1.386 | chance-compatible |
| B/arithmetic/d3 | 32 | 25.0% (13.3%–42.1%) | 1.386 | chance-compatible |
| B/sorting/d1 | 32 | 28.1% (15.6%–45.4%) | 1.386 | chance-compatible |
| B/sorting/d2 | 32 | 18.8% (8.9%–35.3%) | 1.386 | chance-compatible |
| B/sorting/d3 | 32 | 18.8% (8.9%–35.3%) | 1.386 | chance-compatible |
| C/multihop/d1 | 32 | 21.9% (11.0%–38.8%) | 1.386 | chance-compatible |
| C/multihop/d2 | 32 | 18.8% (8.9%–35.3%) | 1.386 | chance-compatible |
| C/multihop/d3 | 32 | 34.4% (20.4%–51.7%) | 1.386 | chance-compatible |
| D/state_tracking/d1 | 32 | 31.2% (18.0%–48.6%) | 1.386 | chance-compatible |
| D/state_tracking/d2 | 32 | 18.8% (8.9%–35.3%) | 1.386 | chance-compatible |
| D/state_tracking/d3 | 32 | 12.5% (5.0%–28.1%) | 1.386 | chance-compatible |

Random baseline: 25%; uniform-choice NLL: 1.386 nats.

Every cell in this and all earlier tiers has Wilson lower 95% bound >= threshold

- Uncalibrated synthetic probes; no size-to-tier guarantee.
- D is static state tracking, not interactive agent capability.
- Choice NLL is conditional on four candidates, not corpus LM loss.
- Wilson intervals are item-level, not adjusted for multiple comparisons or template dependence.
