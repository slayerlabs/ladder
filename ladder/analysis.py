"""Seed calibration and cross-rung rank validity. No sampling or synthetic evidence."""
import itertools
import math
import statistics
from collections import defaultdict


def summarize_runs(runs):
    if not runs:
        raise ValueError("No runs")
    signatures = {(r["suite_sha256"], r["protocol_sha256"]) for r in runs}
    if len(signatures) != 1:
        raise ValueError("All runs must share a frozen suite and scoring protocol")
    seen, result = set(), []
    configurations = defaultdict(set)
    for run in runs:
        key = (run["rung"], run["mix"], run["seed"])
        if key in seen:
            raise ValueError("Duplicate rung/mix/seed")
        seen.add(key)
        configurations[(run["rung"], run["mix"])].add(run["training_config_sha256"])
        if run["tier"] != "fast" or not run["decision_eligible"]:
            raise ValueError("Decisions require complete fast-tier evaluations")
        checkpoints = sorted(run["checkpoints"], key=lambda c: c["step"])
        if len(checkpoints) < 3 or len({c["step"] for c in checkpoints}) != len(checkpoints):
            raise ValueError("At least three distinct checkpoint steps are required")
        last = checkpoints[-3:]
        keys = set(last[0]["metrics"])
        if any(set(c["metrics"]) != keys for c in last):
            raise ValueError("Checkpoint metric coverage differs")
        metrics = {}
        for metric in sorted(keys):
            values = [c["metrics"][metric] for c in last]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                raise ValueError("Nonfinite metric")
            metrics[metric] = statistics.mean(values)
        result.append({**{k: run[k] for k in ("rung", "mix", "seed")},
                       "metrics": metrics, "steps": [c["step"] for c in last]})
    if any(len(v) != 1 for v in configurations.values()):
        raise ValueError("Training settings differ across seeds of the same rung/mix")
    if any(set(r["metrics"]) != set(result[0]["metrics"]) for r in result):
        raise ValueError("Run metric coverage differs")
    return result


def calibrate(runs, control_mix):
    rows = summarize_runs(runs)
    output = {}
    for rung in sorted({r["rung"] for r in rows}):
        group = [r for r in rows if r["rung"] == rung]
        control = [r for r in group if r["mix"] == control_mix]
        if len(control) < 3:
            raise ValueError(f"Need at least three control training seeds at {rung}M")
        # config_id attests identical non-seed training settings for each mix/rung.
        originals = [r for r in runs if r["rung"] == rung and r["mix"] == control_mix]
        if len({r["training_config_sha256"] for r in originals}) != 1:
            raise ValueError("Control seeds differ in training configuration")
        metrics = {}
        for metric in control[0]["metrics"]:
            sigma = statistics.stdev(r["metrics"][metric] for r in control)
            by_mix = defaultdict(list)
            for row in group:
                by_mix[row["mix"]].append(row["metrics"][metric])
            means = {mix: statistics.mean(v) for mix, v in by_mix.items()}
            across = statistics.stdev(means.values()) if len(means) >= 2 else None
            snr = across / sigma if sigma > 0 and across is not None else None
            spread = max(means.values()) - min(means.values())
            separated = len(means) >= 2 and spread > 2 * sigma
            keep = None if across is None else separated and (snr >= 1.5 if snr is not None else across > 0)
            metrics[metric] = {"sigma_seed": sigma, "control_seeds": len(control),
                               "sigma_across_mix_means": across, "snr": snr,
                               "zero_seed_variance": sigma == 0, "mix_means": means,
                               "seed_counts": {k: len(v) for k, v in by_mix.items()},
                               "separation_gt_2sigma": separated, "dev_composite_eligible": keep}
        output[str(rung)] = metrics
    return {"control_mix": control_mix, "initial_8m_25m_calibration_present": {"8", "25"} <= set(output), "rungs": output,
            "note": "SNR uses mix means; limited seeds and noisy mix means affect its estimate. Zero observed seed variance is not proof of zero noise. Admission must be rechecked at each rung."}


def ranks(values):
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result = [0.] * len(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[ordered[start]] == values[ordered[end]]:
            end += 1
        for position in ordered[start:end]:
            result[position] = (start + 1 + end) / 2
        start = end
    return result


def spearman(a, b):
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("Matched rankings required")
    x, y = ranks(a), ranks(b)
    if len(set(x)) == 1 or len(set(y)) == 1:
        return None
    return statistics.correlation(x, y)


def rank_agreement(runs):
    rows = summarize_runs(runs)
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["rung"]][row["mix"]].append(row)
    if not {8, 25, 120} <= set(grouped):
        raise ValueError("Rank validity requires 8M, 25M and 120M")
    mixes = sorted(grouped[120])
    if not 4 <= len(mixes) <= 6 or any(set(grouped[r]) != set(mixes) for r in (8, 25)):
        raise ValueError("Use the same four to six candidate mixes at every rung")
    output = {}
    for metric in rows[0]["metrics"]:
        values = {rung: [statistics.mean(row["metrics"][metric] for row in grouped[rung][mix])
                         for mix in mixes] for rung in (8, 25, 120)}
        output[metric] = {}
        for rung in (8, 25):
            rho = spearman(values[rung], values[120])
            permutations = list(itertools.permutations(values[120])) if rho is not None else []
            extreme = sum(abs(spearman(values[rung], list(p))) >= abs(rho) - 1e-12 for p in permutations)
            output[metric][str(rung)] = {"rho": rho,
                "exact_two_sided_permutation_p": extreme / len(permutations) if permutations else None,
                "mixes": mixes, "small_values": values[rung], "reference_values": values[120],
                "validity": "undefined: tied ranking" if rho is None else "report rho; no automatic validity threshold"}
    return {"reference_rung": 120, "metrics": output,
            "seed_counts": {str(r): {m: len(v) for m, v in grouped[r].items()} for r in (8, 25, 120)},
            "note": "Only 4–6 mixes; publish all per-metric correlations. High rho is evidence for these mixes, not universal transfer. No claim of publication novelty is made."}


def decision(runs, control_mix, candidate_mix, rung, pair_metrics):
    calibration = calibrate(runs, control_mix)
    rows = [r for r in summarize_runs(runs) if r["rung"] == rung]
    if not pair_metrics or any(not key.startswith("pair/") for key in pair_metrics):
        raise ValueError("Explicit continuous pair-probability metric IDs required")
    metrics = ["bpb/out_of_mix", *pair_metrics]
    if any(len([r for r in rows if r["mix"] == mix]) < 3 for mix in (control_mix, candidate_mix)):
        raise ValueError("Decision needs at least three seeds for each compared mix")
    deltas = {}
    for metric in metrics:
        try:
            base = statistics.mean(r["metrics"][metric] for r in rows if r["mix"] == control_mix)
            new = statistics.mean(r["metrics"][metric] for r in rows if r["mix"] == candidate_mix)
            sigma = calibration["rungs"][str(rung)][metric]["sigma_seed"]
        except KeyError as exc:
            raise ValueError(f"Missing decision metric {metric}") from exc
        deltas[metric] = {"improvement": base - new if metric.startswith("bpb/") else new - base,
                          "sigma_seed": sigma}
    adopt = deltas[metrics[0]]["improvement"] > 2 * deltas[metrics[0]]["sigma_seed"] and all(
        deltas[key]["improvement"] >= 0 for key in pair_metrics)
    return {"rung": rung, "control": control_mix, "candidate": candidate_mix,
            "decision": "adopt" if adopt else "investigate", "metrics": deltas,
            "rule": "Out-of-mix BPB improves >2 seed SD; none of the selected pair probabilities regress."}
