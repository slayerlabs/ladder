import importlib.metadata
import math
import random
import statistics
from collections import defaultdict

from .tasks import TIERS, digest, seed_for
from .gating import resolve_rung, scale_policy

PROTOCOL = {"version": 1, "prompting": "base-completion-zero-shot", "continuation_prefix": " ",
            "primary_score": "sum_logprob", "secondary_score": "logprob_per_utf8_byte",
            "ties": "item-seeded-uniform", "context_overflow": "error"}


def wilson(successes, n):
    z = 1.959963984540054
    p = successes / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0, center - half), min(1, center + half)]


def winner(scores, item_id):
    best = max(scores)
    ties = [i for i, v in enumerate(scores) if v == best]
    return random.Random(seed_for("tie", item_id)).choice(ties), len(ties) > 1


def score_item(item, scores):
    if len(scores) != len(item.choices) or not all(math.isfinite(s) and s <= 0 for s in scores):
        raise ValueError(f"Expected four finite, nonpositive log probabilities: {item.id}")
    lengths = [len((" " + c).encode("utf-8")) for c in item.choices]
    normalized = [s / n for s, n in zip(scores, lengths)]
    pred, tied = winner(scores, item.id)
    norm_pred, _ = winner(normalized, item.id)
    peak = max(scores)
    choice_nll = peak + math.log(sum(math.exp(s - peak) for s in scores)) - scores[item.answer]
    return {"id": item.id, "family": item.family, "tier": item.tier, "difficulty": item.difficulty,
            "gold": item.answer, "prediction": pred, "correct": int(pred == item.answer),
            "byte_normalized_correct": int(norm_pred == item.answer), "tied": tied,
            "logprobs": scores, "choice_nll": choice_nll,
            "gold_completion_nll": -scores[item.answer],
            "gold_completion_bytes": lengths[item.answer],
            "gold_completion_bits_per_byte": -normalized[item.answer] / math.log(2)}


def aggregate(rows):
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    acc = correct / n
    ci = wilson(correct, n)
    return {"n": n, "accuracy": acc, "accuracy_ci95": ci, "chance": 0.25,
            "chance_adjusted_accuracy": (acc - 0.25) / 0.75,
            "byte_normalized_accuracy": statistics.mean(r["byte_normalized_correct"] for r in rows),
            "choice_nll": statistics.mean(r["choice_nll"] for r in rows),
            "gold_completion_nll": statistics.mean(r["gold_completion_nll"] for r in rows),
            "gold_completion_bits_per_byte": statistics.mean(r["gold_completion_bits_per_byte"] for r in rows),
            "mc_correct_choice_bpb": sum(r["gold_completion_nll"] for r in rows) /
                (math.log(2) * sum(r["gold_completion_bytes"] for r in rows)),
            "tie_rate": statistics.mean(r["tied"] for r in rows),
            "signal": "ceiling" if ci[0] >= 0.95 else "chance-compatible" if ci[0] <= 0.25 <= ci[1]
                      else "below-chance" if ci[1] < 0.25 else "above-chance"}


def summarize(items, rows, model, threshold=0.75):
    if not 0.25 < threshold < 1:
        raise ValueError("threshold must be between chance (0.25) and 1")
    if not items or [i.id for i in items] != [r["id"] for r in rows]:
        raise ValueError("Results must match the complete ordered suite")
    groups = defaultdict(list)
    for row in rows:
        groups[f'{row["tier"]}/{row["family"]}/d{row["difficulty"]}'].append(row)
    cells = {key: aggregate(values) for key, values in groups.items()}
    tiers = {}
    highest = None
    contiguous = True
    for tier in TIERS:
        values = [v for k, v in cells.items() if k.startswith(tier + "/")]
        cleared = bool(values) and all(v["accuracy_ci95"][0] >= threshold for v in values)
        tiers[tier] = {"macro_accuracy": statistics.mean(v["accuracy"] for v in values) if values else None,
                       "cleared": cleared}
        contiguous = contiguous and cleared
        if contiguous:
            highest = tier
    rung = resolve_rung(model.get("parameters"), model.get("rung_millions"))
    policy = scale_policy(rung) if rung is not None else None
    accuracy_reportable = policy is None or policy["metrics"]["mc_accuracy"]["status"] == "reportable"
    return {"schema_version": 2, "suite_sha256": digest(items), "protocol": PROTOCOL,
            "model": model, "clearance_threshold": threshold,
            "clearance_rule": "Every cell in this and all earlier tiers has Wilson lower 95% bound >= threshold",
            "scale_policy": policy, "mc_accuracy_reportable": accuracy_reportable,
            "highest_cleared_tier": highest if accuracy_reportable else None,
            "diagnostic_highest_cleared_tier": highest,
            "tiers": tiers, "cells": cells, "items": rows,
            "notes": ["Uncalibrated synthetic probes; no size-to-tier guarantee.",
                      "D is static state tracking, not interactive agent capability.",
                      "Choice NLL is conditional on four candidates, not corpus LM loss.",
                      "Wilson intervals are item-level, not adjusted for multiple comparisons or template dependence."]}


def evaluate(items, backend="chance", model_path=None, revision=None, device="cpu", batch_size=8,
             parameters=None, training_tokens=None, threshold=0.75, rung=None):
    rung = resolve_rung(parameters, rung)
    metadata = {"backend": backend, "name": model_path or backend, "revision": revision,
                "parameters": parameters, "training_tokens": training_tokens, "rung_millions": rung}
    if backend in ("chance", "oracle"):
        scores = [[-math.log(4)] * 4 if backend == "chance" else
                  [-0.01 if j == item.answer else -20.0 for j in range(4)] for item in items]
        metadata["synthetic_control"] = True
    elif backend == "hf":
        if not model_path:
            raise ValueError("--model is required for hf")
        try:
            from lm_eval.models.huggingface import HFLM
            from lm_eval.api.instance import Instance
        except ImportError as exc:
            raise RuntimeError('Install the HF backend: pip install -e ".[legacy-hf]"') from exc
        args = dict(pretrained=model_path, device=device, batch_size=batch_size, trust_remote_code=False)
        if revision:
            args["revision"] = revision
        lm = HFLM(**args)
        requests = []
        for item in items:
            for j, choice in enumerate(item.choices):
                continuation = " " + choice
                # Same boundary encoding used by HFLM; refuse silent left truncation.
                context_tokens, continuation_tokens = lm._encode_pair(item.prompt, continuation)
                if len(context_tokens) + len(continuation_tokens) > lm.max_length:
                    raise ValueError(f"Context exceeds model limit ({lm.max_length}): {item.id}")
                requests.append(Instance(request_type="loglikelihood", doc={},
                                         arguments=(item.prompt, continuation), idx=j))
        results = lm.loglikelihood(requests)
        if len(results) != len(requests):
            raise ValueError("Backend returned an incomplete result set")
        scores = [[float(results[i * 4 + j][0]) for j in range(4)] for i in range(len(items))]
        tokenizer_json = lm.tokenizer.backend_tokenizer.to_str() if hasattr(lm.tokenizer, "backend_tokenizer") else str(sorted(lm.tokenizer.get_vocab().items()))
        import hashlib
        metadata.update({"synthetic_control": False, "device": device,
                         "actual_parameters": sum(p.numel() for p in lm.model.parameters()),
                         "resolved_revision": getattr(lm.model.config, "_commit_hash", None),
                         "tokenizer": lm.tokenizer.name_or_path,
                         "tokenizer_sha256": hashlib.sha256(tokenizer_json.encode()).hexdigest(),
                         "max_length": lm.max_length,
                         "packages": {p: importlib.metadata.version(p) for p in ("lm-eval", "transformers", "torch")}})
    else:
        raise ValueError("Unknown backend")
    rows = [score_item(item, score) for item, score in zip(items, scores)]
    return summarize(items, rows, metadata, threshold)


def compare(before, after, draws=2000):
    if before["suite_sha256"] != after["suite_sha256"] or before["protocol"] != after["protocol"]:
        raise ValueError("Paired comparison requires identical suite and scoring protocol")
    a, b = before["items"], after["items"]
    if [r["id"] for r in a] != [r["id"] for r in b]:
        raise ValueError("Item IDs/order differ")
    grouped = defaultdict(list)
    for x, y in zip(a, b):
        grouped[f'{x["tier"]}/{x["family"]}/d{x["difficulty"]}'].append((x, y))
    result = {}
    for key, pairs in grouped.items():
        diffs = [y["correct"] - x["correct"] for x, y in pairs]
        r = random.Random(seed_for("bootstrap", key))
        boot = sorted(statistics.mean(r.choices(diffs, k=len(diffs))) for _ in range(draws))
        result[key] = {"n": len(diffs), "accuracy_delta": statistics.mean(diffs),
                       "paired_bootstrap_ci95": [boot[int(draws * .025)], boot[int(draws * .975)]],
                       "choice_nll_delta": statistics.mean(y["choice_nll"] - x["choice_nll"] for x, y in pairs),
                       "improved_items": sum(d > 0 for d in diffs), "regressed_items": sum(d < 0 for d in diffs)}
    return {"before": before["model"], "after": after["model"], "cells": result,
            "note": "Item bootstrap conditional on fixed templates; intervals are not multiplicity-adjusted."}


def markdown(report):
    show_accuracy = report.get("mc_accuracy_reportable", True)
    policy = report.get("scale_policy")
    lines = [f'# Benchmark ladder: {report["model"]["name"]}', '',
             'Control run; not model capability evidence.' if report["model"].get("synthetic_control") else 'Model evaluation.', '']
    if policy:
        lines.extend([f'Scale policy: **{policy["rung_millions"]}M** (user-specified; uncalibrated).', '',
                      '| Measurement | Status | Implementation |', '|---|---|---|'])
        for metric, entry in policy["metrics"].items():
            lines.append(f'| {metric} | {entry["status"]} | {entry["implementation"]} |')
        lines.extend(['', 'Status is a reporting role, not a measured floor or capability claim.', ''])
    else:
        lines.extend(['No scale policy selected; synthetic diagnostic report.', ''])
    if show_accuracy:
        lines.extend([f'Highest cleared synthetic tier: **{report["highest_cleared_tier"] or "none"}**', '',
                      '| Cell | n | Accuracy (95% CI) | Correct-choice BPB | Choice NLL | Signal |',
                      '|---|---:|---:|---:|---:|---|'])
    else:
        lines.extend(['MC accuracy and accuracy-based tier clearance are diagnostic-only at this rung; retained in JSON.', '',
                      '| Cell | n | Correct-choice BPB | Choice NLL |', '|---|---:|---:|---:|'])
    for key, value in report["cells"].items():
        bpb = value["mc_correct_choice_bpb"]
        if show_accuracy:
            lo, hi = value["accuracy_ci95"]
            lines.append(f'| {key} | {value["n"]} | {value["accuracy"]:.1%} ({lo:.1%}–{hi:.1%}) | {bpb:.3f} | {value["choice_nll"]:.3f} | {value["signal"]} |')
        else:
            lines.append(f'| {key} | {value["n"]} | {bpb:.3f} | {value["choice_nll"]:.3f} |')
    lines.extend(['', 'Correct-choice BPB = summed gold completion NLL / (summed UTF-8 bytes × ln(2)); includes the continuation prefix. This is a synthetic proxy, not corpus BPB.', ''])
    if show_accuracy:
        lines.extend(['Random baseline: 25%; uniform-choice NLL: 1.386 nats.', '', report["clearance_rule"], ''])
    lines.extend(f'- {note}' for note in report["notes"])
    return '\n'.join(lines) + '\n'
