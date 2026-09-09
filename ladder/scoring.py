"""Standalone deterministic fp32 causal scoring; no harness or generation."""
import hashlib
import math
import os
from dataclasses import dataclass
from contextlib import nullcontext


@dataclass
class Request:
    text: str
    span: tuple[int, int] | None = None


def pair_probability(good_logprob, bad_logprob):
    margin = good_logprob - bad_logprob
    return 1 / (1 + math.exp(-margin)) if margin >= 0 else math.exp(margin) / (1 + math.exp(margin))


def bpb(total_nll_nats, total_bytes):
    if total_bytes <= 0 or not math.isfinite(total_nll_nats) or total_nll_nats < 0:
        raise ValueError("BPB needs finite nonnegative NLL and positive byte count")
    return total_nll_nats / (math.log(2) * total_bytes)


class Scorer:
    def __init__(self, model_path, revision=None, device="cpu", batch_size=8, context=512, adapter=None, attention=None,
                 dynamic_padding=True):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if batch_size < 1 or context < 2:
            raise ValueError("Positive batch size and context >= 2 required")
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.torch = torch
        if adapter not in (None, 'koliber'):
            raise ValueError('Unknown model adapter')
        self.adapter = adapter
        self.dynamic_padding = bool(dynamic_padding)
        self.attention = attention or ('sdpa_math' if adapter == 'koliber' else 'eager')
        if self.attention not in ('eager', 'sdpa_math') or (adapter == 'koliber' and self.attention != 'sdpa_math'):
            raise ValueError('Use eager or sdpa_math attention; Koliber requires sdpa_math')
        if adapter == 'koliber':
            from .adapters import koliber_snapshot
            snapshot = koliber_snapshot(model_path, revision)
            self.tokenizer = AutoTokenizer.from_pretrained(snapshot, use_fast=True, trust_remote_code=False)
            self.model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.float32,
                                                             trust_remote_code=True).to(device).eval()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision, use_fast=True, trust_remote_code=False)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, revision=revision, dtype=torch.float32,
                                                             attn_implementation="sdpa" if self.attention == "sdpa_math" else "eager", trust_remote_code=False).to(device).eval()
        limit = getattr(self.model.config, "max_position_embeddings", None)
        if limit is not None and context > limit:
            raise ValueError(f"Context {context} exceeds model limit {limit}")
        self.bos = self.tokenizer.bos_token_id
        if self.bos is None:
            self.bos = self.tokenizer.eos_token_id
        if self.bos is None:
            raise ValueError("Tokenizer needs a BOS or EOS document-start token")
        self.device, self.batch_size, self.context = device, batch_size, context
        self.metadata = {"model": model_path, "requested_revision": revision,
                         "resolved_revision": getattr(self.model.config, "_commit_hash", None),
                         "parameters": sum(p.numel() for p in self.model.parameters()),
                         "adapter": adapter, "dtype": "float32", "attention": self.attention, "deterministic_algorithms": True,
                         "batch_shape": [batch_size, "dynamic<=%d" % context] if self.dynamic_padding else [batch_size, context],
                         "stride": context // 2,
                         "document_start_token": self.bos, "device": device,
                         "tokenizer_sha256": hashlib.sha256(self.tokenizer.backend_tokenizer.to_str().encode()).hexdigest()}
        import importlib.metadata
        self.metadata["versions"] = {p: importlib.metadata.version(p) for p in ("torch", "transformers", "tokenizers")}

    def score(self, requests):
        """Score each original byte once for corpus; spans select all overlapping subwords.

        Long documents use half-window stride. Every target token is scored exactly
        once, with bounded left context and a reset at each document boundary.
        """
        t = self.torch
        outputs = []
        jobs = []
        def flush():
            if not jobs:
                return
            width = self.context
            if self.dynamic_padding:
                width = max(len(job[1]) for job in jobs) + 1
            inputs = t.full((self.batch_size, width), self.bos, dtype=t.long, device=self.device)
            masks = t.zeros_like(inputs)
            labels = t.zeros_like(inputs)
            selected = t.zeros_like(inputs, dtype=t.bool)
            for row, (_, ids, targets, choose) in enumerate(jobs):
                n = len(ids)
                inputs[row, :n] = t.tensor(ids, device=self.device)
                labels[row, :n] = t.tensor(targets, device=self.device)
                masks[row, :n] = 1
                selected[row, :n] = t.tensor(choose, device=self.device)
            # Dummy rows have valid attention; no targets are selected.
            masks[len(jobs):, 0] = 1
            from torch.nn.attention import sdpa_kernel, SDPBackend
            attention_context = sdpa_kernel(SDPBackend.MATH) if self.attention == 'sdpa_math' else nullcontext()
            with t.inference_mode(), attention_context:
                logits = self.model(input_ids=inputs, attention_mask=masks, use_cache=False).logits.float()
                losses = t.nn.functional.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
                for row, (index, _, _, _) in enumerate(jobs):
                    values = losses[row][selected[row]].cpu().double().tolist()
                    outputs[index]["parts"].extend(values)
            jobs.clear()
        for index, request in enumerate(requests):
            if not request.text:
                raise ValueError("Empty scoring text")
            encoded = self.tokenizer(request.text, add_special_tokens=False, return_offsets_mapping=True)
            ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
            if self.tokenizer.decode(ids, clean_up_tokenization_spaces=False) != request.text:
                raise ValueError("Tokenizer is not lossless on this text; freeze a canonical text policy first")
            start, end = request.span or (0, len(request.text))
            if not 0 <= start < end <= len(request.text):
                raise ValueError("Invalid character span")
            chosen = [b > start and a < end for a, b in offsets]
            for use, (a, b) in zip(chosen, offsets):
                if use and (a < start or b > end):
                    raise ValueError("Critical-region boundary cuts a token; include leading whitespace or fix the region")
            if not any(chosen):
                raise ValueError("No tokens in scoring region")
            outputs.append({"parts": [], "bytes": len(request.text[start:end].encode("utf-8")),
                            "tokens": sum(chosen)})
            sequence = [self.bos] + ids
            for lo in range(1, len(sequence), self.context // 2):
                hi = min(lo + self.context // 2, len(sequence))
                begin = max(0, hi - self.context - 1)
                choose = [pos >= lo and chosen[pos - 1] for pos in range(begin + 1, hi)]
                if not any(choose):
                    continue
                jobs.append((index, sequence[begin:hi - 1], sequence[begin + 1:hi], choose))
                if len(jobs) == self.batch_size:
                    flush()
        flush()
        for output in outputs:
            parts = output.pop("parts")
            if len(parts) != output["tokens"] or not all(math.isfinite(v) for v in parts):
                raise ValueError("Missing or nonfinite token scores")
            output["nll_nats"] = math.fsum(parts)
            output["bpb"] = bpb(output["nll_nats"], output["bytes"])
        return outputs


def score_pairs(scorer, pairs):
    requests, plan = [], []
    for pair in pairs:
        entries = {}
        for variant in ("good", "bad"):
            entries[variant] = len(requests)
            requests.append(Request(pair[variant]))
            if pair.get("region"):
                entries[variant + "_region"] = len(requests)
                requests.append(Request(pair[variant], tuple(pair["region"][variant])))
        plan.append(entries)
    scored = scorer.score(requests)
    results = []
    for pair, entries in zip(pairs, plan):
        good, bad = (-scored[entries[k]]["nll_nats"] for k in ("good", "bad"))
        row = {"id": pair["id"], "paradigm": pair["paradigm"], "language": pair["language"],
               "sentence_prob": pair_probability(good, bad), "sentence_log_margin": good - bad,
               "sentence_accuracy": float(good > bad) + .5 * float(good == bad),
               "sentence_good_logprob": good, "sentence_bad_logprob": bad, "region_prob": None}
        if pair.get("region"):
            g, b = (-scored[entries[k + "_region"]]["nll_nats"] for k in ("good", "bad"))
            row.update(region_prob=pair_probability(g, b), region_log_margin=g - b,
                       region_accuracy=float(g > b) + .5 * float(g == b),
                       region_good_logprob=g, region_bad_logprob=b)
        results.append(row)
    return results
