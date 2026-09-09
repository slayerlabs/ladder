"""Continuous benchmark CLI. Legacy synthetic CLI remains under ladder-synthetic."""
import argparse
import json
from pathlib import Path
from .corpus import freeze
from .analysis import calibrate, rank_agreement, decision
from .gating import RUNGS, scale_policy


def dump(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    p = argparse.ArgumentParser(description='Continuous tiny-LLM ladder: freeze, evaluate, calibrate, validate ranks')
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('freeze-val')
    f.add_argument('--sources', required=True)
    f.add_argument('--train', nargs='+', required=True)
    f.add_argument('--candidates', required=True)
    f.add_argument('--reference-tokenizer', required=True)
    f.add_argument('--revision')
    f.add_argument('--output', required=True)
    f = sub.add_parser('import-multiblimp')
    f.add_argument('--tsv', required=True)
    f.add_argument('--revision', required=True)
    f.add_argument('--output', required=True)
    f = sub.add_parser('eval')
    f.add_argument('--model', required=True)
    f.add_argument('--revision')
    f.add_argument('--corpus', required=True)
    f.add_argument('--pl-pairs')
    f.add_argument('--review')
    f.add_argument('--en-pairs')
    f.add_argument('--mc')
    f.add_argument('--tier', choices=['micro', 'fast', 'full'], required=True)
    f.add_argument('--rung', type=int, choices=RUNGS, required=True)
    f.add_argument('--device', default='cpu')
    f.add_argument('--batch-size', type=int, default=8)
    f.add_argument('--context', type=int, default=512)
    f.add_argument('--allow-incomplete', action='store_true')
    f.add_argument('--output', required=True)
    f = sub.add_parser('collect')
    f.add_argument('--manifest', required=True)
    f.add_argument('--output', required=True)
    for command in ('calibrate', 'rank-agreement', 'decide'):
        f = sub.add_parser(command)
        f.add_argument('--runs', required=True)
        f.add_argument('--output', required=True)
        if command in ('calibrate', 'decide'):
            f.add_argument('--control-mix', required=True)
        if command == 'decide':
            f.add_argument('--candidate-mix', required=True)
            f.add_argument('--rung', type=int, choices=RUNGS, required=True)
            f.add_argument('--pair-metrics', nargs='+', required=True)
    f = sub.add_parser('policy')
    f.add_argument('--rung', type=int, choices=RUNGS, required=True)
    f.add_argument('--output', required=True)
    args = p.parse_args()
    try:
        if Path(args.output).exists():
            raise ValueError('Output exists; choose a new version or run path')
        if args.command == 'freeze-val':
            from transformers import AutoTokenizer
            import hashlib
            tokenizer = AutoTokenizer.from_pretrained(args.reference_tokenizer, revision=args.revision, use_fast=True, trust_remote_code=False)
            metadata = {'name': args.reference_tokenizer, 'revision': args.revision,
                        'tokenizer_sha256': hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()}
            freeze(args.sources, args.train, args.candidates, args.output, tokenizer, metadata)
            print(args.output)
            return
        if args.command == 'import-multiblimp':
            from .pairs import import_multiblimp
            import_multiblimp(args.tsv, args.revision, args.output)
            print(args.output)
            return
        if args.command == 'eval':
            from .scoring import Scorer
            from .benchmark import evaluate_benchmark
            scorer = Scorer(args.model, args.revision, args.device, args.batch_size, args.context)
            result = evaluate_benchmark(scorer, args.corpus, args.pl_pairs, args.review, args.en_pairs,
                                        args.tier, args.rung, args.mc, args.allow_incomplete)
        elif args.command == 'collect':
            from .collection import collect
            result = collect(args.manifest)
        elif args.command == 'policy':
            result = scale_policy(args.rung)
        else:
            runs = json.loads(Path(args.runs).read_text())
            if args.command == 'calibrate':
                result = calibrate(runs, args.control_mix)
            elif args.command == 'rank-agreement':
                result = rank_agreement(runs)
            else:
                result = decision(runs, args.control_mix, args.candidate_mix, args.rung, args.pair_metrics)
        dump(args.output, result)
        print(args.output)
    except (ValueError, FileExistsError, KeyError) as exc:
        p.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    main()
