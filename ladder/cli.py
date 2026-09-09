import argparse
import json
from pathlib import Path

from .evaluation import compare, evaluate, markdown
from .tasks import digest, suite
from .gating import RUNGS, scale_policy


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Synthetic capability ladder for tiny LLMs")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "run"):
        p = sub.add_parser(command)
        p.add_argument("--count", type=int, default=32, help="Items per family/difficulty; multiple of four")
        p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--split", choices=["dev", "eval"], default="eval")
        p.add_argument("--output", required=True)
        if command == "run":
            p.add_argument("--backend", choices=["chance", "oracle", "hf"], default="chance")
            p.add_argument("--model")
            p.add_argument("--revision")
            p.add_argument("--device", default="cpu")
            p.add_argument("--batch-size", type=int, default=8)
            p.add_argument("--parameters", type=int)
            p.add_argument("--rung", type=int, choices=RUNGS, help="Reporting rung in millions; explicit for approximate checkpoint sizes")
            p.add_argument("--training-tokens", type=int)
            p.add_argument("--threshold", type=float, default=.75)
    p = sub.add_parser("policy")
    p.add_argument("--rung", type=int, choices=RUNGS, required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("compare")
    p.add_argument("before")
    p.add_argument("after")
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "run" and Path(args.output).suffix != ".json":
            raise ValueError("run output must end in .json (a companion .md report is written)")
        if args.command == "policy":
            result = scale_policy(args.rung)
        elif args.command == "compare":
            result = compare(json.loads(Path(args.before).read_text()), json.loads(Path(args.after).read_text()))
        else:
            items = suite(args.count, tuple(args.levels), args.seed, args.split)
            if args.command == "generate":
                result = {"sha256": digest(items), "items": [i.to_dict() for i in items]}
            else:
                result = evaluate(items, args.backend, args.model, args.revision, args.device, args.batch_size,
                                  args.parameters, args.training_tokens, args.threshold, args.rung)
        write(args.output, result)
        if args.command == "run":
            Path(args.output).with_suffix(".md").write_text(markdown(result))
            print(f'{len(result["items"])} items; highest cleared tier: {result["highest_cleared_tier"]}')
        print(args.output)
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
