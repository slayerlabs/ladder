"""Deterministic generators; each item includes machine-verifiable provenance."""
from dataclasses import asdict, dataclass
import hashlib
import json
import random

VERSION = "synthetic-v1"
FAMILIES = {"copy": "0", "progression": "0", "lookup": "A", "arithmetic": "B",
            "sorting": "B", "multihop": "C", "state_tracking": "D"}
TIERS = ("0", "A", "B", "C", "D")
WORDS = "apple bird chair dog egg fish gate hat ice jar kite leaf moon nest owl pear queen rain sun tree urn vase wolf yarn zebra".split()


def seed_for(*parts):
    return int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:16], 16)


@dataclass(frozen=True)
class Item:
    id: str
    family: str
    tier: str
    difficulty: int
    prompt: str
    choices: list[str]
    answer: int
    evidence: dict

    def to_dict(self):
        return asdict(self)


def generate(family, level, index, seed=42, split="eval"):
    if family not in FAMILIES or not 1 <= level <= 12 or index < 0:
        raise ValueError("Unknown family, negative index, or difficulty outside 1..12")
    r = random.Random(seed_for(VERSION, seed, split, family, level, index))
    n = level + 2
    if family == "copy":
        seq = r.choices(WORDS, k=n)
        gold = " ".join(seq)
        alternatives = set()
        while len(alternatives) < 3:
            alt = seq.copy()
            pos = r.randrange(n)
            alt[pos] = r.choice([w for w in WORDS if w != seq[pos]])
            alternatives.add(" ".join(alt))
        prompt = f"Copy the words exactly.\nWords: {gold}\nCopy:"
        evidence = {"sequence": seq}
    elif family == "progression":
        start, step = r.randrange(100), r.randint(1, level * 3)
        seq = [start + i * step for i in range(n)]
        gold = str(start + n * step)
        alternatives = {str(int(gold) + delta) for delta in (-step, step, 2 * step)}
        prompt = f"Continue the number sequence.\nSequence: {', '.join(map(str, seq))}\nNext:"
        evidence = {"start": start, "step": step, "length": n}
    elif family == "lookup":
        keys = r.sample(WORDS, n)
        vals = r.sample(range(10, 100), n)
        pairs = dict(zip(keys, vals))
        key = r.choice(keys)
        gold = str(pairs[key])
        alternatives = {str(v) for v in r.sample([v for v in range(10, 100) if str(v) != gold], 3)}
        prompt = "\n".join(f"{k}: {v}" for k, v in pairs.items()) + f"\nValue of {key}:"
        evidence = {"pairs": pairs, "query": key}
    elif family == "arithmetic":
        terms = [r.randint(0, 19) for _ in range(level + 1)]
        gold = str(sum(terms))
        alternatives = {str(int(gold) + delta) for delta in (-1, 1, 2)}
        prompt = " + ".join(map(str, terms)) + " ="
        evidence = {"terms": terms}
    elif family == "sorting":
        seq = r.sample(range(10, 100), n)
        ordered = sorted(seq)
        gold = " ".join(map(str, ordered))
        alternatives = set()
        while len(alternatives) < 3:
            alt = ordered.copy()
            a, b = r.sample(range(n), 2)
            alt[a], alt[b] = alt[b], alt[a]
            alternatives.add(" ".join(map(str, alt)))
        prompt = f"Sort from smallest to largest.\nNumbers: {' '.join(map(str, seq))}\nSorted:"
        evidence = {"sequence": seq}
    elif family == "multihop":
        nodes = r.sample(WORDS, level + 3)
        links = dict(zip(nodes[:-1], nodes[1:]))
        facts = [f"{a} points to {b}." for a, b in links.items()]
        r.shuffle(facts)
        hops = level + 1
        gold = nodes[hops]
        alternatives = set(r.sample([w for w in WORDS if w != gold], 3))
        prompt = "\n".join(facts) + f"\nStart at {nodes[0]}. Follow {hops} links.\nEnd:"
        evidence = {"links": links, "start": nodes[0], "hops": hops}
    else:
        initial = r.randint(1, 9)
        state = initial
        operations = []
        for _ in range(level + 1):
            op = r.choice(["add", "subtract", "set"])
            value = r.randint(1, 9)
            operations.append([op, value])
            state = state + value if op == "add" else state - value if op == "subtract" else value
        gold = str(state)
        alternatives = {str(state + delta) for delta in (-1, 1, 2)}
        prompt = f"Initial register: {initial}.\n" + "\n".join(f"{op} {v}" for op, v in operations) + "\nFinal register:"
        evidence = {"initial": initial, "operations": operations}
    choices = sorted(alternatives)
    # Exactly balanced answer positions in each cell when count is divisible by 4.
    block = list(range(4))
    random.Random(seed_for(seed, split, family, level, index // 4)).shuffle(block)
    answer = block[index % 4]
    choices.insert(answer, gold)
    return Item(f"{VERSION}/{split}/{seed}/{family}/{level}/{index}", family,
                FAMILIES[family], level, prompt, choices, answer, evidence)


def verify(item):
    e = item.evidence
    if item.family == "copy":
        gold = " ".join(e["sequence"])
    elif item.family == "progression":
        gold = str(e["start"] + e["step"] * e["length"])
    elif item.family == "lookup":
        gold = str(e["pairs"][e["query"]])
    elif item.family == "arithmetic":
        gold = str(sum(e["terms"]))
    elif item.family == "sorting":
        gold = " ".join(map(str, sorted(e["sequence"])))
    elif item.family == "multihop":
        gold = e["start"]
        for _ in range(e["hops"]):
            gold = e["links"][gold]
    elif item.family == "state_tracking":
        value = e["initial"]
        for op, v in e["operations"]:
            value = {"add": lambda: value + v, "subtract": lambda: value - v, "set": lambda: v}[op]()
        gold = str(value)
    else:
        raise ValueError("Unknown family")
    if len(item.choices) != 4 or len(set(item.choices)) != 4 or item.choices[item.answer] != gold:
        raise ValueError(f"Invalid item: {item.id}")


def suite(count=32, levels=(1, 2, 3), seed=42, split="eval"):
    if count < 4 or count % 4:
        raise ValueError("count must be a positive multiple of 4 (at least 4)")
    if not levels or len(set(levels)) != len(levels):
        raise ValueError("levels must be nonempty and unique")
    items, seen = [], set()
    for family in FAMILIES:
        for level in levels:
            accepted, index = 0, 0
            while accepted < count:
                item = generate(family, level, index, seed, split)
                index += 1
                fingerprint = (item.prompt, tuple(sorted(item.choices)))
                if fingerprint in seen:
                    if index > count * 100:
                        raise ValueError("Requested suite exceeds unique generator capacity")
                    continue
                # Rebalance after any duplicate rejection.
                choices = item.choices.copy()
                gold = choices.pop(item.answer)
                positions = list(range(4))
                random.Random(seed_for("balance", seed, split, family, level, accepted // 4)).shuffle(positions)
                position = positions[accepted % 4]
                choices.insert(position, gold)
                item = Item(item.id, family, item.tier, level, item.prompt, choices, position, item.evidence)
                verify(item)
                seen.add(fingerprint)
                items.append(item)
                accepted += 1
    return items


def digest(items):
    return hashlib.sha256(json.dumps([i.to_dict() for i in items], sort_keys=True).encode()).hexdigest()
