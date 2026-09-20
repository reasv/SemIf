"""Compare prompt formats on each fixture with the repo's own evaluator (paired bootstrap vs the json control)."""
import json, os, sys
from pathlib import Path
_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "benchmarks"))
from evaluate import read_jsonl, evaluate, align, balanced_metric

REPO = Path(_REPO); FMT = Path(_HERE).parent / "results" / "predictions"
FIXTURES = {"authored144": REPO / "benchmarks/data/authored144.jsonl",
            "perturbations108": REPO / "benchmarks/data/perturbations108.jsonl",
            "wanli256": FMT / "wanli256.jsonl"}
PUBLISHED = {"authored144": "direct-authored144.jsonl", "perturbations108": "direct-perturbations108.jsonl",
             "wanli256": "direct-wanli256.jsonl"}


def by_id(rows): return {r["id"]: r for r in rows}


def argmax_id(r): return r["option_ids"][max(range(len(r["probabilities"])), key=r["probabilities"].__getitem__)]


def parity(a, b):
    a, b = by_id(a), by_id(b); ids = sorted(set(a) & set(b))
    flips = sum(argmax_id(a[i]) != argmax_id(b[i]) for i in ids)
    worst = max(max(abs(x - y) for x, y in zip(a[i]["probabilities"], b[i]["probabilities"])) for i in ids)
    return len(ids), flips, worst


for name, gold_path in FIXTURES.items():
    gold = read_jsonl(gold_path)
    preds = {f: read_jsonl(p) for f in ("json", "compact", "plain") if (p := FMT / f"{f}-{name}.jsonl").exists()}
    if "json" not in preds: print(f"{name}: no json control yet"); continue
    print(f"\n=== {name} ({len(gold)} rows) ===")
    cli = FMT / f"cli-json-{name}.jsonl"
    if cli.exists():
        n, flips, worst = parity(preds["json"], read_jsonl(cli))
        print(f"  parity json-vs-CLI: {n} rows, {flips} argmax flips, worst prob diff {worst:.4f}")
    pub = REPO / "results/raw/predictions" / PUBLISHED[name]
    if pub.exists():
        n, flips, worst = parity(preds["json"], read_jsonl(pub))
        print(f"  parity json-vs-published (3090 run): {n} rows, {flips} argmax flips, worst prob diff {worst:.4f}")
    base = evaluate(gold, preds["json"])
    print(f"  {'format':<8} {'tokens':>7} {'bal.acc':>8} {'acc':>6} {'nll':>6}  diff vs json [paired 95% CI]   flips")
    for f, rows in preds.items():
        res = evaluate(gold, rows, comparison=preds["json"] if f != "json" else None)
        toks = sum(r["input_tokens"] for r in rows) / len(rows)
        fam = res["family_results"]; acc = sum(v["n"] * v["accuracy"] for v in fam.values()) / len(gold)
        nll = sum(v["n"] * v["nll"] for v in fam.values()) / len(gold)
        extra = ""
        if f != "json":
            pc = res["paired_comparison"]; lo, hi = pc["paired_source_group_bootstrap_95"]
            _, flips, _ = parity(rows, preds["json"])
            extra = f"{pc['difference']:+.4f} [{lo:+.3f}, {hi:+.3f}]   {flips}"
        print(f"  {f:<8} {toks:7.1f} {res['mean_family_balanced_accuracy']:8.4f} {acc:6.3f} {nll:6.3f}  {extra}")
