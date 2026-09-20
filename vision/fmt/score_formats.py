"""Score fixtures under alternative prompt formats using SemIf's unchanged direct readout.

Formats:
  json    - SemIf's shipped format (control; must reproduce the CLI output)
  compact - same JSON shape but options as a {"A": desc, ...} object
  plain   - plain text: Evidence / Criterion / lettered options
Usage: score_formats.py <fixture.jsonl> <out_dir> [format ...]
"""
import json, sys, time
import os
from pathlib import Path
_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "src")); sys.path.insert(0, os.path.dirname(_HERE))
import semif_phase1.direct as direct
from semif_phase1.core import DIRECT_SYSTEM, LETTERS, validate_row, load_causal_model, direct_messages as json_messages

MODEL, REV = "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def _state_text(state):
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def compact_messages(row):
    validate_row(row)
    payload = {"evidence": row["state"], "criterion": row["question"],
               "options": {LETTERS[i]: o["description"] for i, o in enumerate(row["options"])}}
    return [{"role": "system", "content": DIRECT_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def plain_messages(row):
    validate_row(row)
    lines = [f"Evidence: {_state_text(row['state'])}", f"Criterion: {row['question']}", "Options:"]
    lines += [f"{LETTERS[i]}) {o['description']}" for i, o in enumerate(row["options"])]
    return [{"role": "system", "content": DIRECT_SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


FORMATS = {"json": json_messages, "compact": compact_messages, "plain": plain_messages}


def main():
    fixture, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    formats = sys.argv[3:] or list(FORMATS)
    rows = [json.loads(l) for l in fixture.read_text().splitlines() if l.strip()]
    model, tok, meta = load_causal_model(MODEL, REV)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in formats:
        direct.direct_messages = FORMATS[name]  # encode_prompt looks this up in the module namespace
        out = out_dir / f"{name}-{fixture.stem}.jsonl"
        if out.exists():
            print(f"{name}: {out} exists, skipping"); continue
        t0 = time.perf_counter(); toks = []
        with out.open("x") as sink:
            for row in rows:
                r = direct.score(model, tok, row, {**meta, "prompt_format": name})
                toks.append(r["input_tokens"]); sink.write(json.dumps(r) + "\n")
        print(f"{name:<8} {fixture.stem}: {len(rows)} rows, mean input tokens {sum(toks)/len(toks):6.1f}, "
              f"{time.perf_counter()-t0:5.1f}s", flush=True)


if __name__ == "__main__":
    main()
