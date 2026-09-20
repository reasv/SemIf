"""Probe: does SemIf's direct option-logit readout work with image evidence?

SemIf loads only the text tower (Qwen3_5ForCausalLM). This loads the full
Qwen3_5ForConditionalGeneration checkpoint plus its processor, puts an image
in the user turn, and reads the same single-token A/B/C logits at the last
position. Synthetic images are generated with PIL so every expected answer
is known. Run with exactly one visible GPU.
"""
from __future__ import annotations

import inspect
import json
import sys
import time
from pathlib import Path

import torch
import transformers
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from semif_phase1.core import DIRECT_SYSTEM, LETTERS, softmax  # noqa: E402
from semif_phase1.direct import _slot_ids  # noqa: E402

MODEL = "Qwen/Qwen3.5-4B"
REV = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results" / "vision_probe_results.jsonl"
IMG_DIR = OUT.parent / "vision_probe_images"
IMG_DIR.mkdir(parents=True, exist_ok=True)


def font(size):
    for p in ("/usr/share/fonts/TTF/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


def shape_image(color, shape):
    im = Image.new("RGB", (448, 448), "white")
    d = ImageDraw.Draw(im)
    if shape == "circle":
        d.ellipse((90, 90, 358, 358), fill=color)
    else:
        d.rectangle((90, 90, 358, 358), fill=color)
    return im


def text_image(lines, bg="white"):
    im = Image.new("RGB", (896, 448), bg)
    d = ImageDraw.Draw(im)
    y = 40
    for line, size, fill in lines:
        d.text((40, y), line, fill=fill, font=font(size))
        y += size + 18
    return im


def count_image(n):
    im = Image.new("RGB", (640, 320), "white")
    d = ImageDraw.Draw(im)
    for i in range(n):
        x = 40 + i * 110
        d.ellipse((x, 110, x + 90, 200), fill="black")
    return im


CASES = [
    # (id, image, question, options, expected option id)
    ("color-red", shape_image("red", "circle"), "What color is the shape in the image?",
     [("red", "The shape is red."), ("blue", "The shape is blue."), ("green", "The shape is green.")], "red"),
    ("color-blue", shape_image("blue", "circle"), "What color is the shape in the image?",
     [("red", "The shape is red."), ("blue", "The shape is blue."), ("green", "The shape is green.")], "blue"),
    ("shape-square", shape_image("green", "square"), "What shape is drawn in the image?",
     [("circle", "A circle."), ("square", "A square."), ("triangle", "A triangle.")], "square"),
    ("count-3", count_image(3), "How many black dots are in the image?",
     [("two", "Two dots."), ("three", "Three dots."), ("four", "Four dots."), ("five", "Five dots.")], "three"),
    ("count-5", count_image(5), "How many black dots are in the image?",
     [("two", "Two dots."), ("three", "Three dots."), ("four", "Four dots."), ("five", "Five dots.")], "five"),
    ("deploy-failed", text_image([("Deployment dashboard", 40, "black"),
                                  ("release 2026.09.20  14:02 UTC", 28, "gray"),
                                  ("STATUS: FAILED", 48, "red"),
                                  ("Health checks: 0/3 zones healthy", 30, "black"),
                                  ("Rollback: initiated", 30, "black")]),
     "Is there evidence that the deployment succeeded?",
     [("yes", "The deployment succeeded."), ("no", "The deployment did not succeed."),
      ("insufficient", "The evidence is insufficient to decide.")], "no"),
    ("deploy-ok", text_image([("Deployment dashboard", 40, "black"),
                              ("release 2026.09.20  14:02 UTC", 28, "gray"),
                              ("STATUS: SUCCESS", 48, "green"),
                              ("Health checks: 3/3 zones healthy", 30, "black"),
                              ("Rollback: none", 30, "black")]),
     "Is there evidence that the deployment succeeded?",
     [("yes", "The deployment succeeded."), ("no", "The deployment did not succeed."),
      ("insufficient", "The evidence is insufficient to decide.")], "yes"),
    ("ticket-route", text_image([("Support ticket #4821", 40, "black"),
                                 ("From: customer@example.com", 28, "gray"),
                                 ("I was charged twice for my September", 32, "black"),
                                 ("invoice. Please refund the duplicate", 32, "black"),
                                 ("payment of $49.", 32, "black")]),
     "Which queue should handle this request?",
     [("account_access", "Account access and authentication support."),
      ("billing", "Billing and payment support."), ("sales", "Sales and product evaluation.")], "billing"),
]


def main():
    assert torch.cuda.device_count() == 1, "expose exactly one GPU"
    t0 = time.perf_counter()
    processor = transformers.AutoProcessor.from_pretrained(MODEL, revision=REV)
    tokenizer = processor.tokenizer
    model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, revision=REV, dtype=torch.bfloat16, device_map={"": "cuda:0"}, low_cpu_mem_usage=True)
    model.eval()
    print(f"loaded full VLM in {time.perf_counter()-t0:.1f}s; "
          f"vram={torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)
    params = inspect.signature(model.forward).parameters

    correct = 0
    with OUT.open("x") as sink:
        for cid, image, question, options, expected in CASES:
            image.save(IMG_DIR / f"{cid}.png")
            payload = {
                "evidence": "The attached image.",
                "criterion": question,
                "options": [{"letter": LETTERS[i], "description": d} for i, (_, d) in enumerate(options)],
            }
            messages = [
                {"role": "system", "content": DIRECT_SYSTEM},
                {"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": json.dumps(payload)}]},
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda:0")
            slots = _slot_ids(tokenizer, len(options))
            kwargs = dict(inputs, use_cache=False, return_dict=True)
            if "logits_to_keep" in params:
                kwargs["logits_to_keep"] = 1
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            with torch.inference_mode():
                logits = model(**kwargs).logits[0, -1, :].float()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t1
            sel = logits[slots].cpu().tolist()
            probs = softmax(sel)
            ids = [o for o, _ in options]
            pick = ids[max(range(len(probs)), key=probs.__getitem__)]
            ok = pick == expected
            correct += ok
            n_img = int((inputs["input_ids"] == model.config.image_token_id).sum())
            rec = {"id": cid, "expected": expected, "argmax": pick, "correct": ok,
                   "probabilities": dict(zip(ids, [round(p, 4) for p in probs])),
                   "input_tokens": int(inputs["input_ids"].shape[1]), "image_tokens": n_img,
                   "forward_seconds": round(dt, 4)}
            print(json.dumps(rec), flush=True)
            sink.write(json.dumps(rec) + "\n")
    print(f"\n{correct}/{len(CASES)} correct", flush=True)


if __name__ == "__main__":
    main()
