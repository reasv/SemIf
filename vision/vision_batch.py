import json, sys, time, statistics as st, inspect
import os
import torch, transformers
from PIL import Image, ImageDraw
_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "src")); sys.path.insert(0, _HERE)
from semif_phase1.core import DIRECT_SYSTEM

MODEL, REV = "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
proc = transformers.AutoProcessor.from_pretrained(MODEL, revision=REV)
proc.tokenizer.padding_side = "left"
model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL, revision=REV, dtype=torch.bfloat16, device_map={"": "cuda:0"}, low_cpu_mem_usage=True).eval()
params = inspect.signature(model.forward).parameters

def frame(w, h, seed):
    im = Image.new("RGB", (w, h), (30, 30, 30)); d = ImageDraw.Draw(im)
    for i in range(0, w, 97): d.rectangle((i, (i*seed*7)%h, i+60, (i*seed*7)%h+60), fill=(200, 40, 40))
    return im

CRITERIA = ["Is a red object visible?", "Is the scene mostly dark?", "Is there a person visible?",
            "Is any text readable?", "Is the frame blurry?", "Is a vehicle visible?", "Is it daytime?",
            "Is the frame empty?", "Is anything moving?", "Is there a door visible?", "Is a screen visible?",
            "Is the camera obstructed?", "Is there a hazard?", "Is a hand visible?", "Is it indoors?", "Is a light on?"]

def prompt(q):
    p = {"evidence": "The attached camera frame.", "criterion": q,
         "options": [{"letter": "A", "description": "Yes."}, {"letter": "B", "description": "No."}]}
    m = [{"role": "system", "content": DIRECT_SYSTEM},
         {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": json.dumps(p)}]}]
    return proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)

def bench(label, images, texts, n=10):
    tot = []
    for i in range(n + 3):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        inp = proc(text=texts, images=images, return_tensors="pt", padding=True).to("cuda:0")
        k = dict(inp, use_cache=False, return_dict=True)
        if "logits_to_keep" in params: k["logits_to_keep"] = 1
        with torch.inference_mode():
            logits = model(**k).logits[:, -1, :]
        torch.cuda.synchronize(); t1 = time.perf_counter()
        if i >= 3: tot.append(t1 - t0)
    b = len(texts); m = st.median(tot) * 1000
    print(f"{label:<34} batch={b:2d} seq={inp['input_ids'].shape[1]:5d} end2end={m:7.1f}ms  per-decision={m/b:6.1f}ms  decisions/s={b/(m/1000):7.1f}", flush=True)

for w, h in [(448, 448), (1280, 720)]:
    print(f"--- {w}x{h}: one frame, N criteria ---")
    f = frame(w, h, 1)
    for b in (1, 4, 8, 16):
        bench(f"1 frame x {b} criteria", [f]*b, [prompt(q) for q in CRITERIA[:b]])
    print(f"--- {w}x{h}: N frames, one criterion ---")
    for b in (1, 4, 8, 16):
        bench(f"{b} frames x 1 criterion", [frame(w, h, s) for s in range(1, b+1)], [prompt(CRITERIA[0])]*b)
print(f"peak vram {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
