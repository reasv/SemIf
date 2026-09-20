import json, sys, time, statistics as st, inspect
import os
import torch, transformers
from PIL import Image, ImageDraw
_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "src")); sys.path.insert(0, _HERE)
from semif_phase1.core import DIRECT_SYSTEM, LETTERS
from semif_phase1.direct import _slot_ids

MODEL, REV = "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
proc = transformers.AutoProcessor.from_pretrained(MODEL, revision=REV)
model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL, revision=REV, dtype=torch.bfloat16, device_map={"": "cuda:0"}, low_cpu_mem_usage=True).eval()
params = inspect.signature(model.forward).parameters

def frame(w, h):
    im = Image.new("RGB", (w, h), (30, 30, 30))
    d = ImageDraw.Draw(im)
    for i in range(0, w, 97): d.rectangle((i, (i*7)%h, i+60, (i*7)%h+60), fill=(200, 40, 40))
    return im

payload = {"evidence": "The attached camera frame.", "criterion": "Is a red object visible?",
           "options": [{"letter": "A", "description": "Yes."}, {"letter": "B", "description": "No."}]}
msgs = [{"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": json.dumps(payload)}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
slots = _slot_ids(proc.tokenizer, 2)

def run(w, h, n=15, max_pixels=None):
    im = frame(w, h)
    kw = {"max_pixels": max_pixels} if max_pixels else {}
    pre, vis, lm, tot = [], [], [], []
    for i in range(n + 3):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        inp = proc(text=[text], images=[im], return_tensors="pt", **({"images_kwargs": kw} if kw else {})).to("cuda:0")
        torch.cuda.synchronize(); t1 = time.perf_counter()
        with torch.inference_mode():
            # vision encoder alone
            _ = model.model.get_image_features(inp["pixel_values"], inp["image_grid_thw"])
            torch.cuda.synchronize(); t2 = time.perf_counter()
            k = dict(inp, use_cache=False, return_dict=True)
            if "logits_to_keep" in params: k["logits_to_keep"] = 1
            logits = model(**k).logits[0, -1, :]
            torch.cuda.synchronize(); t3 = time.perf_counter()
        if i >= 3:
            pre.append(t1-t0); vis.append(t2-t1); lm.append((t3-t2)-(t2-t1)); tot.append((t1-t0)+(t3-t2))
    ntok = int(inp["input_ids"].shape[1]); nimg = int((inp["input_ids"] == model.config.image_token_id).sum())
    med = lambda x: st.median(x)*1000
    print(f"{w}x{h:<5} {'mp='+str(max_pixels) if max_pixels else '':<12} tokens={ntok:5d} img={nimg:5d} "
          f"preproc={med(pre):6.1f}ms  vision={med(vis):6.1f}ms  lm={med(lm):6.1f}ms  end2end={med(tot):6.1f}ms  p95={sorted(tot)[int(0.95*len(tot))-1]*1000:6.1f}ms", flush=True)

for (w, h) in [(224, 224), (448, 448), (640, 480), (896, 448), (1280, 720), (1920, 1080)]:
    run(w, h)
print("--- 1080p frame downscaled via max_pixels ---")
for mp in (256*28*28, 512*28*28, 1024*28*28):
    run(1920, 1080, max_pixels=mp)
