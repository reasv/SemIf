"""Shared-image prefix scoring: encode one frame + fixed prefix once, then score N
criteria as batched suffix branches on a replicated cache (port of SemIf shared mode).

Usage: CUDA_VISIBLE_DEVICES=1 python vision/shared_image.py [bench|check]
"""
from __future__ import annotations
import json, os, sys, time, statistics as st
import os
import torch, transformers
from PIL import Image, ImageDraw, ImageFont
_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "src")); sys.path.insert(0, _HERE)
from semif_phase1.core import DIRECT_SYSTEM, LETTERS, softmax
from semif_phase1.direct import _slot_ids

MODEL, REV = "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
EVIDENCE = "The attached image."
DEV = "cuda:0"


def load():
    proc = transformers.AutoProcessor.from_pretrained(MODEL, revision=REV)
    attn = os.environ.get("SEMIF_ATTN")  # e.g. flash_attention_4; applied to the text model only
    if attn == "flash_attention_4":
        import fa4_patch; fa4_patch.apply()
    extra = {"attn_implementation": {"text_config": attn, "vision_config": "sdpa"}} if attn else {}
    model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, revision=REV, dtype=torch.bfloat16, device_map={"": DEV}, low_cpu_mem_usage=True, **extra).eval()
    print("attn implementation:", model.config.text_config._attn_implementation, "| vision:", model.config.vision_config._attn_implementation, flush=True)
    return proc, model


def full_prompt(proc, question, options):
    if os.environ.get("SEMIF_FMT", "json") == "compact":
        payload = {"evidence": EVIDENCE, "criterion": question,
                   "options": {LETTERS[i]: d for i, d in enumerate(options)}}
    else:
        payload = {"evidence": EVIDENCE, "criterion": question,
                   "options": [{"letter": LETTERS[i], "description": d} for i, d in enumerate(options)]}
    msgs = [{"role": "system", "content": DIRECT_SYSTEM},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": json.dumps(payload)}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return text, json.dumps(payload)


def prefix_prompt(proc):
    text, payload = full_prompt(proc, "prefix boundary placeholder", ["Yes", "No"])
    assert text.count(payload) == 1
    evidence = json.dumps({"evidence": EVIDENCE})[:-1]
    assert payload.startswith(evidence)
    return text[: text.index(payload)] + evidence


class SharedImageScorer:
    def __init__(self, proc, model):
        self.proc, self.model, self.tok = proc, model, proc.tokenizer
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.prefix_text = prefix_prompt(proc)

    @torch.inference_mode()
    def score(self, image, decisions):
        """decisions: list of (question, [option descriptions]). Returns probs per decision + timing."""
        t0 = time.perf_counter()
        p = self.proc(text=[self.prefix_text], images=[image], return_tensors="pt").to(DEV)
        prefix_ids = p["input_ids"][0, :-1]  # drop boundary token (may merge with following JSON)
        P = prefix_ids.shape[0]
        suffixes, slots = [], []
        for q, opts in decisions:
            text, _ = full_prompt(self.proc, q, opts)
            ids = self.proc(text=[text], images=[image], return_tensors="pt")["input_ids"][0].to(DEV)
            if ids.shape[0] <= P or not torch.equal(ids[:P], prefix_ids):
                raise ValueError("prefix mismatch for criterion %r" % q)
            suffixes.append(ids[P:].tolist())
            slots.append(_slot_ids(self.tok, len(opts)))
        N = len(suffixes); W = max(map(len, suffixes))
        sfx = torch.full((N, W), self.pad, dtype=torch.long)
        mask = torch.zeros((N, P + W), dtype=torch.long); mask[:, :P] = 1
        pos = torch.zeros((N, W), dtype=torch.long); ends = []
        for i, s in enumerate(suffixes):
            L = len(s); sfx[i, :L] = torch.tensor(s); mask[i, P:P + L] = 1
            pos[i, :L] = torch.arange(P, P + L); ends.append(L - 1)
        sel = sorted(set(ends))
        torch.cuda.synchronize(); t1 = time.perf_counter()
        # 1) prefill image + prefix once
        extra = {"mm_token_type_ids": p["mm_token_type_ids"][:, :-1]} if "mm_token_type_ids" in p else {}
        out = self.model(input_ids=prefix_ids[None], attention_mask=p["attention_mask"][:, :-1],
                         pixel_values=p["pixel_values"], image_grid_thw=p["image_grid_thw"],
                         use_cache=True, return_dict=True, logits_to_keep=1, **extra)
        cache = out.past_key_values; delta = self.model.model.rope_deltas  # (1,)
        assert cache.get_seq_length() == P
        torch.cuda.synchronize(); t2 = time.perf_counter()
        # 2) replicate cache across N branches
        cache.reorder_cache(torch.zeros(N, dtype=torch.long, device=DEV))
        # 3) batched suffix forward with M-RoPE offset (text positions = index + delta on all 3 axes)
        pos3 = (pos.to(DEV) + delta.to(DEV)).unsqueeze(0).expand(3, N, W).contiguous()
        out = self.model(input_ids=sfx.to(DEV), attention_mask=mask.to(DEV), position_ids=pos3,
                         past_key_values=cache, use_cache=True, return_dict=True,
                         logits_to_keep=torch.tensor(sel, dtype=torch.long, device=DEV))
        torch.cuda.synchronize(); t3 = time.perf_counter()
        probs = []
        for i in range(N):
            v = out.logits[i, sel.index(ends[i]), :].float()
            probs.append(softmax(v[slots[i]].cpu().tolist()))
        return probs, {"encode_ms": (t1 - t0) * 1e3, "prefill_ms": (t2 - t1) * 1e3, "suffix_ms": (t3 - t2) * 1e3,
                       "total_ms": (t3 - t0) * 1e3, "prefix_tokens": P, "suffix_width": W, "n": N}


@torch.inference_mode()
def direct(proc, model, image, q, opts):
    text, _ = full_prompt(proc, q, opts)
    inp = proc(text=[text], images=[image], return_tensors="pt").to(DEV)
    logits = model(**inp, use_cache=False, return_dict=True, logits_to_keep=1).logits[0, -1].float()
    return softmax(logits[_slot_ids(proc.tokenizer, len(opts))].cpu().tolist())


def frame(w=640, h=480):
    im = Image.new("RGB", (w, h), "white"); d = ImageDraw.Draw(im)
    f = ImageFont.load_default(28)
    d.text((30, 30), "Support ticket #4821", fill="black", font=ImageFont.load_default(36))
    for i, l in enumerate(["I was charged twice for my September", "invoice. Please refund the duplicate",
                           "payment of $49.", "", "Sent from a phone at 2:14 AM"]):
        d.text((30, 100 + i * 40), l, fill="black", font=f)
    d.ellipse((470, 300, 600, 430), fill="red")
    return im


YN = ["Yes.", "No."]
CRITERIA = [("Which queue should handle this request?",
             ["Account access and authentication support.", "Billing and payment support.", "Sales and product evaluation."]),
            ("Is a red shape visible in the image?", YN), ("Does the customer ask for a refund?", YN),
            ("Is the customer asking to reset a password?", YN), ("Is the amount mentioned more than $100?", YN),
            ("Was the message sent during business hours?", YN), ("Is the image mostly white?", YN),
            ("Is there a ticket number visible?", YN), ("Is the customer angry?", YN),
            ("Is the ticket about September?", YN), ("Is a blue shape visible?", YN),
            ("Does the text mention an invoice?", YN), ("Is the sender's device mentioned?", YN),
            ("Is the ticket in English?", YN), ("Does it request a duplicate charge?", YN),
            ("Is there a photo of a person?", YN)]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
    proc, model = load()
    scorer = SharedImageScorer(proc, model)
    im = frame()
    if mode in ("check", "both"):
        probs, t = scorer.score(im, CRITERIA)
        worst = 0.0
        for (q, opts), ps in zip(CRITERIA, probs):
            d = direct(proc, model, im, q, opts)
            diff = max(abs(a - b) for a, b in zip(ps, d)); worst = max(worst, diff)
            print(f"{q[:48]:<50} shared={[round(x,3) for x in ps]} direct={[round(x,3) for x in d]} maxdiff={diff:.4f}")
        print(f"CHECK worst max-abs prob diff over {len(CRITERIA)} criteria: {worst:.4f}  (prefix_tokens={t['prefix_tokens']})", flush=True)
    if mode in ("bench", "both"):
        for n in (1, 4, 8, 16, 32, 64):
            dec = (CRITERIA * 4)[:n]
            ts = []
            for i in range(10):
                _, t = scorer.score(im, dec)
                if i >= 2: ts.append(t)
            m = {k: st.median(x[k] for x in ts) for k in ("encode_ms", "prefill_ms", "suffix_ms", "total_ms")}
            print(f"n={n:3d} encode={m['encode_ms']:6.1f}ms prefill={m['prefill_ms']:6.1f}ms suffix={m['suffix_ms']:6.1f}ms "
                  f"total={m['total_ms']:6.1f}ms per-decision={m['total_ms']/n:6.1f}ms  gpu-only={(m['prefill_ms']+m['suffix_ms']):6.1f}ms "
                  f"(prefix={ts[0]['prefix_tokens']}, W={ts[0]['suffix_width']})", flush=True)
        print(f"peak vram {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")


if __name__ == "__main__":
    main()
