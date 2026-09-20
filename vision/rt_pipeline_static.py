"""Real-time pipeline with a cached static text prefix (variant of rt_pipeline.py).

Layout per branch: [system][user: static_context text][<vision_start> image <vision_end>][JSON start][decision suffix_i].
The static text is a text content part placed BEFORE the image part of the user turn, so SemIf's system prompt and
the JSON payload after the image stay byte-identical to shared_image.full_prompt (static_context="" reproduces it).

cache_static=True (default): at construction, [system][static text] is prefilled ONCE (eager) into a DynamicCache.
For this model (Qwen3.5-4B: 24 linear_attention + 8 full_attention layers) that cache holds, per layer:
  - LinearAttentionLayer: conv_states[0] (B, 2*key_dim+value_dim, conv_kernel=4) and recurrent_states[0]
    (B, num_v_heads, head_k_dim, head_v_dim). Both are mutated IN PLACE by later forwards (`.copy_`), so the
    working cache gets its own preallocated buffers, filled with copy_ inside the graph.
  - DynamicLayer: keys/values (B, kv_heads, S, head_dim). These are only ever rebound (torch.cat / index_select),
    never written in place, so the working cache aliases the static tensors (documented + verified by a snapshot check).
Per frame, the captured CUDA graph: copy static cache -> prefill only the image segment (vision_start + image
tokens + vision_end + JSON start) with M-RoPE positions sliced from get_rope_index on the full prefix -> reorder_cache
to N branches -> N suffixes -> gather option logits.
cache_static=False: rt_pipeline.py's approach with the static text folded into the per-frame prefix (baseline).

Usage: SEMIF_FMT=compact CUDA_VISIBLE_DEVICES=1 .venv/bin/python vision/rt_pipeline_static.py [quick]
"""
from __future__ import annotations
import json, os, random, sys, time, statistics as st
import os
import torch
from PIL import Image
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer, LinearAttentionCacheLayerMixin
_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "src")); sys.path.insert(0, _HERE)
import shared_image
from shared_image import load, frame, CRITERIA, EVIDENCE, DEV
from semif_phase1.core import DIRECT_SYSTEM, LETTERS, softmax
from semif_phase1.direct import _slot_ids

RESULTS = os.environ.get("SEMIF_RESULTS", os.path.join(_HERE, "results", "local", "rt_pipeline_static_results.txt"))


# ----------------------------------------------------------------------------- prompts
def normalize_static(static: str) -> str:
    static = (static or "").strip()
    return static + "\n" if static else ""


def full_prompt(proc, question, options, static=""):
    """shared_image.full_prompt with an optional text part before the image part (static == "" -> identical)."""
    if os.environ.get("SEMIF_FMT", "json") == "compact":
        payload = {"evidence": EVIDENCE, "criterion": question,
                   "options": {LETTERS[i]: d for i, d in enumerate(options)}}
    else:
        payload = {"evidence": EVIDENCE, "criterion": question,
                   "options": [{"letter": LETTERS[i], "description": d} for i, d in enumerate(options)]}
    parts = ([{"type": "text", "text": static}] if static else []) + \
            [{"type": "image"}, {"type": "text", "text": json.dumps(payload)}]
    msgs = [{"role": "system", "content": DIRECT_SYSTEM}, {"role": "user", "content": parts}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return text, json.dumps(payload)


def prefix_prompt(proc, static=""):
    text, payload = full_prompt(proc, "prefix boundary placeholder", ["Yes", "No"], static)
    assert text.count(payload) == 1
    evidence = json.dumps({"evidence": EVIDENCE})[:-1]
    assert payload.startswith(evidence)
    return text[: text.index(payload)] + evidence


@torch.inference_mode()
def direct(proc, model, image, q, opts, static=""):
    """Independent full forward over the whole prompt (static text included)."""
    text, _ = full_prompt(proc, q, opts, static)
    inp = proc(text=[text], images=[image], return_tensors="pt").to(DEV)
    logits = model(**inp, use_cache=False, return_dict=True, logits_to_keep=1).logits[0, -1].float()
    return softmax(logits[_slot_ids(proc.tokenizer, len(opts))].cpu().tolist())


def make_rules(tok, n_tokens: int, seed: int = 0) -> str:
    """Deterministic filler: numbered support-triage policy statements of ~n_tokens tokens."""
    if n_tokens <= 0:
        return ""
    rng = random.Random(seed)
    subj = ["Tickets that mention a refund", "Messages received outside business hours", "Requests containing an invoice number",
            "Tickets with an attached screenshot", "Customers on an annual plan", "Reports of a duplicate charge",
            "Password reset requests", "Tickets flagged as urgent by the sender", "Messages written in a language other than English",
            "Complaints about response time", "Requests to cancel a subscription", "Tickets referencing a prior case",
            "Questions about pricing tiers", "Reports of a failed login", "Tickets sent from a mobile device"]
    act = ["must be routed to the billing queue", "are escalated to a tier-2 agent within {n} hours",
           "require a manager acknowledgement before any reply is sent", "are answered with the standard template {t}",
           "must be tagged with the reporting month before closure", "are held until the identity of the sender is verified",
           "should be merged with any open case from the same account", "are assigned a priority of {p} on the shared board",
           "must not be closed without a written summary of the resolution", "are reviewed in the weekly quality audit",
           "may be resolved by the first responder without approval", "trigger an automatic acknowledgement within {n} minutes"]
    qual = ["unless the amount is below ${a}", "when the account is less than {n} days old", "if the sender has replied more than {n} times",
            "except during a declared incident", "regardless of the channel used", "only after the payment record has been checked",
            "when the ticket is older than {n} hours", "if no other rule applies"]
    head = "Operating policy for support triage. Apply these rules only where they are relevant to the criterion.\n"
    lines, i, n = [], 1, 0
    while n < n_tokens:
        s = rng.choice(subj); a = rng.choice(act); q = rng.choice(qual)
        fmt = dict(n=rng.choice([2, 4, 8, 12, 24, 48]), t=rng.choice(["B-1", "R-7", "A-3", "S-9"]), p=rng.choice(["P1", "P2", "P3"]),
                   a=rng.choice([10, 25, 50, 100, 250]))
        lines.append(f"{i}. {s} {a.format(**fmt)} {q.format(**fmt)}.")
        i += 1
        n = len(tok.encode(head + "\n".join(lines), add_special_tokens=False))
    return head + "\n".join(lines)


# ----------------------------------------------------------------------------- pipeline
class GraphPipeline:
    def __init__(self, proc, model, image_size, decisions, static_context: str = "", cache_static: bool = True):
        self.proc, self.model, self.tok = proc, model, proc.tokenizer
        self.lm, self.head = model.model.language_model, model.lm_head
        self.embed = self.lm.embed_tokens
        self.static, self.cache_static = normalize_static(static_context), cache_static
        self.prefix_text = prefix_prompt(proc, self.static)
        dummy = Image.new("RGB", image_size, "gray")
        p = proc(text=[self.prefix_text], images=[dummy], return_tensors="pt")
        self.prefix_ids = p["input_ids"][0, :-1].to(DEV)  # drop boundary token (may merge with following JSON)
        self.grid = p["image_grid_thw"]
        self.pix = p["pixel_values"].to(DEV).clone()  # static input buffer
        P = self.prefix_ids.shape[0]
        # static / image boundary = the single <|vision_start|> token
        vs = (self.prefix_ids == model.config.vision_start_token_id).nonzero().flatten()
        assert vs.numel() == 1, "expected exactly one image in the prefix"
        S = int(vs[0]); self.S, self.P = S, P
        self.static_ids, self.img_ids = self.prefix_ids[:S], self.prefix_ids[S:]
        self.image_mask_full = (self.prefix_ids == model.config.image_token_id)
        assert not self.image_mask_full[:S].any()
        self.image_mask = self.image_mask_full[S:]
        self.n_img = int(self.image_mask.sum())
        mm = p["mm_token_type_ids"][:, :-1] if "mm_token_type_ids" in p else None
        pos, delta = model.model.get_rope_index(self.prefix_ids[None].cpu(), image_grid_thw=self.grid,
                                                attention_mask=torch.ones(1, P, dtype=torch.long), mm_token_type_ids=mm)
        self.pos_prefix = pos.to(DEV).contiguous()  # (3,1,P) M-RoPE positions of the full prefix
        self.pos_static = self.pos_prefix[:, :, :S].contiguous()
        self.pos_img = self.pos_prefix[:, :, S:].contiguous()
        ar = torch.arange(S, device=DEV)
        assert all(torch.equal(self.pos_static[k, 0], ar) for k in range(3)), "static text positions must be 0..S-1"
        suffixes, self.slots = [], []
        for q, opts in decisions:
            text, _ = full_prompt(proc, q, opts, self.static)
            ids = proc(text=[text], images=[dummy], return_tensors="pt")["input_ids"][0]
            assert torch.equal(ids[:P], self.prefix_ids.cpu()), "prefix mismatch"
            suffixes.append(ids[P:].tolist()); self.slots.append(_slot_ids(self.tok, len(opts)))
        N, W = len(suffixes), max(map(len, suffixes)); self.N, self.W = N, W
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        sfx = torch.full((N, W), pad, dtype=torch.long); pos2 = torch.zeros((N, W), dtype=torch.long); ends = []
        for i, s in enumerate(suffixes):
            L = len(s); sfx[i, :L] = torch.tensor(s); pos2[i, :L] = torch.arange(P, P + L); ends.append(L - 1)
        self.sfx, self.ends = sfx.to(DEV), torch.tensor(ends, device=DEV)
        self.pos_sfx = (pos2.to(DEV) + delta.to(DEV)).unsqueeze(0).expand(3, N, W).contiguous()
        maxopt = max(map(len, self.slots))
        idx = torch.zeros((N, maxopt), dtype=torch.long)
        for i, s in enumerate(self.slots): idx[i, :len(s)] = torch.tensor(s)
        self.slot_idx = idx.to(DEV)
        self.feats = torch.zeros((self.n_img, model.config.text_config.hidden_size), dtype=torch.bfloat16, device=DEV)
        self.branch_idx = torch.zeros(self.N, dtype=torch.long, device=DEV)
        self.graph, self.out = None, None
        self.static_cache = None
        if cache_static:
            self._build_static_cache()

    # --- static cache -----------------------------------------------------------------
    @torch.inference_mode()
    def _build_static_cache(self):
        cache = DynamicCache(config=self.lm.config)
        self.lm(inputs_embeds=self.embed(self.static_ids[None]), position_ids=self.pos_static,
                past_key_values=cache, use_cache=True, return_dict=True)
        assert cache.get_seq_length() == self.S
        self.work_conv, self.work_rec, snap = {}, {}, []
        for i, layer in enumerate(cache.layers):
            if isinstance(layer, LinearAttentionCacheLayerMixin):
                assert layer.number_of_states == 1 and layer.has_previous_state[0]
                assert layer.is_conv_states_initialized[0] and layer.is_recurrent_states_initialized[0]
                self.work_conv[i] = torch.empty_like(layer.conv_states[0])
                self.work_rec[i] = torch.empty_like(layer.recurrent_states[0])
                snap.append((layer.conv_states[0].clone(), layer.recurrent_states[0].clone()))
            else:
                assert isinstance(layer, DynamicLayer) and layer.get_seq_length() == self.S
                snap.append((layer.keys.clone(), layer.values.clone()))
        torch.cuda.synchronize()
        self.static_cache, self._snapshot = cache, snap

    def describe_cache(self) -> str:
        c = self.static_cache; lin = [l for l in c.layers if isinstance(l, LinearAttentionCacheLayerMixin)]
        full = [l for l in c.layers if isinstance(l, DynamicLayer)]
        return (f"static cache: {len(lin)} {type(lin[0]).__name__} layers "
                f"[conv_states{tuple(lin[0].conv_states[0].shape)} {lin[0].conv_states[0].dtype}, "
                f"recurrent_states{tuple(lin[0].recurrent_states[0].shape)} {lin[0].recurrent_states[0].dtype}] + "
                f"{len(full)} {type(full[0]).__name__} layers [keys/values{tuple(full[0].keys.shape)} {full[0].keys.dtype}]; "
                f"get_seq_length()={c.get_seq_length()}")

    def static_cache_intact(self) -> bool:
        """True iff the static cache tensors are bit-identical to their post-prefill snapshot (aliasing check)."""
        torch.cuda.synchronize()
        for layer, (a, b) in zip(self.static_cache.layers, self._snapshot):
            if isinstance(layer, LinearAttentionCacheLayerMixin):
                if not (torch.equal(layer.conv_states[0], a) and torch.equal(layer.recurrent_states[0], b)): return False
            elif not (torch.equal(layer.keys, a) and torch.equal(layer.values, b)): return False
        return True

    def _working_cache(self) -> DynamicCache:
        """Fresh DynamicCache whose state equals the static cache: linear-attention states copied into preallocated
        buffers (they are updated in place by the model), full-attention K/V aliased (only ever rebound by cat)."""
        src, dst = self.static_cache, DynamicCache(config=self.lm.config)
        for i, (s, d) in enumerate(zip(src.layers, dst.layers)):
            if isinstance(s, LinearAttentionCacheLayerMixin):
                d.dtype, d.device = s.dtype, s.device
                d.conv_kernel_size[0] = s.conv_kernel_size[0]
                d.conv_states[0] = self.work_conv[i].copy_(s.conv_states[0])
                d.recurrent_states[0] = self.work_rec[i].copy_(s.recurrent_states[0])
                d.is_conv_states_initialized[0] = d.is_recurrent_states_initialized[0] = d.has_previous_state[0] = True
            else:
                d.lazy_initialization(s.keys, s.values)
                d.keys, d.values = s.keys, s.values
        return dst

    # --- per-frame stages --------------------------------------------------------------
    @torch.inference_mode()
    def _vision(self):
        out = self.model.model.get_image_features(self.pix, self.grid.to(DEV), return_dict=True)
        self.feats.copy_(torch.cat(out.pooler_output, 0))

    @torch.inference_mode()
    def _lm(self):
        if self.cache_static:
            cache = self._working_cache()  # past_seen_tokens == S
            emb = self.embed(self.img_ids[None]).masked_scatter(self.image_mask[None, :, None], self.feats)
            self.lm(inputs_embeds=emb, position_ids=self.pos_img, past_key_values=cache, use_cache=True, return_dict=True)
        else:
            cache = DynamicCache(config=self.lm.config)
            emb = self.embed(self.prefix_ids[None]).masked_scatter(self.image_mask_full[None, :, None], self.feats)
            self.lm(inputs_embeds=emb, position_ids=self.pos_prefix, past_key_values=cache, use_cache=True, return_dict=True)
        cache.reorder_cache(self.branch_idx)
        h = self.lm(inputs_embeds=self.embed(self.sfx), position_ids=self.pos_sfx, past_key_values=cache,
                    use_cache=True, return_dict=True).last_hidden_state
        logits = self.head(h[torch.arange(self.N, device=DEV), self.ends]).float()
        return torch.gather(logits, 1, self.slot_idx)

    def capture(self):
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): self._lm()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self._lm()

    @torch.inference_mode()
    def score(self, image, use_graph=True):
        t0 = time.perf_counter()
        pv = self.proc.image_processor(images=[image], return_tensors="pt")["pixel_values"]
        assert pv.shape == self.pix.shape
        self.pix.copy_(pv, non_blocking=False)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        self._vision()
        torch.cuda.synchronize(); t2 = time.perf_counter()
        if use_graph: self.graph.replay(); out = self.out
        else: out = self._lm()
        torch.cuda.synchronize(); t3 = time.perf_counter()
        sel = out.cpu().tolist()
        probs = [softmax(sel[i][:len(self.slots[i])]) for i in range(self.N)]
        return probs, {"pre_ms": (t1-t0)*1e3, "vision_ms": (t2-t1)*1e3, "lm_ms": (t3-t2)*1e3, "total_ms": (t3-t0)*1e3}


# ----------------------------------------------------------------------------- driver
def _cmp(a, b): return max(abs(x - y) for x, y in zip(a, b))
def _argmax(v): return max(range(len(v)), key=v.__getitem__)


def _capture(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    return g


def _replay_ms(g, n=12):
    xs = []
    for _ in range(n):
        torch.cuda.synchronize(); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize(); xs.append(a.elapsed_time(b))
    return st.median(xs[2:])


def _attribution(pipe):
    """Increments: [cache copy, image-segment prefill, reorder_cache, suffixes+head] for the cached path."""
    @torch.inference_mode()
    def stage(k):
        c = pipe._working_cache()
        if k >= 1:
            emb = pipe.embed(pipe.img_ids[None]).masked_scatter(pipe.image_mask[None, :, None], pipe.feats)
            pipe.lm(inputs_embeds=emb, position_ids=pipe.pos_img, past_key_values=c, use_cache=True, return_dict=True)
        if k >= 2: c.reorder_cache(pipe.branch_idx)
    ts = [_replay_ms(_capture(lambda k=k: stage(k))) for k in range(3)] + [_replay_ms(_capture(pipe._lm))]
    return [ts[0]] + [ts[i] - ts[i - 1] for i in range(1, 4)]


def main():
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    lines = []
    def log(s=""):
        print(s, flush=True); lines.append(s)
    proc, model = load(); tok = proc.tokenizer
    im = frame(640, 480); im2 = frame(640, 480).transpose(Image.FLIP_LEFT_RIGHT)
    # static_context="" must reproduce the exact SemIf/shared_image template
    for q, opts in CRITERIA[:2]:
        assert full_prompt(proc, q, opts, "") == shared_image.full_prompt(proc, q, opts)
    targets = [0, 2000] if quick else [0, 500, 2000]
    statics = {n: make_rules(tok, n) for n in targets}
    ntok = {n: len(tok.encode(normalize_static(s), add_special_tokens=False)) for n, s in statics.items()}
    log(f"SEMIF_FMT={os.environ.get('SEMIF_FMT', 'json')}  static_context targets={targets} actual tokens={ntok}")
    log("layout: [system][user: static text part][image part][JSON payload part] -> shared JSON start, per-decision suffix")

    # ---- correctness ------------------------------------------------------------------
    for n in targets:
        static = statics[n]
        pipe = GraphPipeline(proc, model, (640, 480), CRITERIA, static_context=static, cache_static=True); pipe.capture()
        if n == targets[-1]: log(pipe.describe_cache())
        refs = [direct(proc, model, im, q, opts, normalize_static(static)) for q, opts in CRITERIA]
        pg, _ = pipe.score(im, use_graph=True); pe, _ = pipe.score(im, use_graph=False)
        wg = max(_cmp(a, d) for a, d in zip(pg, refs)); we = max(_cmp(b, d) for b, d in zip(pe, refs))
        wge = max(_cmp(a, b) for a, b in zip(pg, pe))
        agree = sum(_argmax(a) == _argmax(d) for a, d in zip(pg, refs))
        log(f"CHECK cached static={ntok[n]:4d}tok S={pipe.S} P={pipe.P} (img {pipe.n_img}) W={pipe.W} N=16: "
            f"graph-vs-direct worst prob diff={wg:.4f}  eager-vs-direct={we:.4f}  graph-vs-eager={wge:.4f}  "
            f"argmax agree {agree}/16  {'PASS' if wg < 0.03 and agree == 16 else 'FAIL'}")
        # aliasing test: another frame, then the original again; static cache must be untouched and results reproduce
        pipe.score(im2, use_graph=True); pipe.score(im2, use_graph=False)
        pg2, _ = pipe.score(im, use_graph=True); pe2, _ = pipe.score(im, use_graph=False)
        rep = max(max(_cmp(a, b) for a, b in zip(pg, pg2)), max(_cmp(a, b) for a, b in zip(pe, pe2)))
        wg2 = max(_cmp(a, d) for a, d in zip(pg2, refs))
        log(f"      after 2 more frames: static cache intact={pipe.static_cache_intact()}  "
            f"repeat-frame max prob diff={rep:.6f}  graph-vs-direct={wg2:.4f}")
        if n == targets[-1]:  # baseline (recompute) path on the same prompt for reference
            del pipe; torch.cuda.empty_cache()
            pipe = GraphPipeline(proc, model, (640, 480), CRITERIA, static_context=static, cache_static=False); pipe.capture()
            pr, _ = pipe.score(im, use_graph=True)
            wr = max(_cmp(a, d) for a, d in zip(pr, refs)); ar = sum(_argmax(a) == _argmax(d) for a, d in zip(pr, refs))
            log(f"CHECK recompute static={ntok[n]:4d}tok N=16: graph-vs-direct worst prob diff={wr:.4f}  argmax agree {ar}/16")
        del pipe; torch.cuda.empty_cache()

    # ---- benchmark --------------------------------------------------------------------
    Ns = [1, 32] if quick else [1, 16, 32]
    frames, rounds = (8, 1) if quick else (13, 2)
    log(""); log(f"BENCH ms per frame: median of {frames - 3} warm frames, min over {rounds} rounds "
                 "(lm = LM stage: cache copy + image-segment prefill + reorder + suffixes + head; total = pre + vision + lm)")
    hdr = f"{'static':>7} {'P':>5} {'N':>3} {'mode':<9} | {'graph lm':>9} {'graph total':>11} | {'eager lm':>9} {'eager total':>11} | {'pre':>5} {'vision':>6}"
    log(hdr); log("-" * len(hdr))
    for n in targets:
        for N in Ns:
            dec = (CRITERIA * 8)[:N]
            for cache_static in (True, False):
                pipe = GraphPipeline(proc, model, (640, 480), dec, static_context=statics[n], cache_static=cache_static); pipe.capture()
                best = {}
                for _ in range(rounds):
                    rows = {True: [], False: []}
                    for g in (True, False):
                        for i in range(frames):
                            _, t = pipe.score(im, use_graph=g)
                            if i >= 3: rows[g].append(t)
                    for g in (True, False):
                        for k in ("lm_ms", "total_ms", "pre_ms", "vision_ms"):
                            v = st.median(r[k] for r in rows[g]); best[g, k] = min(v, best.get((g, k), v))
                m = lambda g, k: best[g, k]
                log(f"{ntok[n]:>7} {pipe.P:>5} {N:>3} {'cached' if cache_static else 'recompute':<9} | "
                    f"{m(True,'lm_ms'):>9.1f} {m(True,'total_ms'):>11.1f} | {m(False,'lm_ms'):>9.1f} {m(False,'total_ms'):>11.1f} | "
                    f"{m(True,'pre_ms'):>5.1f} {m(True,'vision_ms'):>6.1f}")
                del pipe; torch.cuda.empty_cache()

    # ---- attribution of the cached LM stage (partial CUDA graphs, CUDA-event timing) ---
    log(""); log("ATTRIBUTION of cached graph LM stage (ms, CUDA events, median of 10 replays): partial graphs capturing "
                 "successively more of the stage; each column is the increment over the previous one")
    for n in targets:
        for N in ([1, 32] if quick else [1, 16, 32]):
            pipe = GraphPipeline(proc, model, (640, 480), (CRITERIA * 8)[:N], static_context=statics[n], cache_static=True)
            pipe._vision()
            parts = _attribution(pipe)
            log(f"static={ntok[n]:4d} S={pipe.S:4d} N={N:2d}: cache copy={parts[0]:5.2f}  image-segment prefill={parts[1]:6.2f}  "
                f"reorder_cache={parts[2]:5.2f}  suffixes+head={parts[3]:6.2f}  total={sum(parts):6.2f}")
            del pipe; torch.cuda.empty_cache()
    log(f"peak vram {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
