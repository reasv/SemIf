"""Real-time pipeline: one 640x480 frame -> N typed decisions.

Per frame: CPU preprocess -> vision encoder (eager) -> ONE captured CUDA graph that
embeds the fixed prefix, scatters image features, prefills, replicates the cache to N
branches, runs the N criterion suffixes, and gathers the option logits.
Criteria are tokenized once at construction. Usage: python rt_pipeline.py [N ...]
"""
from __future__ import annotations
import sys, time, statistics as st
import os
import torch
from PIL import Image
from transformers import DynamicCache
_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "src")); sys.path.insert(0, _HERE)
from shared_image import load, SharedImageScorer, direct, frame, CRITERIA, full_prompt, DEV
from semif_phase1.core import softmax
from semif_phase1.direct import _slot_ids


class GraphPipeline:
    def __init__(self, proc, model, image_size, decisions):
        self.proc, self.model, self.tok = proc, model, proc.tokenizer
        self.lm, self.head = model.model.language_model, model.lm_head
        self.embed = self.lm.embed_tokens
        self.prefix_text = SharedImageScorer(proc, model).prefix_text
        dummy = Image.new("RGB", image_size, "gray")
        p = proc(text=[self.prefix_text], images=[dummy], return_tensors="pt")
        self.prefix_ids = p["input_ids"][0, :-1].to(DEV)
        self.grid = p["image_grid_thw"]
        self.pix = p["pixel_values"].to(DEV).clone()  # static input buffer
        P = self.prefix_ids.shape[0]
        self.image_mask = (self.prefix_ids == model.config.image_token_id)
        self.n_img = int(self.image_mask.sum())
        mm = p["mm_token_type_ids"][:, :-1] if "mm_token_type_ids" in p else None
        pos, delta = model.model.get_rope_index(self.prefix_ids[None].cpu(), image_grid_thw=self.grid,
                                                attention_mask=torch.ones(1, P, dtype=torch.long), mm_token_type_ids=mm)
        self.pos_prefix = pos.to(DEV).contiguous()  # (3,1,P) M-RoPE positions
        suffixes, self.slots = [], []
        for q, opts in decisions:
            text, _ = full_prompt(proc, q, opts)
            ids = proc(text=[text], images=[dummy], return_tensors="pt")["input_ids"][0]
            assert torch.equal(ids[:P], self.prefix_ids.cpu()), "prefix mismatch"
            suffixes.append(ids[P:].tolist()); self.slots.append(_slot_ids(self.tok, len(opts)))
        N, W = len(suffixes), max(map(len, suffixes)); self.N, self.W, self.P = N, W, P
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
        self.graph, self.out = None, None

    @torch.inference_mode()
    def _vision(self):
        out = self.model.model.get_image_features(self.pix, self.grid.to(DEV), return_dict=True)
        self.feats.copy_(torch.cat(out.pooler_output, 0))

    @torch.inference_mode()
    def _lm(self):
        emb = self.embed(self.prefix_ids[None]).masked_scatter(self.image_mask[None, :, None], self.feats)
        cache = DynamicCache(config=self.lm.config)
        self.lm(inputs_embeds=emb, position_ids=self.pos_prefix, past_key_values=cache, use_cache=True, return_dict=True)
        cache.reorder_cache(torch.zeros(self.N, dtype=torch.long, device=DEV))
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


def main():
    Ns = [int(a) for a in sys.argv[1:]] or [1, 4, 8, 16, 32, 64]
    proc, model = load()
    im = frame(640, 480)
    # correctness at N=16 against independent full forwards
    pipe = GraphPipeline(proc, model, (640, 480), CRITERIA); pipe.capture()
    worst_g, worst_e, agree = 0.0, 0.0, 0
    pg, _ = pipe.score(im, use_graph=True); pe, _ = pipe.score(im, use_graph=False)
    for (q, opts), a, b in zip(CRITERIA, pg, pe):
        d = direct(proc, model, im, q, opts)
        worst_g = max(worst_g, max(abs(x - y) for x, y in zip(a, d)))
        worst_e = max(worst_e, max(abs(x - y) for x, y in zip(b, d)))
        agree += (max(range(len(a)), key=a.__getitem__) == max(range(len(d)), key=d.__getitem__))
    print(f"CHECK N=16: graph-vs-direct worst prob diff={worst_g:.4f}  eager-pipeline-vs-direct={worst_e:.4f}  argmax agree {agree}/16"
          f"  (prefix={pipe.P} tokens incl {pipe.n_img} image, W={pipe.W})", flush=True)
    del pipe
    for n in Ns:
        dec = (CRITERIA * 8)[:n]
        pipe = GraphPipeline(proc, model, (640, 480), dec); pipe.capture()
        rows = {True: [], False: []}
        for g in (True, False):
            for i in range(15):
                _, t = pipe.score(im, use_graph=g)
                if i >= 3: rows[g].append(t)
        m = lambda g, k: st.median(r[k] for r in rows[g])
        print(f"N={n:3d}  graph: pre={m(True,'pre_ms'):5.1f} vision={m(True,'vision_ms'):5.1f} lm={m(True,'lm_ms'):6.1f} "
              f"total={m(True,'total_ms'):6.1f}ms per-decision={m(True,'total_ms')/n:5.1f}ms  |  eager lm={m(False,'lm_ms'):6.1f} total={m(False,'total_ms'):6.1f}ms",
              flush=True)
        del pipe; torch.cuda.empty_cache()
    print(f"peak vram {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")


if __name__ == "__main__":
    main()
