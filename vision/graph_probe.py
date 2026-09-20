"""Can the Qwen3.5 language model (fla Triton + reference conv1d) be captured in a CUDA graph?"""
import sys, time, statistics as st, traceback
import torch, transformers
MODEL, REV = "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL, revision=REV, dtype=torch.bfloat16, device_map={"": "cuda:0"}, low_cpu_mem_usage=True).eval()
lm, head = model.model.language_model, model.lm_head

def make(B, L):
    emb = torch.randn(B, L, 2560, dtype=torch.bfloat16, device="cuda")
    pos = torch.arange(L, device="cuda").view(1, 1, L).expand(3, B, L).contiguous()
    mask = torch.ones(B, L, dtype=torch.long, device="cuda")
    return emb, pos, mask

def fwd(emb, pos, mask):
    h = lm(inputs_embeds=emb, position_ids=pos, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state[:, -1]
    return head(h)

def timeit(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return st.median(ts) * 1e3

with torch.inference_mode():
    for B, L in [(1, 340), (16, 48), (16, 340), (64, 48)]:
        emb, pos, mask = make(B, L)
        eager = timeit(lambda: fwd(emb, pos, mask))
        try:
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3): fwd(emb, pos, mask)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = fwd(emb, pos, mask)
            ref = fwd(emb, pos, mask); g.replay(); torch.cuda.synchronize()
            ok = torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)
            graph = timeit(g.replay)
            print(f"B={B:2d} L={L:3d} eager={eager:6.1f}ms  cudagraph={graph:6.1f}ms  match={ok}", flush=True)
        except Exception as e:
            print(f"B={B:2d} L={L:3d} eager={eager:6.1f}ms  CAPTURE FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
            traceback.print_exc(limit=3)
