"""Top CUDA kernels for the batched suffix shape (B=16, L=48) and prefill shape (B=1, L=340)."""
import torch, transformers
from torch.profiler import profile, ProfilerActivity
MODEL, REV = "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL, revision=REV, dtype=torch.bfloat16, device_map={"": "cuda:0"}, low_cpu_mem_usage=True).eval()
lm = model.model.language_model
def run(B, L):
    emb = torch.randn(B, L, 2560, dtype=torch.bfloat16, device="cuda")
    pos = torch.arange(L, device="cuda").view(1, 1, L).expand(3, B, L).contiguous()
    with torch.inference_mode():
        for _ in range(3): lm(inputs_embeds=emb, position_ids=pos, use_cache=False)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(5): lm(inputs_embeds=emb, position_ids=pos, use_cache=False)
            torch.cuda.synchronize()
    print(f"\n=== B={B} L={L} (5 iters) ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12, max_name_column_width=70))
run(16, 48); run(1, 340)
