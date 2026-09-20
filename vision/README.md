# SemIf on images: typed decisions from video frames in real time

This directory adds image input and a real-time per-frame pipeline to SemIf, using the **same frozen
Qwen3.5-4B checkpoint and the same option-logit readout** as the shipped text paths. Nothing is trained.
On one RTX PRO 6000 Blackwell a 640x480 frame yields 16 decisions in about 85 ms, or one decision in
about 47 ms.

Target platform: Linux, one NVIDIA sm_120 GPU (RTX PRO 6000 / RTX 50xx), Python 3.12, the repo's pinned
torch 2.10.0+cu128 and transformers 5.17.0. Other platforms are untested and not a priority.

## Why it works

The checkpoint is `Qwen3_5ForConditionalGeneration`: a vision tower plus the language model. SemIf's
loader in `src/semif_phase1/core.py` deliberately loads only the text tower. The scripts here load the full
model with its processor, place an image content part before SemIf's JSON payload in the user turn, and
read the same single-token A/B/C logits at the last position. Because nothing was tuned for text, there is
no reason to expect the image path to lag the text path relative to the base model's own ability in each
modality; that is an inference, not a measurement (see Caveats).

## Setup

```bash
git clone https://github.com/reasv/SemIf && cd SemIf
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e '.[test]'
uv pip install --prerelease=allow -r vision/requirements-vision.txt
pytest -q                                   # 26 passed, 1 skipped (MLX)
```

Optional but recommended, the linear-attention conv kernel (otherwise transformers uses a correct but
~1.5 ms slower reference path and prints a warning):

```bash
# either: prebuilt wheel for torch 2.10.0+cu128 / Python 3.12 / linux x86_64, sm_70..sm_120
uv pip install https://github.com/reasv/SemIf/releases/download/kernels-v1/causal_conv1d-1.7.0-cp312-cp312-linux_x86_64.whl
# or: build it (needs gcc-14; assembles a private CUDA 12.8 toolchain, ~10 min)
bash vision/setup_kernels.sh
```

The model (`Qwen/Qwen3.5-4B` at revision `851bf6e8`, ~9.3 GB) downloads to the Hugging Face cache on
first run. Every script needs exactly one visible GPU: `CUDA_VISIBLE_DEVICES=0`.

Three environment switches are honored by all scripts through `vision/shared_image.py`:

| Variable | Values | Effect |
|---|---|---|
| `SEMIF_FMT` | `json` (default), `compact` | Prompt format; see the format experiment below. Use `compact`. |
| `SEMIF_ATTN` | unset (SDPA), `flash_attention_4` | Attention for the text model. FA4 is patched for sm_120 at import time by `vision/fa4_patch.py`. |
| `SEMIF_RESULTS` | a path | Where `rt_pipeline_static.py` writes its table. |

## Quick start

```bash
# 1. Does vision work at all? Eight synthetic images with known answers. Expect 8/8.
CUDA_VISIBLE_DEVICES=0 python vision/vision_probe.py

# 2. Real-time pipeline: correctness check vs full forwards, then a sweep over decisions per frame.
SEMIF_FMT=compact SEMIF_ATTN=flash_attention_4 CUDA_VISIBLE_DEVICES=0 python vision/rt_pipeline.py 1 8 16 32

# 3. Same, with a cached static text block before the image (rulebook / state schema).
SEMIF_FMT=compact SEMIF_ATTN=flash_attention_4 CUDA_VISIBLE_DEVICES=0 python vision/rt_pipeline_static.py quick
```

First calls take seconds (Triton and CuTe DSL compile, graph capture); the timings printed are warm medians.

## How a frame is processed

```
[system prompt][static text, optional][image][start of JSON]   <- prefix: once per frame (static part cached)
                                        + [criterion_i suffix]   <- N branches, one criterion each, batched
```

Per frame: CPU preprocess (3 ms) -> vision encoder, eager (14 ms) -> one captured CUDA graph that embeds the
prefix, scatters image features, prefills, replicates the cache to N branches, runs the N suffixes, and
gathers option logits (28 ms + about 2.6 ms per decision at the compact format's padded width). Each branch
sees only its own criterion; the batching is a throughput device, not a change to what the model reads.

Criteria are tokenized once at construction; changing them means recapturing the graph (about a second).
`rt_pipeline_static.py` additionally prefills the static text once at startup and copies its cache (linear-
attention conv and recurrent states copied; full-attention KV aliased) at the top of every frame's graph.

## Results (all in `vision/results/`)

Single decision, one image, eager, per stage (`single-decision-latency.txt`):

| Frame | Image tokens | Preprocess | Vision | LM | End to end |
|---|---:|---:|---:|---:|---:|
| 448x448 | 196 | 3.3 ms | 14.6 ms | 47.8 ms | 65.8 ms |
| 640x480 | 300 | 4.3 ms | 14.4 ms | 48.0 ms | 66.8 ms |
| 1280x720 | 880 | 16.7 ms | 21.6 ms | 50.4 ms | 88.9 ms |
| 1920x1080 | 2040 | 24.6 ms | 56.9 ms | 92.3 ms | 174.0 ms |

Decisions per frame, 640x480, compact format, graph replay (`pipeline-compact-format.txt`):

| Decisions | 1 | 8 | 16 | 24 | 32 | 48 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Per frame | 53 ms | 63 ms | 82 ms | 105 ms | 124 ms | 171 ms | 214 ms |

Cached static context before the image, N=16, FA4 (`static-prefix-fa4.txt`): 0 tokens 86 ms, 520 tokens
89 ms, 2010 tokens 94 ms. Recomputing the prefix each frame instead: 86 / 108 / 174 ms.

Correctness: shared-prefix and graph outputs match independent full forwards with a worst probability gap
of 0.02 to 0.03 (BF16 summation order) and 16/16 argmax agreement, at every static length; the static cache
is bit-identical after repeated frames.

### Prompt format experiment

SemIf's option list of `{"letter": "A", "description": ...}` objects was compared against a `{"A": ...,
"B": ...}` map (`compact`) and plain lettered lines (`plain`), same system prompt and readout, on the repo's
fixtures with the repo's evaluator and paired bootstrap (`format-comparison.txt`, row-level predictions in
`results/predictions/`):

| Fixture | json | compact | plain |
|---|---:|---:|---:|
| Authored 144, mean family balanced accuracy | 0.813 | **0.904** (+0.091 [+0.058, +0.134]) | 0.871 |
| Perturbations 108 | 0.766 | **0.830** (+0.064 [+0.010, +0.125]) | 0.816 |
| WANLI 256 | 0.625 | 0.648 | **0.679** (+0.055 [+0.005, +0.105]) |

The json baseline reproduces the published 0.8132 exactly on this hardware. Compact is shorter (a yes/no
criterion suffix is 40 tokens instead of 54) and more accurate; it is the recommended format here. It is a
different prompt from the one the repo's README claims are about, so those claims are not restated for it.

## Files

| File | Purpose |
|---|---|
| `vision_probe.py` | Eight synthetic images, known answers, full-VLM readout |
| `shared_image.py` | Prompt builders, image prefix reuse scorer, `direct()` full-forward reference, `load()` |
| `rt_pipeline.py` | Per-frame CUDA-graph pipeline; correctness check then a sweep over N |
| `rt_pipeline_static.py` | Same with a cached static text prefix; `cached` vs `recompute` benchmark |
| `fa4_patch.py` | Runtime fix for FA4's sm_120 tile config at head_dim 256 |
| `fmt/score_formats.py`, `fmt/compare.py` | Score any fixture under json/compact/plain; evaluate with the repo's evaluator |
| `vision_latency.py`, `vision_batch.py` | Per-stage latency across resolutions; naive batching sweep |
| `graph_probe.py`, `profile_suffix.py` | CUDA graph capture probe; kernel profile of the suffix shape |
| `setup_kernels.sh` | Build causal-conv1d against torch cu128 on a CUDA 13 system |

To re-run the format experiment on WANLI, fetch and build the fixture with the repo's scripts (the built
rows are not committed, following the repo's policy on third-party records):

```bash
python benchmarks/fetch_sources.py --output /path/to/sources
python benchmarks/build_wanli.py --source /path/to/sources/wanli-test.jsonl \
  --selection benchmarks/manifests/source-selection.jsonl --output vision/results/predictions/wanli256.jsonl
CUDA_VISIBLE_DEVICES=0 python vision/fmt/score_formats.py vision/results/predictions/wanli256.jsonl vision/results/predictions
python vision/fmt/compare.py
```

## Notes for sm_120 (consumer Blackwell)

- **FlashAttention-4** 4.0.0b31 has an sm_120 kernel, but its tile heuristic requests 128 KB of shared memory at
  head_dim 256 and this GPU allows 99 KB. `fa4_patch.py` switches to 64x64 tiles for head_dim > 128. Gain is
  bounded: about 11% of the frame at 32 decisions over 2000 tokens of context, nothing at small N.
- **FlashAttention-2** ships sm_120 wheels only up to torch 2.9; this repo pins 2.10, hence FA4.
- **causal-conv1d** has no wheel for torch 2.10 / cu128. The PyPI sdist builds against the system CUDA; on a
  CUDA 13 system that produces a library that fails to load. `setup_kernels.sh` handles it.
- **transformers' linear-attention path** without `flash-linear-attention` and `causal-conv1d` is a reference
  implementation; install both or the LM floor is ~62 ms instead of ~50 ms eager.
- Serving engines were assessed (vLLM 0.29, SGLang 0.5.20, Sept 2026): both load the model and return
  per-token-ID logprobs without decoding, but neither caches the linear-attention state across frames at
  arbitrary prefix lengths with CUDA-graph prefill for a VLM. This pipeline stays hand-rolled for now.

## Caveats

- Every number above is latency. **No accuracy number on real frames exists.** The eight probe images are
  trivially easy. Any intended use needs a few hundred labeled frames scored with `benchmarks/evaluate.py`.
- Probabilities are option scores, not calibrated confidences (as the repo states); thresholds per criterion
  must come from labeled data.
- Measurements were taken with the GPU otherwise idle; another process rendering on the same GPU roughly
  doubles every stage.
- The shipped text paths under `src/`, the benchmarks, and `results/` are unchanged, and the repo's own checks
  (`pytest`, the SHA-256 manifest, `verify_published.py`) still pass.
