"""Runtime fix for flash-attn-4 on sm_120 (consumer Blackwell) at head_dim 256.

FA4 4.0.0b31's sm_120 tile heuristic picks 128x64 tiles for any head_dim > 64. At head_dim 256 that
needs Q 64 KB + K 32 KB + V 32 KB = 128 KB of shared memory; sm_120 allows ~99 KB, so the launch fails
with CUDA_LAUNCH_INVALID_CONFIG. 64x64 tiles need 96 KB and work. Qwen3.5 uses head_dim 256.
Call apply() before the first attention call; shared_image.load() does this when SEMIF_ATTN=flash_attention_4.
"""


def apply() -> bool:
    try:
        from flash_attn.cute import interface as fa
    except ImportError:
        return False
    if getattr(fa, "_semif_sm120_patched", False):
        return True
    original = fa._get_fwd_config

    def patched(*, arch, head_dim, tile_mn=None, **kw):
        if tile_mn is None and arch // 10 == 12 and head_dim > 128:
            tile_mn = (64, 64)
        return original(arch=arch, head_dim=head_dim, tile_mn=tile_mn, **kw)

    fa._get_fwd_config = patched
    fa._semif_sm120_patched = True
    return True
