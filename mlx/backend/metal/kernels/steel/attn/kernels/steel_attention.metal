// Copyright © 2024-25 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"

#define instantiate_attn(tname, dtype, bq, bk, bd, wm, wn, mname, mtype) \
  instantiate_kernel(                                                    \
      "steel_attention_" #tname "_bq" #bq "_bk" #bk "_bd" #bd            \
      "_wm" #wm "_wn" #wn "_mask" #mname,                                \
  attention, dtype, bq, bk, bd, wm, wn, mtype, float)

// BD=512 instantiation attempted (BQ=16/BK=8/WM=2/WN=1 + padQ=0) on
// 2026-05-16 — compiled and ran functionally but A/B at 8K Gemma-4-26B
// full-attn showed -25% prefill regression vs the matmul fallback. Cause:
// smaller per-TG parallelism (64 vs 128 threads, WM=2 instead of 4) and
// padQ=0 bank conflicts overwhelm the [B,H,L,L] scores tensor saving.
// Reverted; full-attn falls back to matmul+softmax+matmul which is faster
// on this hardware. Re-investigate if NAX-capable hardware (M5+) tips
// the balance — NAX UD=32 fragments could close the per-TG gap.
#define instantiate_attn_shapes_helper(iname, itype, mname, mtype)  \
    instantiate_attn(iname, itype, 32, 16, 256, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 16, 128, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 32,  80, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 32,  64, 4, 1, mname, mtype)

#define instantiate_attn_mask_helper(iname, itype) \
    instantiate_attn_shapes_helper(iname, itype, iname, itype) \
    instantiate_attn_shapes_helper(iname, itype, bool_, bool)

instantiate_attn_mask_helper(float16, half);
instantiate_attn_mask_helper(bfloat16, bfloat16_t);

instantiate_attn_mask_helper(float32, float);
// clang-format on
