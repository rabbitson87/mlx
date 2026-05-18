// lumen-rs Phase 1.8 M4.8 — Metal backend impl for lumen::FlashAttnBf16.
//
// Encodes the bf16 flash-attention kernel into mlx's per-stream compute
// encoder so the kernel becomes a lazy graph node (mlx's scheduler decides
// commit/wait timing). This removes the per-call eval-sync that the M4.7
// bridge path was paying (~12 ms / step at PROMPT_LEN=4096).
//
// Shader source ported from lumen-rs
// `crates/turboquant-mlx/src/native_metal_bridge.rs::SHADER_SRC` — only the
// flash_attn_bf16 (FA-2, prefill) + sdpa_vector_bf16 (decode-specialized)
// kernels needed by `eval_gpu`. Everything else stays in the bridge.

#include <stdexcept>
#include <string>

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/lumen_primitives.h"

namespace mlx::core::lumen {

namespace {

const std::string& lumen_lib_name() {
  static const std::string name = "lumen_flash_attn";
  return name;
}

constexpr const char* SHADER_SRC = R"KMETAL(
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

// ── bf16 Flash Attention 2, GQA in-kernel, additive mask ────────────────
//
// Ported from lumen-rs `native_metal_bridge.rs::SHADER_SRC::flash_attn_bf16`.
// Identical algorithm; only the registration path differs (mlx primitive
// instead of bridge dispatch).
//
// Buffers:
//   [0] Q      [B, H,    Sq,  D]  bf16
//   [1] K      [B, H_kv, Skv, D]  bf16
//   [2] V      [B, H_kv, Skv, D]  bf16
//   [3] O      [B, H,    Sq,  D]  bf16
//   [4] mask   [Sq, Skv]          bf16  (additive; ignored when has_mask==0,
//                                       but bound to a placeholder buffer)
//   [5..11]    B / H / Sq / Skv (u32), scale (f32), has_mask (u32), group (u32)

constant uint TFA_D     = 256;
constant uint TFA_BLOCK = 8;
constant uint TFA_SG    = 32;
constant uint TFA_NSG   = TFA_D / TFA_SG;  // 8

kernel void lumen_flash_attn_bf16(
    device const bfloat* __restrict__ Q    [[buffer(0)]],
    device const bfloat* __restrict__ K    [[buffer(1)]],
    device const bfloat* __restrict__ V    [[buffer(2)]],
    device       bfloat*              O    [[buffer(3)]],
    device const bfloat* __restrict__ mask [[buffer(4)]],
    constant uint&   B_val    [[buffer(5)]],
    constant uint&   H_val    [[buffer(6)]],
    constant uint&   Sq_val   [[buffer(7)]],
    constant uint&   Skv_val  [[buffer(8)]],
    constant float&  scale    [[buffer(9)]],
    constant uint&   has_mask [[buffer(10)]],
    constant uint&   group    [[buffer(11)]],
    // Phase 1.x M4.8 strided-cache fix: head-stride (axis-1) in elements
    // for K and V. For a contiguous [B, H_kv, Skv, D] = Skv*D. For a
    // NativeKvCache slice view [B, H_kv, Skv, D] over a buffer with
    // capacity (≥Skv) on axis-2 = capacity*D. Axis-2 stride is always D
    // (innermost data dim is contiguous) so no per-row stride arg needed.
    constant uint&   k_h_stride [[buffer(12)]],
    constant uint&   v_h_stride [[buffer(13)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint d    [[thread_position_in_threadgroup]]
) {
    uint linear = tgid;
    uint qi = linear % Sq_val;
    uint h  = (linear / Sq_val) % H_val;
    uint b  = linear / (Sq_val * H_val);
    if (b >= B_val) return;

    uint H_kv  = H_val / group;
    uint h_kv  = h / group;

    uint q_off   = (b * H_val * Sq_val  + h    * Sq_val  + qi) * TFA_D;
    uint k_base  = (b * H_kv + h_kv) * k_h_stride;
    uint v_base  = (b * H_kv + h_kv) * v_h_stride;

    threadgroup float tg_q   [256];
    threadgroup float tg_k   [8][256];
    threadgroup float tg_v   [8][256];
    threadgroup float tg_s   [8];
    threadgroup float tg_sg  [8][8];
    threadgroup float tg_stats[3];
    threadgroup float tg_o   [256];

    tg_q[d] = float(Q[q_off + d]);
    tg_o[d] = 0.0f;
    if (d == 0) {
        tg_stats[0] = -HUGE_VALF;
        tg_stats[1] = 0.0f;
        tg_stats[2] = 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sg_id = d / TFA_SG;
    uint sg_d  = d % TFA_SG;

    for (uint kv_start = 0; kv_start < Skv_val; kv_start += TFA_BLOCK) {
        uint blk = min(TFA_BLOCK, Skv_val - kv_start);

        for (uint j = 0; j < TFA_BLOCK; j++) {
            tg_k[j][d] = (j < blk)
                ? float(K[k_base + (kv_start + j) * TFA_D + d])
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = 0; j < TFA_BLOCK; j++) {
            float contrib = tg_q[d] * tg_k[j][d];
            float sg_sum  = simd_sum(contrib);
            if (sg_d == 0) tg_sg[sg_id][j] = sg_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (d < TFA_BLOCK) {
            float s = 0.0f;
            for (uint sg = 0; sg < TFA_NSG; sg++) s += tg_sg[sg][d];
            s *= scale;
            if (has_mask && d < blk) {
                s += float(mask[qi * Skv_val + kv_start + d]);
            }
            tg_s[d] = (d < blk) ? s : (-HUGE_VALF);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (d == 0) {
            float m_old = tg_stats[0];
            float m_new = m_old;
            for (uint j = 0; j < blk; j++) m_new = max(m_new, tg_s[j]);

            float corr    = exp(m_old - m_new);
            float l_delta = 0.0f;
            for (uint j = 0; j < blk; j++) {
                float p   = exp(tg_s[j] - m_new);
                tg_s[j]   = p;
                l_delta  += p;
            }
            tg_stats[0] = m_new;
            tg_stats[1] = tg_stats[1] * corr + l_delta;
            tg_stats[2] = corr;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        tg_o[d] *= tg_stats[2];

        for (uint j = 0; j < TFA_BLOCK; j++) {
            tg_v[j][d] = (j < blk)
                ? float(V[v_base + (kv_start + j) * TFA_D + d])
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float acc = 0.0f;
        for (uint j = 0; j < blk; j++) acc += tg_s[j] * tg_v[j][d];
        tg_o[d] += acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    O[q_off + d] = bfloat(tg_o[d] / tg_stats[1]);
}

// ── bf16 SDPA-vector — decode-specialized (Sq=1, KV-parallel) ──────────
//
// 1024 threads/TG = 32 simdgroups × 32 lanes. Each thread holds 8 q + 8 o
// in registers; per-simdgroup independent online softmax; cross-simdgroup
// merge at the end. Same buffer signature as lumen_flash_attn_bf16.

constant uint TSD_D = 256;
constant uint TSD_BD = 32;
constant uint TSD_BN = 32;
constant uint TSD_PER = 8;

kernel void lumen_sdpa_vector_bf16(
    device const bfloat* __restrict__ Q    [[buffer(0)]],
    device const bfloat* __restrict__ K    [[buffer(1)]],
    device const bfloat* __restrict__ V    [[buffer(2)]],
    device       bfloat*              O    [[buffer(3)]],
    device const bfloat* __restrict__ mask [[buffer(4)]],
    constant uint&   B_val    [[buffer(5)]],
    constant uint&   H_val    [[buffer(6)]],
    constant uint&   Sq_val   [[buffer(7)]],
    constant uint&   Skv_val  [[buffer(8)]],
    constant float&  scale    [[buffer(9)]],
    constant uint&   has_mask [[buffer(10)]],
    constant uint&   group    [[buffer(11)]],
    // Strided-cache fix (M4.8): axis-1 head-stride in elements for K, V.
    constant uint&   k_h_stride [[buffer(12)]],
    constant uint&   v_h_stride [[buffer(13)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    ushort simd_gid_us [[simdgroup_index_in_threadgroup]],
    ushort simd_lid_us [[thread_index_in_simdgroup]]
) {
    uint simd_gid = (uint)simd_gid_us;
    uint simd_lid = (uint)simd_lid_us;

    uint linear = tid.x;
    uint qi     = tid.y;
    uint h      = linear % H_val;
    uint b      = linear / H_val;
    if (b >= B_val) return;

    uint H_kv = H_val / group;
    uint h_kv = h / group;

    uint q_offset = (b * H_val * Sq_val  + h    * Sq_val  + qi) * TSD_D;
    uint k_base   = (b * H_kv + h_kv) * k_h_stride;
    uint v_base   = (b * H_kv + h_kv) * v_h_stride;
    uint o_offset = q_offset;

    device const bfloat* q_ptr = Q + q_offset + simd_lid * TSD_PER;
    device const bfloat* k_ptr = K + k_base   + simd_gid * TSD_D + simd_lid * TSD_PER;
    device const bfloat* v_ptr = V + v_base   + simd_gid * TSD_D + simd_lid * TSD_PER;
    device       bfloat* o_ptr = O + o_offset + simd_gid * TSD_PER;
    device const bfloat* m_ptr = mask;
    if (has_mask != 0) {
        m_ptr = mask + qi * Skv_val + simd_gid;
    }

    uint inner_kv_stride   = TSD_BN * TSD_D;
    uint inner_mask_stride = TSD_BN;

    float q[TSD_PER];
    float k[TSD_PER];
    float o[TSD_PER];

    for (uint i = 0; i < TSD_PER; ++i) {
        q[i] = scale * float(q_ptr[i]);
    }
    for (uint i = 0; i < TSD_PER; ++i) {
        o[i] = 0.0f;
    }

    float max_score     = -INFINITY;
    float sum_exp_score = 0.0f;

    threadgroup float outputs       [TSD_BN * TSD_BD];
    threadgroup float max_scores    [TSD_BN];
    threadgroup float sum_exp_scores[TSD_BN];

    for (uint i = simd_gid; i < Skv_val; i += TSD_BN) {
        for (uint j = 0; j < TSD_PER; ++j) {
            k[j] = float(k_ptr[j]);
        }

        float score = 0.0f;
        for (uint j = 0; j < TSD_PER; ++j) {
            score += q[j] * k[j];
        }
        score = simd_sum(score);

        if (has_mask != 0) {
            score += float(m_ptr[0]);
        }

        float new_max = max(max_score, score);
        float factor  = exp(max_score - new_max);
        float exp_s   = exp(score - new_max);
        max_score     = new_max;
        sum_exp_score = sum_exp_score * factor + exp_s;

        for (uint j = 0; j < TSD_PER; ++j) {
            o[j] = o[j] * factor + exp_s * float(v_ptr[j]);
        }

        k_ptr += inner_kv_stride;
        v_ptr += inner_kv_stride;
        if (has_mask != 0) {
            m_ptr += inner_mask_stride;
        }
    }

    if (simd_lid == 0) {
        max_scores    [simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    max_score = max_scores[simd_lid];
    float new_max = simd_max(max_score);
    float factor  = exp(max_score - new_max);
    sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

    for (uint i = 0; i < TSD_PER; ++i) {
        outputs[simd_lid * TSD_BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * TSD_BD + simd_lid] * factor);
        o[i] = (sum_exp_score == 0.0f) ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        for (uint i = 0; i < TSD_PER; ++i) {
            o_ptr[i] = bfloat(o[i]);
        }
    }
}

// ── Phase 1.x cleanup follow-on: Gemma-4-specialized no-mask sdpa-vector ──
//
// Variant of `lumen_sdpa_vector_bf16` with all mask-handling code removed
// at the kernel-source level (no runtime `if (has_mask != 0)` branches, no
// mask buffer bind, no `m_ptr`/`inner_mask_stride` register pressure). Used
// for Gemma 4 sliding-attention decode (`Sq=1`, `mask=None`).
//
// Buffer signature trimmed: no mask buffer at slot [4], no has_mask scalar
// at slot [10]. Renumbered to keep the slot space dense. Other slots
// (B/H/Sq/Skv/scale/group) match the masked variant.

kernel void lumen_sdpa_vector_bf16_nomask(
    device const bfloat* __restrict__ Q    [[buffer(0)]],
    device const bfloat* __restrict__ K    [[buffer(1)]],
    device const bfloat* __restrict__ V    [[buffer(2)]],
    device       bfloat*              O    [[buffer(3)]],
    constant uint&   B_val    [[buffer(4)]],
    constant uint&   H_val    [[buffer(5)]],
    constant uint&   Sq_val   [[buffer(6)]],
    constant uint&   Skv_val  [[buffer(7)]],
    constant float&  scale    [[buffer(8)]],
    constant uint&   group    [[buffer(9)]],
    // Strided-cache fix (M4.8): axis-1 head-stride in elements for K, V.
    constant uint&   k_h_stride [[buffer(10)]],
    constant uint&   v_h_stride [[buffer(11)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    ushort simd_gid_us [[simdgroup_index_in_threadgroup]],
    ushort simd_lid_us [[thread_index_in_simdgroup]]
) {
    uint simd_gid = (uint)simd_gid_us;
    uint simd_lid = (uint)simd_lid_us;

    uint linear = tid.x;
    uint qi     = tid.y;
    uint h      = linear % H_val;
    uint b      = linear / H_val;
    if (b >= B_val) return;

    uint H_kv = H_val / group;
    uint h_kv = h / group;

    uint q_offset = (b * H_val * Sq_val  + h    * Sq_val  + qi) * TSD_D;
    uint k_base   = (b * H_kv + h_kv) * k_h_stride;
    uint v_base   = (b * H_kv + h_kv) * v_h_stride;
    uint o_offset = q_offset;

    device const bfloat* q_ptr = Q + q_offset + simd_lid * TSD_PER;
    device const bfloat* k_ptr = K + k_base   + simd_gid * TSD_D + simd_lid * TSD_PER;
    device const bfloat* v_ptr = V + v_base   + simd_gid * TSD_D + simd_lid * TSD_PER;
    device       bfloat* o_ptr = O + o_offset + simd_gid * TSD_PER;

    uint inner_kv_stride = TSD_BN * TSD_D;

    float q[TSD_PER];
    float k[TSD_PER];
    float o[TSD_PER];

    for (uint i = 0; i < TSD_PER; ++i) {
        q[i] = scale * float(q_ptr[i]);
    }
    for (uint i = 0; i < TSD_PER; ++i) {
        o[i] = 0.0f;
    }

    float max_score     = -INFINITY;
    float sum_exp_score = 0.0f;

    threadgroup float outputs       [TSD_BN * TSD_BD];
    threadgroup float max_scores    [TSD_BN];
    threadgroup float sum_exp_scores[TSD_BN];

    for (uint i = simd_gid; i < Skv_val; i += TSD_BN) {
        for (uint j = 0; j < TSD_PER; ++j) {
            k[j] = float(k_ptr[j]);
        }

        float score = 0.0f;
        for (uint j = 0; j < TSD_PER; ++j) {
            score += q[j] * k[j];
        }
        score = simd_sum(score);

        float new_max = max(max_score, score);
        float factor  = exp(max_score - new_max);
        float exp_s   = exp(score - new_max);
        max_score     = new_max;
        sum_exp_score = sum_exp_score * factor + exp_s;

        for (uint j = 0; j < TSD_PER; ++j) {
            o[j] = o[j] * factor + exp_s * float(v_ptr[j]);
        }

        k_ptr += inner_kv_stride;
        v_ptr += inner_kv_stride;
    }

    if (simd_lid == 0) {
        max_scores    [simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    max_score = max_scores[simd_lid];
    float new_max = simd_max(max_score);
    float factor  = exp(max_score - new_max);
    sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

    for (uint i = 0; i < TSD_PER; ++i) {
        outputs[simd_lid * TSD_BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * TSD_BD + simd_lid] * factor);
        o[i] = (sum_exp_score == 0.0f) ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        for (uint i = 0; i < TSD_PER; ++i) {
            o_ptr[i] = bfloat(o[i]);
        }
    }
}

// ── FlashDecode pass 1: per-split partial SDPA (no-mask, decode) ─────────
//
// Splits the KV dimension across `n_splits` threadgroups per (b, h). Each
// TG handles its own KV sub-range and writes a partial result (per-split
// max_score, sum_exp_score, o[D]) to a scratch buffer. Pass 2 merges.
//
// Scratch layout (flat float32 array, sized for n_splits up to TSD_FD_MAX_SPLITS):
//   scratch[(b * H + h) * N_SPLITS * (D + 2) + split * (D + 2) + 0]   max_score
//   scratch[(b * H + h) * N_SPLITS * (D + 2) + split * (D + 2) + 1]   sum_exp
//   scratch[(b * H + h) * N_SPLITS * (D + 2) + split * (D + 2) + 2 + d]  o[d]

constant uint TSD_FD_D       = 256;
constant uint TSD_FD_BN      = 32;   // simdgroups per TG
constant uint TSD_FD_BD      = 32;   // simd width
constant uint TSD_FD_PER     = 8;    // D / BD

kernel void lumen_sdpa_flashdecode_pass1_bf16(
    device const bfloat* __restrict__ Q       [[buffer(0)]],
    device const bfloat* __restrict__ K       [[buffer(1)]],
    device const bfloat* __restrict__ V       [[buffer(2)]],
    device       float*               scratch [[buffer(3)]],
    constant uint&   B_val     [[buffer(4)]],
    constant uint&   H_val     [[buffer(5)]],
    constant uint&   Skv_val   [[buffer(6)]],
    constant float&  scale     [[buffer(7)]],
    constant uint&   group     [[buffer(8)]],
    constant uint&   n_splits  [[buffer(9)]],
    // Strided-cache fix (M4.8): axis-1 head-stride in elements for K, V.
    constant uint&   k_h_stride [[buffer(10)]],
    constant uint&   v_h_stride [[buffer(11)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    ushort simd_gid_us [[simdgroup_index_in_threadgroup]],
    ushort simd_lid_us [[thread_index_in_simdgroup]]
) {
    uint simd_gid = (uint)simd_gid_us;
    uint simd_lid = (uint)simd_lid_us;

    uint h     = tid.x % H_val;
    uint b     = tid.x / H_val;
    uint split = tid.y;
    if (b >= B_val) return;
    if (split >= n_splits) return;

    uint H_kv = H_val / group;
    uint h_kv = h / group;

    // KV range for this split (handles non-divisible Skv).
    uint per   = (Skv_val + n_splits - 1) / n_splits;
    uint k_lo  = split * per;
    uint k_hi  = min(k_lo + per, Skv_val);
    bool empty = (k_lo >= k_hi);

    // Sq=1 decode: qi=0
    uint q_offset = (b * H_val + h) * TSD_FD_D;
    uint k_base   = (b * H_kv + h_kv) * k_h_stride;
    uint v_base   = (b * H_kv + h_kv) * v_h_stride;

    device const bfloat* q_ptr = Q + q_offset + simd_lid * TSD_FD_PER;
    device const bfloat* k_ptr = K + k_base + (k_lo + simd_gid) * TSD_FD_D
                                 + simd_lid * TSD_FD_PER;
    device const bfloat* v_ptr = V + v_base + (k_lo + simd_gid) * TSD_FD_D
                                 + simd_lid * TSD_FD_PER;

    uint inner_kv_stride = TSD_FD_BN * TSD_FD_D;

    float q[TSD_FD_PER];
    float k[TSD_FD_PER];
    float o[TSD_FD_PER];

    for (uint i = 0; i < TSD_FD_PER; ++i) {
        q[i] = scale * float(q_ptr[i]);
    }
    for (uint i = 0; i < TSD_FD_PER; ++i) {
        o[i] = 0.0f;
    }

    float max_score     = -INFINITY;
    float sum_exp_score = 0.0f;

    threadgroup float outputs       [TSD_FD_BN * TSD_FD_BD];
    threadgroup float max_scores    [TSD_FD_BN];
    threadgroup float sum_exp_scores[TSD_FD_BN];

    if (!empty) {
        for (uint i = k_lo + simd_gid; i < k_hi; i += TSD_FD_BN) {
            for (uint j = 0; j < TSD_FD_PER; ++j) {
                k[j] = float(k_ptr[j]);
            }

            float score = 0.0f;
            for (uint j = 0; j < TSD_FD_PER; ++j) {
                score += q[j] * k[j];
            }
            score = simd_sum(score);

            float new_max = max(max_score, score);
            float factor  = exp(max_score - new_max);
            float exp_s   = exp(score - new_max);
            max_score     = new_max;
            sum_exp_score = sum_exp_score * factor + exp_s;

            for (uint j = 0; j < TSD_FD_PER; ++j) {
                o[j] = o[j] * factor + exp_s * float(v_ptr[j]);
            }

            k_ptr += inner_kv_stride;
            v_ptr += inner_kv_stride;
        }
    }

    // Cross-simdgroup merge within this TG (same as nomask kernel).
    if (simd_lid == 0) {
        max_scores    [simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    max_score = max_scores[simd_lid];
    float new_max = simd_max(max_score);
    float factor  = exp(max_score - new_max);
    sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

    for (uint i = 0; i < TSD_FD_PER; ++i) {
        outputs[simd_lid * TSD_FD_BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * TSD_FD_BD + simd_lid] * factor);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write per-split partial to scratch. NB: we DO NOT divide by sum here —
    // that happens in Pass 2 after global-max correction.
    uint split_base = ((b * H_val + h) * n_splits + split) * (TSD_FD_D + 2);
    if (simd_lid == 0 && simd_gid == 0) {
        scratch[split_base + 0] = new_max;        // post-reduction global max for this split
        scratch[split_base + 1] = sum_exp_score;  // sum AT new_max scale
    }
    if (simd_lid == 0) {
        for (uint i = 0; i < TSD_FD_PER; ++i) {
            scratch[split_base + 2 + simd_gid * TSD_FD_PER + i] = o[i];
        }
    }
}

// ── FlashDecode pass 2: merge per-split partials → final O ──────────────
//
// One TG per (b, h). 256 threads, one per D element. Reads N splits'
// (max, sum, o[D]) from scratch, computes global softmax-correct merge.
//
// Algorithm:
//   global_max = max over s of split_max[s]
//   factor[s]  = exp(split_max[s] - global_max)
//   global_sum = sum over s of split_sum[s] * factor[s]
//   o_final[d] = (sum over s of o_split[s][d] * factor[s]) / global_sum

kernel void lumen_sdpa_flashdecode_pass2_bf16(
    device const float*  __restrict__ scratch [[buffer(0)]],
    device       bfloat*              O       [[buffer(1)]],
    constant uint&   B_val    [[buffer(2)]],
    constant uint&   H_val    [[buffer(3)]],
    constant uint&   n_splits [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    uint d  [[thread_position_in_threadgroup]]
) {
    uint h = tg % H_val;
    uint b = tg / H_val;
    if (b >= B_val) return;

    threadgroup float tg_global_max;
    threadgroup float tg_factors[16];  // upper bound on n_splits
    threadgroup float tg_global_sum;

    if (d == 0) {
        float m = -INFINITY;
        for (uint s = 0; s < n_splits; ++s) {
            uint sb = ((b * H_val + h) * n_splits + s) * (TSD_FD_D + 2);
            float sm = scratch[sb + 0];
            m = max(m, sm);
        }
        tg_global_max = m;

        float gs = 0.0f;
        for (uint s = 0; s < n_splits; ++s) {
            uint sb = ((b * H_val + h) * n_splits + s) * (TSD_FD_D + 2);
            float f = exp(scratch[sb + 0] - m);
            tg_factors[s] = f;
            gs += scratch[sb + 1] * f;
        }
        tg_global_sum = gs;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_sum = tg_global_sum;

    // Each thread accumulates one D-element across splits.
    float acc = 0.0f;
    for (uint s = 0; s < n_splits; ++s) {
        uint sb = ((b * H_val + h) * n_splits + s) * (TSD_FD_D + 2);
        acc += scratch[sb + 2 + d] * tg_factors[s];
    }
    float result = (global_sum > 0.0f) ? (acc / global_sum) : 0.0f;

    uint o_offset = (b * H_val + h) * TSD_FD_D + d;
    O[o_offset] = bfloat(result);
}
// ── bf16 Flash-Attention-2 PREFILL — head_dim=256, Q-tiled, in-register mask ─
//
// 2026-05-15 — proper FA-2 redesign for prefill (Sq>1). Each TG processes a
// Q-tile of TPF_QBLOCK rows, loading each K/V tile once and reusing it across
// the QBLOCK rows → asymptotically KV BW ↓ by factor QBLOCK vs the original
// 1-Q-per-TG kernel above.
//
// Layout:
//   128 threads/TG = 4 simdgroups (TPF256_NSG) × 32 lanes
//   Each SG owns 1 Q row (its tile index = sg_id). Lane owns D/32 = 8 elements.
//   Shared: K_tile[KBLOCK][D] + V_tile[KBLOCK][D] = 32 KB
//   Per-lane registers: q[8], o[8], scores[KBLOCK], m, l
//
// Mask: in-register from kv_offset + q_pos vs kv_start + j; supports causal +
// optional sliding window (window_size=0 → causal only).

constant uint TPF256_D       = 256;
constant uint TPF256_QBLOCK  = 32;      // Q rows per TG (one per SG)
constant uint TPF256_KBLOCK  = 16;      // KV rows per inner iteration
constant uint TPF256_NSG     = 32;      // # simdgroups per TG
constant uint TPF256_LANES   = 32;      // simd width
constant uint TPF256_PER     = TPF256_D / TPF256_LANES;  // 8 — D elements per lane

kernel void lumen_flash_attn_prefill_bf16_d256(
    device const bfloat* __restrict__ Q [[buffer(0)]],
    device const bfloat* __restrict__ K [[buffer(1)]],
    device const bfloat* __restrict__ V [[buffer(2)]],
    device       bfloat*              O [[buffer(3)]],
    constant uint&   B_val        [[buffer(4)]],
    constant uint&   H_val        [[buffer(5)]],
    constant uint&   Sq_val       [[buffer(6)]],
    constant uint&   Skv_val      [[buffer(7)]],
    constant float&  scale        [[buffer(8)]],
    constant uint&   group        [[buffer(9)]],
    constant uint&   window_size  [[buffer(10)]],  // 0 = no window
    constant uint&   kv_offset    [[buffer(11)]],
    constant uint&   k_h_stride   [[buffer(12)]],
    constant uint&   v_h_stride   [[buffer(13)]],
    uint3            tgid         [[threadgroup_position_in_grid]],
    uint3            tid_in_tg3   [[thread_position_in_threadgroup]],
    ushort           simd_gid_us  [[simdgroup_index_in_threadgroup]],
    ushort           simd_lid_us  [[thread_index_in_simdgroup]]
) {
    uint sg_id = (uint)simd_gid_us;     // 0..3, this SG's Q row index in tile
    uint lane  = (uint)simd_lid_us;     // 0..31
    uint tid_in_tg = tid_in_tg3.x;

    uint bh = tgid.x;                   // 0..B*H-1
    uint q_block_idx = tgid.y;          // 0..ceil(Sq/QBLOCK)-1
    uint h = bh % H_val;
    uint b = bh / H_val;
    if (b >= B_val) return;

    uint q_row = q_block_idx * TPF256_QBLOCK + sg_id;
    bool q_valid = (q_row < Sq_val);

    uint H_kv = H_val / group;
    uint h_kv = h / group;

    uint q_off = (b * H_val * Sq_val + h * Sq_val + q_row) * TPF256_D
               + lane * TPF256_PER;
    uint k_base = (b * H_kv + h_kv) * k_h_stride;
    uint v_base = (b * H_kv + h_kv) * v_h_stride;
    uint o_off = q_off;

    // Per-lane registers
    float q[TPF256_PER];
    float o[TPF256_PER];
    float scores[TPF256_KBLOCK];
    float m_state = -INFINITY;
    float l_state = 0.0f;

    if (q_valid) {
        for (uint i = 0; i < TPF256_PER; ++i) {
            q[i] = scale * float(Q[q_off + i]);
        }
    } else {
        for (uint i = 0; i < TPF256_PER; ++i) q[i] = 0.0f;
    }
    for (uint i = 0; i < TPF256_PER; ++i) o[i] = 0.0f;

    // Shared K/V tile — single 8 KB buffer reused for K (Phase A) then
    // overwritten by V (Phase D). Halves TGM vs separate K+V tiles →
    // doubles concurrent-TG-per-core occupancy on M3 Max.
    threadgroup bfloat KV_tile[TPF256_KBLOCK * TPF256_D];

    uint Q_abs = kv_offset + q_row;
    uint TG_THREADS = TPF256_NSG * TPF256_LANES;       // 1024
    uint TILE_ELEMS = TPF256_KBLOCK * TPF256_D;        // 4096
    uint LOADS_PER_THREAD = TILE_ELEMS / TG_THREADS;   // 4

    for (uint kv_start = 0; kv_start < Skv_val; kv_start += TPF256_KBLOCK) {
        // Phase A — cooperative K load (coalesced: consecutive threads →
        // consecutive memory).
        for (uint li = 0; li < LOADS_PER_THREAD; ++li) {
            uint idx    = tid_in_tg + li * TG_THREADS;
            uint kv_row = idx / TPF256_D;
            uint d_col  = idx % TPF256_D;
            uint global_kv = kv_start + kv_row;
            KV_tile[idx] = (global_kv < Skv_val)
                ? K[k_base + global_kv * TPF256_D + d_col]
                : bfloat(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase B — compute scores using K from KV_tile. Each SG handles
        // its own Q row; each lane reduces its 8-element stripe.
        for (uint j = 0; j < TPF256_KBLOCK; ++j) {
            float partial = 0.0f;
            for (uint i = 0; i < TPF256_PER; ++i) {
                partial += q[i]
                         * float(KV_tile[j * TPF256_D + lane * TPF256_PER + i]);
            }
            float s = simd_sum(partial);  // all 32 lanes of SG get same s

            uint K_abs = kv_start + j;
            bool in_bounds = (K_abs < Skv_val) && q_valid;
            bool causal    = (Q_abs >= K_abs);
            bool windowed  = (window_size == 0u)
                          || (Q_abs < K_abs + window_size);
            if (!(in_bounds && causal && windowed)) {
                s = -INFINITY;
            }
            scores[j] = s;
        }

        // Phase C — softmax state update (per-row, no shared mem).
        float m_block = -INFINITY;
        for (uint j = 0; j < TPF256_KBLOCK; ++j) {
            m_block = max(m_block, scores[j]);
        }
        float m_new = max(m_state, m_block);
        float corr  = exp(m_state - m_new);

        for (uint i = 0; i < TPF256_PER; ++i) o[i] *= corr;
        l_state *= corr;

        // Barrier to ensure all threads done reading K before V overwrites.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase D — cooperative V load over the same KV_tile buffer.
        for (uint li = 0; li < LOADS_PER_THREAD; ++li) {
            uint idx    = tid_in_tg + li * TG_THREADS;
            uint kv_row = idx / TPF256_D;
            uint d_col  = idx % TPF256_D;
            uint global_kv = kv_start + kv_row;
            KV_tile[idx] = (global_kv < Skv_val)
                ? V[v_base + global_kv * TPF256_D + d_col]
                : bfloat(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase E — accumulate o += p * V.
        for (uint j = 0; j < TPF256_KBLOCK; ++j) {
            float p = exp(scores[j] - m_new);
            l_state += p;
            for (uint i = 0; i < TPF256_PER; ++i) {
                o[i] += p
                      * float(KV_tile[j * TPF256_D + lane * TPF256_PER + i]);
            }
        }
        m_state = m_new;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (q_valid) {
        float inv_l = (l_state > 0.0f) ? (1.0f / l_state) : 0.0f;
        for (uint i = 0; i < TPF256_PER; ++i) {
            O[o_off + i] = bfloat(o[i] * inv_l);
        }
    }
}

// ── bf16 Flash-Attention-2 PREFILL — head_dim=512, causal-only ──────────
//
// Same structure as d256, but:
//   D=512 → each lane owns 16 elements (16*32 = 512)
//   KBLOCK=16 → K_tile + V_tile = 16*512*2 + 16*512*2 = 32KB ✓
//   No sliding window (Gemma 4 full-attention layers don't use one).

constant uint TPF512_D       = 512;
constant uint TPF512_QBLOCK  = 32;      // Q rows per TG (one per SG)
constant uint TPF512_KBLOCK  = 16;      // KV rows per inner iteration
constant uint TPF512_NSG     = 32;      // # simdgroups per TG
constant uint TPF512_LANES   = 32;      // simd width
constant uint TPF512_PER     = TPF512_D / TPF512_LANES;  // 16

kernel void lumen_flash_attn_prefill_bf16_d512(
    device const bfloat* __restrict__ Q [[buffer(0)]],
    device const bfloat* __restrict__ K [[buffer(1)]],
    device const bfloat* __restrict__ V [[buffer(2)]],
    device       bfloat*              O [[buffer(3)]],
    constant uint&   B_val        [[buffer(4)]],
    constant uint&   H_val        [[buffer(5)]],
    constant uint&   Sq_val       [[buffer(6)]],
    constant uint&   Skv_val      [[buffer(7)]],
    constant float&  scale        [[buffer(8)]],
    constant uint&   group        [[buffer(9)]],
    constant uint&   kv_offset    [[buffer(10)]],
    constant uint&   k_h_stride   [[buffer(11)]],
    constant uint&   v_h_stride   [[buffer(12)]],
    uint3            tgid         [[threadgroup_position_in_grid]],
    uint3            tid_in_tg3   [[thread_position_in_threadgroup]],
    ushort           simd_gid_us  [[simdgroup_index_in_threadgroup]],
    ushort           simd_lid_us  [[thread_index_in_simdgroup]]
) {
    uint sg_id = (uint)simd_gid_us;
    uint lane  = (uint)simd_lid_us;
    uint tid_in_tg = tid_in_tg3.x;

    uint bh = tgid.x;
    uint q_block_idx = tgid.y;
    uint h = bh % H_val;
    uint b = bh / H_val;
    if (b >= B_val) return;

    uint q_row = q_block_idx * TPF512_QBLOCK + sg_id;
    bool q_valid = (q_row < Sq_val);

    uint H_kv = H_val / group;
    uint h_kv = h / group;

    uint q_off = (b * H_val * Sq_val + h * Sq_val + q_row) * TPF512_D
               + lane * TPF512_PER;
    uint k_base = (b * H_kv + h_kv) * k_h_stride;
    uint v_base = (b * H_kv + h_kv) * v_h_stride;
    uint o_off = q_off;

    float q[TPF512_PER];
    float o[TPF512_PER];
    float scores[TPF512_KBLOCK];
    float m_state = -INFINITY;
    float l_state = 0.0f;

    if (q_valid) {
        for (uint i = 0; i < TPF512_PER; ++i) {
            q[i] = scale * float(Q[q_off + i]);
        }
    } else {
        for (uint i = 0; i < TPF512_PER; ++i) q[i] = 0.0f;
    }
    for (uint i = 0; i < TPF512_PER; ++i) o[i] = 0.0f;

    // Shared K/V tile (16 KB) — single buffer reused for K then V.
    threadgroup bfloat KV_tile[TPF512_KBLOCK * TPF512_D];

    uint Q_abs = kv_offset + q_row;
    uint TG_THREADS = TPF512_NSG * TPF512_LANES;       // 1024
    uint TILE_ELEMS = TPF512_KBLOCK * TPF512_D;        // 8192
    uint LOADS_PER_THREAD = TILE_ELEMS / TG_THREADS;   // 8

    for (uint kv_start = 0; kv_start < Skv_val; kv_start += TPF512_KBLOCK) {
        // Phase A — K load
        for (uint li = 0; li < LOADS_PER_THREAD; ++li) {
            uint idx    = tid_in_tg + li * TG_THREADS;
            uint kv_row = idx / TPF512_D;
            uint d_col  = idx % TPF512_D;
            uint global_kv = kv_start + kv_row;
            KV_tile[idx] = (global_kv < Skv_val)
                ? K[k_base + global_kv * TPF512_D + d_col]
                : bfloat(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase B — scores
        for (uint j = 0; j < TPF512_KBLOCK; ++j) {
            float partial = 0.0f;
            for (uint i = 0; i < TPF512_PER; ++i) {
                partial += q[i]
                         * float(KV_tile[j * TPF512_D + lane * TPF512_PER + i]);
            }
            float s = simd_sum(partial);

            uint K_abs = kv_start + j;
            bool in_bounds = (K_abs < Skv_val) && q_valid;
            bool causal    = (Q_abs >= K_abs);
            if (!(in_bounds && causal)) s = -INFINITY;
            scores[j] = s;
        }

        // Phase C — softmax state
        float m_block = -INFINITY;
        for (uint j = 0; j < TPF512_KBLOCK; ++j) {
            m_block = max(m_block, scores[j]);
        }
        float m_new = max(m_state, m_block);
        float corr  = exp(m_state - m_new);

        for (uint i = 0; i < TPF512_PER; ++i) o[i] *= corr;
        l_state *= corr;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase D — V load over same buffer
        for (uint li = 0; li < LOADS_PER_THREAD; ++li) {
            uint idx    = tid_in_tg + li * TG_THREADS;
            uint kv_row = idx / TPF512_D;
            uint d_col  = idx % TPF512_D;
            uint global_kv = kv_start + kv_row;
            KV_tile[idx] = (global_kv < Skv_val)
                ? V[v_base + global_kv * TPF512_D + d_col]
                : bfloat(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase E — accumulate
        for (uint j = 0; j < TPF512_KBLOCK; ++j) {
            float p = exp(scores[j] - m_new);
            l_state += p;
            for (uint i = 0; i < TPF512_PER; ++i) {
                o[i] += p
                      * float(KV_tile[j * TPF512_D + lane * TPF512_PER + i]);
            }
        }
        m_state = m_new;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (q_valid) {
        float inv_l = (l_state > 0.0f) ? (1.0f / l_state) : 0.0f;
        for (uint i = 0; i < TPF512_PER; ++i) {
            O[o_off + i] = bfloat(o[i] * inv_l);
        }
    }
}

// ── bf16 Flash-Attn-2 PREFILL — simdgroup_matrix (HW tensor unit) ───────
//
// 2026-05-15 iter 3 — uses Apple Silicon's simdgroup_matrix<bfloat, 8, 8>
// for Q@K^T and P@V matmuls. Targets the same compute path mlx's gemm
// fallback uses for the [B,H,L,L] scores matmul, but without materializing
// the scores tensor (FA-2 online softmax).
//
// Layout:
//   1 simdgroup per TG (32 threads), BR=8 Q rows, BC=8 KV cols per iter.
//   D=256: 32 D-fragments (Q@K^T = 32 mma; P@V = 32 mma per KV block).
//   D=512: 64 D-fragments.
// TGM (D=256):
//   O_acc[BR*D] f32  = 8 KB
//   K_tile/V_tile bf16 (shared, K→V re-use) = 4 KB
//   scores_tile f32 (8x8) = 256 B
//   P_tile bf16 (8x8) = 128 B
//   m_state[BR], l_state[BR] = 64 B
// Total: ~12 KB → 2-3 concurrent TGs/core on M3 Max.

constant uint TPFM256_D  = 256;
constant uint TPFM256_BR = 8;
constant uint TPFM256_BC = 8;
constant uint TPFM256_DF = TPFM256_D / 8;  // 32 D-fragments

// Lane → fragment-element mapping for simdgroup_matrix<T, 8, 8>:
//   thread t (0..31) owns 2 elements at (row = t / 4, col = (t % 4) * 2 + e)
//     for e in {0, 1}.
// This drives the per-row softmax via simd_shuffle_xor and the per-row
// corr application to O fragments.

kernel void lumen_flash_attn_prefill_sgm_bf16_d256(
    device const bfloat* __restrict__ Q [[buffer(0)]],
    device const bfloat* __restrict__ K [[buffer(1)]],
    device const bfloat* __restrict__ V [[buffer(2)]],
    device       bfloat*              O [[buffer(3)]],
    constant uint&   B_val        [[buffer(4)]],
    constant uint&   H_val        [[buffer(5)]],
    constant uint&   Sq_val       [[buffer(6)]],
    constant uint&   Skv_val      [[buffer(7)]],
    constant float&  scale        [[buffer(8)]],
    constant uint&   group        [[buffer(9)]],
    constant uint&   window_size  [[buffer(10)]],
    constant uint&   kv_offset    [[buffer(11)]],
    constant uint&   k_h_stride   [[buffer(12)]],
    constant uint&   v_h_stride   [[buffer(13)]],
    uint3            tgid         [[threadgroup_position_in_grid]],
    ushort           lane         [[thread_index_in_simdgroup]]
) {
    uint bh = tgid.x;
    uint q_tile_idx = tgid.y;
    uint h = bh % H_val;
    uint b = bh / H_val;
    if (b >= B_val) return;

    uint H_kv = H_val / group;
    uint h_kv = h / group;
    uint q_row_base = q_tile_idx * TPFM256_BR;

    uint q_off_base = (b * H_val * Sq_val + h * Sq_val + q_row_base) * TPFM256_D;
    uint k_base = (b * H_kv + h_kv) * k_h_stride;
    uint v_base = (b * H_kv + h_kv) * v_h_stride;
    uint o_off_base = q_off_base;

    // K, V tiles in TGM (4 KB + 4 KB = 8 KB → 4 concurrent TGs/core on M3 Max)
    threadgroup bfloat K_tile[TPFM256_BC * TPFM256_D];
    threadgroup bfloat V_tile[TPFM256_BC * TPFM256_D];

    // O accumulator in REGISTERS (32 fragments × 2 elems/lane × 4 B = 256 B/lane)
    simdgroup_matrix<float, 8, 8> O_frags[TPFM256_DF];
    for (uint df = 0; df < TPFM256_DF; ++df) {
        O_frags[df] = simdgroup_matrix<float, 8, 8>(0);
    }
    // Per-row softmax state, in registers — broadcast across 4 lanes of same row
    uint my_row = lane / 4;     // 0..7
    float m_state = -INFINITY;
    float l_state = 0.0f;

    for (uint kv_start = 0; kv_start < Skv_val; kv_start += TPFM256_BC) {
        // ── Phase A — cooperative K, V load (2048 elems each, 64/lane) ──
        for (uint li = 0; li < TPFM256_BC * TPFM256_D / 32; ++li) {
            uint idx = lane + li * 32;
            uint kv_row = idx / TPFM256_D;
            uint d_col  = idx % TPFM256_D;
            uint global_kv = kv_start + kv_row;
            if (global_kv < Skv_val) {
                K_tile[idx] = K[k_base + global_kv * TPFM256_D + d_col];
                V_tile[idx] = V[v_base + global_kv * TPFM256_D + d_col];
            } else {
                K_tile[idx] = bfloat(0);
                V_tile[idx] = bfloat(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Phase B — scores = Q @ K^T (32 D-frag MMA) ──
        simdgroup_matrix<float, 8, 8> scores_frag =
            simdgroup_matrix<float, 8, 8>(0);
        for (uint df = 0; df < TPFM256_DF; ++df) {
            simdgroup_matrix<bfloat, 8, 8> Q_frag;
            simdgroup_matrix<bfloat, 8, 8> K_frag;
            simdgroup_load(Q_frag, Q + q_off_base + df * 8, TPFM256_D);
            simdgroup_load(K_frag, K_tile + df * 8, TPFM256_D, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(scores_frag, Q_frag, K_frag, scores_frag);
        }

        // ── Phase C — scale + mask via thread_elements ──
        // Each lane owns 2 elements at (my_row, my_col0), (my_row, my_col0+1)
        uint my_col0 = (lane % 4) * 2;
        thread auto& s_elems = scores_frag.thread_elements();
        for (uint e = 0; e < 2; ++e) {
            uint j = my_col0 + e;
            float s = float(s_elems[e]) * scale;
            uint q_row = q_row_base + my_row;
            uint kv_col = kv_start + j;
            uint Q_abs = kv_offset + q_row;
            uint K_abs = kv_col;
            bool valid = (q_row < Sq_val)
                      && (kv_col < Skv_val)
                      && (Q_abs >= K_abs)
                      && (window_size == 0u || Q_abs < K_abs + window_size);
            s_elems[e] = valid ? s : -INFINITY;
        }

        // ── Phase D — per-row softmax via simd_shuffle_xor ──
        // Row reduce across 4 lanes (XOR 1 + XOR 2 = butterfly 4)
        float my_max = max(float(s_elems[0]), float(s_elems[1]));
        my_max = max(my_max, simd_shuffle_xor(my_max, (ushort)1));
        my_max = max(my_max, simd_shuffle_xor(my_max, (ushort)2));
        // All 4 row-lanes now hold the same row_max.

        float m_new = max(m_state, my_max);
        float corr  = exp(m_state - m_new);
        m_state = m_new;

        // P[my_row, col] = exp(s - m_new) — each lane computes its 2
        float p0 = exp(float(s_elems[0]) - m_new);
        float p1 = exp(float(s_elems[1]) - m_new);
        s_elems[0] = bfloat(p0);  // we'll reuse scores_frag as P after retype
        s_elems[1] = bfloat(p1);

        // Row sum of P
        float my_sum = p0 + p1;
        my_sum = my_sum + simd_shuffle_xor(my_sum, (ushort)1);
        my_sum = my_sum + simd_shuffle_xor(my_sum, (ushort)2);

        l_state = l_state * corr + my_sum;

        // ── Phase E — scale O_frags by corr + accumulate O += P @ V ──
        // Convert scores_frag (float) to a bfloat P_frag for matmul input.
        // Apple's simdgroup_matrix allows mixed-type MMA (bfloat × bfloat
        // → float accumulator), so we need a real bfloat fragment.
        simdgroup_matrix<bfloat, 8, 8> P_frag;
        thread auto& p_elems = P_frag.thread_elements();
        p_elems[0] = bfloat(p0);
        p_elems[1] = bfloat(p1);

        for (uint df = 0; df < TPFM256_DF; ++df) {
            // Scale O fragment elements (this lane owns 2) by corr.
            thread auto& o_elems = O_frags[df].thread_elements();
            o_elems[0] *= corr;
            o_elems[1] *= corr;

            simdgroup_matrix<bfloat, 8, 8> V_frag;
            simdgroup_load(V_frag, V_tile + df * 8, TPFM256_D);

            simdgroup_multiply_accumulate(
                O_frags[df], P_frag, V_frag, O_frags[df]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Final — normalize and write O ──
    float inv_l = (l_state > 0.0f) ? (1.0f / l_state) : 0.0f;
    uint q_row = q_row_base + my_row;
    bool row_valid = q_row < Sq_val;

    for (uint df = 0; df < TPFM256_DF; ++df) {
        thread auto& o_elems = O_frags[df].thread_elements();
        o_elems[0] *= inv_l;
        o_elems[1] *= inv_l;
    }

    // Store fragments back to device O via TGM staging (one frag at a time).
    threadgroup bfloat O_stage[TPFM256_BR * 8];  // 128 B
    for (uint df = 0; df < TPFM256_DF; ++df) {
        // Convert float fragment to bfloat via thread_elements + manual write
        // (no direct float→bfloat fragment in Apple MSL; store to TGM as
        // float then cast on read.)
        threadgroup float O_stage_f[TPFM256_BR * 8];
        simdgroup_store(O_frags[df], O_stage_f, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // 64 elements / 32 lanes = 2 per lane
        for (uint li = 0; li < 2; ++li) {
            uint idx = lane + li * 32;
            O_stage[idx] = bfloat(O_stage_f[idx]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // Cooperative write to device O
        for (uint li = 0; li < 2; ++li) {
            uint idx = lane + li * 32;
            uint row = idx / 8;
            uint col_in = idx % 8;
            uint q_row_w = q_row_base + row;
            if (q_row_w < Sq_val) {
                O[o_off_base + row * TPFM256_D + df * 8 + col_in] = O_stage[idx];
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ── D=512 variant (causal only, no sliding window) ──
constant uint TPFM512_D  = 512;
constant uint TPFM512_BR = 8;
constant uint TPFM512_BC = 8;
constant uint TPFM512_DF = TPFM512_D / 8;  // 64

kernel void lumen_flash_attn_prefill_sgm_bf16_d512(
    device const bfloat* __restrict__ Q [[buffer(0)]],
    device const bfloat* __restrict__ K [[buffer(1)]],
    device const bfloat* __restrict__ V [[buffer(2)]],
    device       bfloat*              O [[buffer(3)]],
    constant uint&   B_val        [[buffer(4)]],
    constant uint&   H_val        [[buffer(5)]],
    constant uint&   Sq_val       [[buffer(6)]],
    constant uint&   Skv_val      [[buffer(7)]],
    constant float&  scale        [[buffer(8)]],
    constant uint&   group        [[buffer(9)]],
    constant uint&   kv_offset    [[buffer(10)]],
    constant uint&   k_h_stride   [[buffer(11)]],
    constant uint&   v_h_stride   [[buffer(12)]],
    uint3            tgid         [[threadgroup_position_in_grid]],
    ushort           lane         [[thread_index_in_simdgroup]]
) {
    uint bh = tgid.x;
    uint q_tile_idx = tgid.y;
    uint h = bh % H_val;
    uint b = bh / H_val;
    if (b >= B_val) return;

    uint H_kv = H_val / group;
    uint h_kv = h / group;
    uint q_row_base = q_tile_idx * TPFM512_BR;

    uint q_off_base = (b * H_val * Sq_val + h * Sq_val + q_row_base) * TPFM512_D;
    uint k_base = (b * H_kv + h_kv) * k_h_stride;
    uint v_base = (b * H_kv + h_kv) * v_h_stride;
    uint o_off_base = q_off_base;

    // K, V tiles shared via single KV_tile buffer (8 KB → 4 concurrent TGs/core).
    threadgroup bfloat KV_tile[TPFM512_BC * TPFM512_D];

    // O accumulator in REGISTERS — 64 frags × 2 elems/lane × 4 B = 512 B/lane.
    simdgroup_matrix<float, 8, 8> O_frags[TPFM512_DF];
    for (uint df = 0; df < TPFM512_DF; ++df) {
        O_frags[df] = simdgroup_matrix<float, 8, 8>(0);
    }
    uint my_row = lane / 4;
    float m_state = -INFINITY;
    float l_state = 0.0f;

    for (uint kv_start = 0; kv_start < Skv_val; kv_start += TPFM512_BC) {
        // ── Phase A.1 — K load ──
        for (uint li = 0; li < TPFM512_BC * TPFM512_D / 32; ++li) {
            uint idx = lane + li * 32;
            uint kv_row = idx / TPFM512_D;
            uint d_col  = idx % TPFM512_D;
            uint global_kv = kv_start + kv_row;
            KV_tile[idx] = (global_kv < Skv_val)
                ? K[k_base + global_kv * TPFM512_D + d_col]
                : bfloat(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Phase B — Q @ K^T ──
        simdgroup_matrix<float, 8, 8> scores_frag =
            simdgroup_matrix<float, 8, 8>(0);
        for (uint df = 0; df < TPFM512_DF; ++df) {
            simdgroup_matrix<bfloat, 8, 8> Q_frag;
            simdgroup_matrix<bfloat, 8, 8> K_frag;
            simdgroup_load(Q_frag, Q + q_off_base + df * 8, TPFM512_D);
            simdgroup_load(K_frag, KV_tile + df * 8, TPFM512_D, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(scores_frag, Q_frag, K_frag, scores_frag);
        }

        // ── Phase C — scale + causal mask via thread_elements ──
        uint my_col0 = (lane % 4) * 2;
        thread auto& s_elems = scores_frag.thread_elements();
        for (uint e = 0; e < 2; ++e) {
            uint j = my_col0 + e;
            float s = float(s_elems[e]) * scale;
            uint q_row = q_row_base + my_row;
            uint kv_col = kv_start + j;
            uint Q_abs = kv_offset + q_row;
            uint K_abs = kv_col;
            bool valid = (q_row < Sq_val) && (kv_col < Skv_val) && (Q_abs >= K_abs);
            s_elems[e] = valid ? s : -INFINITY;
        }

        // ── Phase D — softmax via simd_shuffle_xor ──
        float my_max = max(float(s_elems[0]), float(s_elems[1]));
        my_max = max(my_max, simd_shuffle_xor(my_max, (ushort)1));
        my_max = max(my_max, simd_shuffle_xor(my_max, (ushort)2));
        float m_new = max(m_state, my_max);
        float corr  = exp(m_state - m_new);
        m_state = m_new;

        float p0 = exp(float(s_elems[0]) - m_new);
        float p1 = exp(float(s_elems[1]) - m_new);
        float my_sum = p0 + p1;
        my_sum = my_sum + simd_shuffle_xor(my_sum, (ushort)1);
        my_sum = my_sum + simd_shuffle_xor(my_sum, (ushort)2);
        l_state = l_state * corr + my_sum;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Phase A.2 — V load (overwrites K in KV_tile) ──
        for (uint li = 0; li < TPFM512_BC * TPFM512_D / 32; ++li) {
            uint idx = lane + li * 32;
            uint kv_row = idx / TPFM512_D;
            uint d_col  = idx % TPFM512_D;
            uint global_kv = kv_start + kv_row;
            KV_tile[idx] = (global_kv < Skv_val)
                ? V[v_base + global_kv * TPFM512_D + d_col]
                : bfloat(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Phase E — O += P @ V (with O *= corr) ──
        simdgroup_matrix<bfloat, 8, 8> P_frag;
        thread auto& p_elems = P_frag.thread_elements();
        p_elems[0] = bfloat(p0);
        p_elems[1] = bfloat(p1);

        for (uint df = 0; df < TPFM512_DF; ++df) {
            thread auto& o_elems = O_frags[df].thread_elements();
            o_elems[0] *= corr;
            o_elems[1] *= corr;

            simdgroup_matrix<bfloat, 8, 8> V_frag;
            simdgroup_load(V_frag, KV_tile + df * 8, TPFM512_D);

            simdgroup_multiply_accumulate(
                O_frags[df], P_frag, V_frag, O_frags[df]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Final write ──
    float inv_l = (l_state > 0.0f) ? (1.0f / l_state) : 0.0f;
    for (uint df = 0; df < TPFM512_DF; ++df) {
        thread auto& o_elems = O_frags[df].thread_elements();
        o_elems[0] *= inv_l;
        o_elems[1] *= inv_l;
    }

    threadgroup float O_stage_f[TPFM512_BR * 8];
    threadgroup bfloat O_stage[TPFM512_BR * 8];
    for (uint df = 0; df < TPFM512_DF; ++df) {
        simdgroup_store(O_frags[df], O_stage_f, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint li = 0; li < 2; ++li) {
            uint idx = lane + li * 32;
            O_stage[idx] = bfloat(O_stage_f[idx]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint li = 0; li < 2; ++li) {
            uint idx = lane + li * 32;
            uint row = idx / 8;
            uint col_in = idx % 8;
            uint q_row_w = q_row_base + row;
            if (q_row_w < Sq_val) {
                O[o_off_base + row * TPFM512_D + df * 8 + col_in] = O_stage[idx];
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}
)KMETAL";

} // anonymous namespace

void FlashAttnBf16::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(outputs.size() == 1);
  const auto& q = inputs[0];
  const auto& k = inputs[1];
  const auto& v = inputs[2];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);

  // Allocate output buffer.
  out.set_data(allocator::malloc(out.nbytes()));

  const int B = q.shape(0);
  const int H = q.shape(1);
  const int Sq = q.shape(2);
  const int H_kv = k.shape(1);
  const int Skv = k.shape(2);
  const uint32_t group_u = static_cast<uint32_t>(H / H_kv);

  // Strided-cache fix (M4.8): K and V might be slice views over a step-
  // allocated KV cache buffer where the axis-2 capacity exceeds Skv.
  // Use the actual axis-1 (head) stride in elements rather than the
  // logical-shape-derived `H_kv * Skv * D`. Axis-2 stride is always D
  // for our cache slices (innermost data dim stays contiguous), so the
  // per-row stride inside the kernel does not change.
  const uint32_t k_h_stride_u = static_cast<uint32_t>(k.strides()[1]);
  const uint32_t v_h_stride_u = static_cast<uint32_t>(v.strides()[1]);

  const bool use_vector = (Sq == 1);
  // Specialized no-mask kernel variant — Gemma 4 sliding-attention decode
  // always hits this path (sliding window already enforced by KV cache, no
  // additive mask needed). Dead-branch elimination saves runtime mask
  // checks inside the KV loop + a buffer bind. Only valid for the vector
  // (decode) path; FA-2 prefill still uses the unified kernel.
  //
  // A/B opt-out: `LUMEN_GEMMA4_SDPA_NOMASK_KERNEL=0` forces the masked
  // variant even when has_mask_=0, for measuring the specialization's
  // contribution against the previous baseline.
  static const bool nomask_enabled = []() {
    const char* v = std::getenv("LUMEN_GEMMA4_SDPA_NOMASK_KERNEL");
    return !(v && std::string(v) == "0");
  }();
  const bool nomask_vector = use_vector && !has_mask_ && nomask_enabled;

  // FlashDecode (split-KV) path — 2-pass dispatch for long-KV decode.
  // Each TG handles a (b, h, split) sub-range; Pass 2 merges via online
  // softmax. Originally hoped to push past the M3 Max occupancy ceiling
  // by splitting 16 TGs into 64 TGs.
  //
  // **2026-05-14 A/B FALSIFIED**: At Gemma 4 sliding-attention shape
  // (Skv=1024, B=1, H=16), median step latency was identical (17.33 ms
  // baseline vs 17.66 ms FlashDecode, within noise). The existing
  // nomask kernel at 1024 threads × 16 TGs already saturates the GPU;
  // Pass 2 dispatch overhead cancels any split-KV parallelism gain.
  // Code preserved (gated default OFF) for hypothetical longer-KV use
  // cases — e.g. if M4.8 ever extends to full-attention head_dim=512
  // where Skv grows linearly past 4096.
  //
  // Opt-in: `LUMEN_GEMMA4_SDPA_FLASHDECODE=1`.
  static const bool fd_enabled = []() {
    const char* v = std::getenv("LUMEN_GEMMA4_SDPA_FLASHDECODE");
    return v && std::string(v) == "1";
  }();
  uint32_t n_splits = 1;
  if (nomask_vector && fd_enabled && Skv >= 512) {
    if (Skv >= 4096)      n_splits = 8;
    else if (Skv >= 2048) n_splits = 4;
    else if (Skv >= 1024) n_splits = 4;
    else                  n_splits = 2;
  }
  const bool flashdecode = (n_splits > 1);

  // Load (and cache) the shader library + pipeline.
  auto lib = d.get_library(lumen_lib_name(), []() {
    return std::string(SHADER_SRC);
  });

  const uint32_t B_u = static_cast<uint32_t>(B);
  const uint32_t H_u = static_cast<uint32_t>(H);
  const uint32_t Sq_u = static_cast<uint32_t>(Sq);
  const uint32_t Skv_u = static_cast<uint32_t>(Skv);

  auto& compute_encoder = d.get_command_encoder(s.index);

  if (flashdecode) {
    // Allocate scratch as a temporary mlx array (f32, residency-tracked).
    const int32_t D = 256;
    array scratch(
        Shape{static_cast<int32_t>(B),
              static_cast<int32_t>(H),
              static_cast<int32_t>(n_splits),
              D + 2},
        float32,
        nullptr,
        {});
    scratch.set_data(allocator::malloc(scratch.nbytes()));

    // ── Pass 1 ──
    auto pipeline1 = d.get_kernel("lumen_sdpa_flashdecode_pass1_bf16", lib);
    compute_encoder.set_compute_pipeline_state(pipeline1);
    compute_encoder.set_input_array(q, 0);
    compute_encoder.set_input_array(k, 1);
    compute_encoder.set_input_array(v, 2);
    compute_encoder.set_output_array(scratch, 3);
    compute_encoder.set_bytes(B_u, 4);
    compute_encoder.set_bytes(H_u, 5);
    compute_encoder.set_bytes(Skv_u, 6);
    compute_encoder.set_bytes(scale_, 7);
    compute_encoder.set_bytes(group_u, 8);
    compute_encoder.set_bytes(n_splits, 9);
    compute_encoder.set_bytes(k_h_stride_u, 10);
    compute_encoder.set_bytes(v_h_stride_u, 11);
    {
      MTL::Size grid = MTL::Size(B * H, n_splits, 1);
      MTL::Size tg   = MTL::Size(1024, 1, 1);
      compute_encoder.dispatch_threadgroups(grid, tg);
    }

    // ── Pass 2 ──
    auto pipeline2 = d.get_kernel("lumen_sdpa_flashdecode_pass2_bf16", lib);
    compute_encoder.set_compute_pipeline_state(pipeline2);
    compute_encoder.set_input_array(scratch, 0);
    compute_encoder.set_output_array(out, 1);
    compute_encoder.set_bytes(B_u, 2);
    compute_encoder.set_bytes(H_u, 3);
    compute_encoder.set_bytes(n_splits, 4);
    {
      MTL::Size grid = MTL::Size(B * H, 1, 1);
      MTL::Size tg   = MTL::Size(256, 1, 1);
      compute_encoder.dispatch_threadgroups(grid, tg);
    }

    // Tear down scratch when the CB completes.
    d.add_temporary(scratch, s.index);
    return;
  }

  std::string kname;
  if (nomask_vector) {
    kname = "lumen_sdpa_vector_bf16_nomask";
  } else if (use_vector) {
    kname = "lumen_sdpa_vector_bf16";
  } else {
    kname = "lumen_flash_attn_bf16";
  }
  auto kernel = d.get_kernel(kname, lib);

  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_output_array(out, 3);

  if (nomask_vector) {
    // Tight signature: no mask buffer, no has_mask scalar.
    compute_encoder.set_bytes(B_u, 4);
    compute_encoder.set_bytes(H_u, 5);
    compute_encoder.set_bytes(Sq_u, 6);
    compute_encoder.set_bytes(Skv_u, 7);
    compute_encoder.set_bytes(scale_, 8);
    compute_encoder.set_bytes(group_u, 9);
    compute_encoder.set_bytes(k_h_stride_u, 10);
    compute_encoder.set_bytes(v_h_stride_u, 11);
  } else {
    // Masked / FA-2 path — full slot layout.
    if (has_mask_) {
      compute_encoder.set_input_array(inputs[3], 4);
    } else {
      // Reuse Q as a placeholder when the FA-2 kernel gets the no-mask
      // value at runtime (kept for non-vector code paths).
      compute_encoder.set_input_array(q, 4);
    }
    const uint32_t has_mask_u = has_mask_ ? 1u : 0u;
    compute_encoder.set_bytes(B_u, 5);
    compute_encoder.set_bytes(H_u, 6);
    compute_encoder.set_bytes(Sq_u, 7);
    compute_encoder.set_bytes(Skv_u, 8);
    compute_encoder.set_bytes(scale_, 9);
    compute_encoder.set_bytes(has_mask_u, 10);
    compute_encoder.set_bytes(group_u, 11);
    compute_encoder.set_bytes(k_h_stride_u, 12);
    compute_encoder.set_bytes(v_h_stride_u, 13);
  }

  if (use_vector) {
    // sdpa_vector: 32 simdgroups × 32 lanes, grid = (B*H, Sq, 1)
    MTL::Size grid_dims = MTL::Size(B * H, Sq, 1);
    MTL::Size group_dims = MTL::Size(1024, 1, 1);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  } else {
    // FA-2: 256 threads/TG, grid = (B*H*Sq, 1, 1)
    MTL::Size grid_dims = MTL::Size(B * H * Sq, 1, 1);
    MTL::Size group_dims = MTL::Size(256, 1, 1);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }
}

void FlashAttnPrefillBf16::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(outputs.size() == 1);
  const auto& q = inputs[0];
  const auto& k = inputs[1];
  const auto& v = inputs[2];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);

  out.set_data(allocator::malloc(out.nbytes()));

  const int B = q.shape(0);
  const int H = q.shape(1);
  const int Sq = q.shape(2);
  const int D = q.shape(3);
  const int H_kv = k.shape(1);
  const int Skv = k.shape(2);
  const uint32_t group_u = static_cast<uint32_t>(H / H_kv);

  const uint32_t k_h_stride_u = static_cast<uint32_t>(k.strides()[1]);
  const uint32_t v_h_stride_u = static_cast<uint32_t>(v.strides()[1]);

  auto lib = d.get_library(lumen_lib_name(), []() {
    return std::string(SHADER_SRC);
  });

  const uint32_t B_u = static_cast<uint32_t>(B);
  const uint32_t H_u = static_cast<uint32_t>(H);
  const uint32_t Sq_u = static_cast<uint32_t>(Sq);
  const uint32_t Skv_u = static_cast<uint32_t>(Skv);

  auto& compute_encoder = d.get_command_encoder(s.index);

  // iter-2 (occupancy-tuned scalar) is the default when the prefill kernel
  // is opted in via LUMEN_GEMMA4_PREFILL_KERNEL=1. SGM (simdgroup_matrix)
  // kernels are kept behind LUMEN_GEMMA4_PREFILL_KERNEL_SGM=1 for further
  // iteration — at iter 3+4 SGM was slower than iter-2 due to TG launch
  // overhead (16k TGs with 1 SG/TG) and fragment register-file pressure.
  static const bool use_sgm = []() {
    const char* v = std::getenv("LUMEN_GEMMA4_PREFILL_KERNEL_SGM");
    return v && std::string(v) == "1";
  }();

  if (D == 256 && use_sgm) {
    // SGM kernel: 1 SG/TG (32 threads), BR=8
    auto pipeline =
        d.get_kernel("lumen_flash_attn_prefill_sgm_bf16_d256", lib);
    compute_encoder.set_compute_pipeline_state(pipeline);
    compute_encoder.set_input_array(q, 0);
    compute_encoder.set_input_array(k, 1);
    compute_encoder.set_input_array(v, 2);
    compute_encoder.set_output_array(out, 3);
    compute_encoder.set_bytes(B_u, 4);
    compute_encoder.set_bytes(H_u, 5);
    compute_encoder.set_bytes(Sq_u, 6);
    compute_encoder.set_bytes(Skv_u, 7);
    compute_encoder.set_bytes(scale_, 8);
    compute_encoder.set_bytes(group_u, 9);
    compute_encoder.set_bytes(window_size_, 10);
    compute_encoder.set_bytes(kv_offset_, 11);
    compute_encoder.set_bytes(k_h_stride_u, 12);
    compute_encoder.set_bytes(v_h_stride_u, 13);

    uint32_t q_tiles = (Sq + 8 - 1) / 8;  // BR=8
    MTL::Size grid = MTL::Size(B * H, q_tiles, 1);
    MTL::Size tg = MTL::Size(32, 1, 1);
    compute_encoder.dispatch_threadgroups(grid, tg);
    return;
  }
  if (D == 512 && use_sgm) {
    auto pipeline =
        d.get_kernel("lumen_flash_attn_prefill_sgm_bf16_d512", lib);
    compute_encoder.set_compute_pipeline_state(pipeline);
    compute_encoder.set_input_array(q, 0);
    compute_encoder.set_input_array(k, 1);
    compute_encoder.set_input_array(v, 2);
    compute_encoder.set_output_array(out, 3);
    compute_encoder.set_bytes(B_u, 4);
    compute_encoder.set_bytes(H_u, 5);
    compute_encoder.set_bytes(Sq_u, 6);
    compute_encoder.set_bytes(Skv_u, 7);
    compute_encoder.set_bytes(scale_, 8);
    compute_encoder.set_bytes(group_u, 9);
    compute_encoder.set_bytes(kv_offset_, 10);
    compute_encoder.set_bytes(k_h_stride_u, 11);
    compute_encoder.set_bytes(v_h_stride_u, 12);

    uint32_t q_tiles = (Sq + 8 - 1) / 8;
    MTL::Size grid = MTL::Size(B * H, q_tiles, 1);
    MTL::Size tg = MTL::Size(32, 1, 1);
    compute_encoder.dispatch_threadgroups(grid, tg);
    return;
  }

  // Legacy scalar kernels (iter 2) — A/B fallback.
  if (D == 256) {
    constexpr uint32_t TPF256_QBLOCK = 32;
    auto pipeline = d.get_kernel("lumen_flash_attn_prefill_bf16_d256", lib);
    compute_encoder.set_compute_pipeline_state(pipeline);
    compute_encoder.set_input_array(q, 0);
    compute_encoder.set_input_array(k, 1);
    compute_encoder.set_input_array(v, 2);
    compute_encoder.set_output_array(out, 3);
    compute_encoder.set_bytes(B_u, 4);
    compute_encoder.set_bytes(H_u, 5);
    compute_encoder.set_bytes(Sq_u, 6);
    compute_encoder.set_bytes(Skv_u, 7);
    compute_encoder.set_bytes(scale_, 8);
    compute_encoder.set_bytes(group_u, 9);
    compute_encoder.set_bytes(window_size_, 10);
    compute_encoder.set_bytes(kv_offset_, 11);
    compute_encoder.set_bytes(k_h_stride_u, 12);
    compute_encoder.set_bytes(v_h_stride_u, 13);

    uint32_t q_tiles = (Sq + TPF256_QBLOCK - 1) / TPF256_QBLOCK;
    MTL::Size grid = MTL::Size(B * H, q_tiles, 1);
    MTL::Size tg = MTL::Size(1024, 1, 1);
    compute_encoder.dispatch_threadgroups(grid, tg);
  } else if (D == 512) {
    constexpr uint32_t TPF512_QBLOCK = 32;
    auto pipeline = d.get_kernel("lumen_flash_attn_prefill_bf16_d512", lib);
    compute_encoder.set_compute_pipeline_state(pipeline);
    compute_encoder.set_input_array(q, 0);
    compute_encoder.set_input_array(k, 1);
    compute_encoder.set_input_array(v, 2);
    compute_encoder.set_output_array(out, 3);
    compute_encoder.set_bytes(B_u, 4);
    compute_encoder.set_bytes(H_u, 5);
    compute_encoder.set_bytes(Sq_u, 6);
    compute_encoder.set_bytes(Skv_u, 7);
    compute_encoder.set_bytes(scale_, 8);
    compute_encoder.set_bytes(group_u, 9);
    compute_encoder.set_bytes(kv_offset_, 10);
    compute_encoder.set_bytes(k_h_stride_u, 11);
    compute_encoder.set_bytes(v_h_stride_u, 12);

    uint32_t q_tiles = (Sq + TPF512_QBLOCK - 1) / TPF512_QBLOCK;
    MTL::Size grid = MTL::Size(B * H, q_tiles, 1);
    MTL::Size tg = MTL::Size(1024, 1, 1);
    compute_encoder.dispatch_threadgroups(grid, tg);
  } else {
    throw std::runtime_error(
        "lumen::FlashAttnPrefillBf16: unsupported head_dim");
  }
}

} // namespace mlx::core::lumen
