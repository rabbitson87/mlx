// lumen-rs TurboQuant — Metal backend impl for lumen::TurboquantEncode.
//
// Replaces the mlx-ops argmin-broadcast path (which materialized an
// O(N × n_levels) intermediate per call) with a single per-element kernel
// that linear-scans the boundaries array in threadgroup memory.
//
// Compute per element: (n_levels - 1) fp32 compares + (n_levels - 1) adds.
// Memory: input x_norm (bf16) + output codes (uint8) + boundaries (f32) in
// threadgroup. For 4-bit n_levels=16, that's 15 compares vs argmin's
// 16-way reduce — same arithmetic count but no intermediate materialization
// and much better cache behavior.

#include <stdexcept>
#include <string>

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/lumen_primitives.h"

namespace mlx::core::lumen {

namespace {

const std::string& tq_lib_name() {
  static const std::string name = "lumen_turboquant";
  return name;
}

constexpr const char* TQ_SHADER_SRC = R"KMETAL(
#include <metal_stdlib>
using namespace metal;

// ── TurboQuant Lloyd-Max nearest-centroid encode (Stage 1) ──────────────
//
// Reads pre-normalized x_norm (≈ N(0,1) after rotation + per-vector σ
// division), looks up nearest centroid via linear scan over the boundary
// table (loaded once into threadgroup memory), writes uint8 code.
//
// Buffers:
//   [0] x_norm     [N_elements] bfloat16  (logically [N_rows, D])
//   [1] boundaries [n_inner = n_levels - 1] float32
//   [2] codes      [N_elements] uint8
//   [3] n_inner    uint32  (= n_levels - 1)
//   [4] n_elements uint32  (= N_rows * D)
//
// Each thread handles one element. Grid = ceil(N / TG_SIZE) threadgroups,
// TG_SIZE threads per group. n_inner is capped at 255 (8-bit) so the
// threadgroup buffer fits in ~1 KB.

constant uint TQ_MAX_INNER = 255;  // 8-bit max

kernel void lumen_tq_encode(
    device const bfloat* __restrict__ x_norm     [[buffer(0)]],
    device const float*  __restrict__ boundaries [[buffer(1)]],
    device       uchar*               codes      [[buffer(2)]],
    constant uint& n_inner    [[buffer(3)]],
    constant uint& n_elements [[buffer(4)]],
    uint gid    [[thread_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint tg_sz  [[threads_per_threadgroup]]
) {
    // Cooperative load of boundaries into TGM.
    threadgroup float bnd_tg[TQ_MAX_INNER];
    for (uint j = lid; j < n_inner; j += tg_sz) {
        bnd_tg[j] = boundaries[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid >= n_elements) return;

    float x = float(x_norm[gid]);
    uint code = 0;
    // Linear scan — branchless via accumulator. For Gaussian-distributed
    // x, most elements cluster near 0 → branch prediction would be fine
    // either way; branchless gives consistent ILP.
    for (uint j = 0; j < n_inner; j++) {
        code += (uint)(x >= bnd_tg[j]);
    }
    codes[gid] = (uchar)code;
}

// ── TurboQuant Stage-1 fused: σ + normalize + Lloyd-Max encode ─────────
//
// One threadgroup per row (= one [b, h, s] slice of D contiguous elements);
// one thread per element. Computes per-row σ = sqrt(mean(x²)) via
// simdgroup reduction, normalizes x in-register, and emits both Lloyd-Max
// codes and σ in a single dispatch.
//
// Replaces the multi-op chain (cast → square → sum_axis → divide → sqrt →
// divide → cast → encode) with one kernel — ~8 dispatches → 1.
//
// Buffers:
//   [0] x_rot      [N_rows × D] bfloat16  (rotated, NOT normalized)
//   [1] boundaries [n_inner]    float32
//   [2] codes      [N_rows × D] uint8
//   [3] sigma_out  [N_rows]     float32
//   [4] n_inner    uint32
//   [5] D          uint32       (last-axis size; multiple of 32, ≤ 1024)
//
// Grid: (N_rows, 1, 1) threadgroups × (D, 1, 1) threads/group.

constant uint TQ_MAX_SG = 32; // D ≤ 1024 → simdgroups per TG ≤ 32

kernel void lumen_tq_encode_fused(
    device const bfloat* __restrict__ x_rot      [[buffer(0)]],
    device const float*  __restrict__ boundaries [[buffer(1)]],
    device       uchar*               codes      [[buffer(2)]],
    device       float*               sigma_out  [[buffer(3)]],
    constant uint& n_inner [[buffer(4)]],
    constant uint& D       [[buffer(5)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint sg_tid [[thread_index_in_simdgroup]]
) {
    const uint row = tgid;
    const uint elem_idx = row * D + lid;

    threadgroup float sg_sums[TQ_MAX_SG];
    threadgroup float sigma_tg;
    threadgroup float bnd_tg[TQ_MAX_INNER];

    // Load element + intra-simdgroup sum of squares.
    float x   = float(x_rot[elem_idx]);
    float xsq = x * x;
    float sg_sum = simd_sum(xsq);
    if (sg_tid == 0) {
        sg_sums[sg_idx] = sg_sum;
    }

    // Boundaries cooperative load runs in parallel with the reduction stage;
    // both writes are made visible by the barrier below.
    for (uint j = lid; j < n_inner; j += D) {
        bnd_tg[j] = boundaries[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cross-simdgroup reduce in the first simdgroup → σ.
    if (sg_idx == 0) {
        uint n_sg = D / 32u;
        float v = (sg_tid < n_sg) ? sg_sums[sg_tid] : 0.0f;
        float total = simd_sum(v);
        if (sg_tid == 0) {
            float sigma = sqrt(total / float(D));
            sigma_tg = sigma;
            sigma_out[row] = sigma;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Normalize + Lloyd-Max nearest-centroid encode.
    float sigma = sigma_tg;
    float x_norm = x / sigma;
    uint code = 0;
    for (uint j = 0; j < n_inner; j++) {
        code += (uint)(x_norm >= bnd_tg[j]);
    }
    codes[elem_idx] = (uchar)code;
}

// ── TurboQuant Stage-1 rotate + encode fused ──────────────────────────────
//
// Equivalent to `(x @ R) → encode_fused`, but skips the bf16 round-trip on
// the rotated tensor (and the separate matmul + cast dispatches). Used by
// the V side of the SlidingTurboquant branch: V_rot is consumed only by
// encode, so materializing it as bf16 is pure waste.
//
// Per-thread compute: one D-long dot product against a column of R, then
// participates in the per-row σ reduction (same pattern as
// `lumen_tq_encode_fused`).
//
// Memory access: R is read in DRAM-coalesced rows — at iteration k, all D
// threads in the TG load R[k, 0..D-1] (contiguous 4·D bytes). For D=256
// that's a 1KB stripe per iteration, fully coalesced.
//
// Buffers:
//   [0] x_bf16     [N_rows × D] bfloat16  (rotated *input*)
//   [1] R_f32      [D × D]      float32   (row-major, shared across rows)
//   [2] boundaries [n_inner]    float32
//   [3] codes      [N_rows × D] uint8
//   [4] sigma_out  [N_rows]     float32
//   [5] n_inner    uint32
//   [6] D          uint32
//
// Grid: (N_rows, 1, 1) threadgroups × (D, 1, 1) threads/group.

constant uint TQ_MAX_D = 1024;

kernel void lumen_tq_rot_encode_fused(
    device const bfloat* __restrict__ x_bf16     [[buffer(0)]],
    device const float*  __restrict__ R_f32      [[buffer(1)]],
    device const float*  __restrict__ boundaries [[buffer(2)]],
    device       uchar*               codes      [[buffer(3)]],
    device       float*               sigma_out  [[buffer(4)]],
    constant uint& n_inner [[buffer(5)]],
    constant uint& D       [[buffer(6)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint sg_tid [[thread_index_in_simdgroup]]
) {
    const uint row = tgid;
    const uint x_base = row * D;

    threadgroup float x_tg[TQ_MAX_D];
    threadgroup float sg_sums[TQ_MAX_SG];
    threadgroup float sigma_tg;
    threadgroup float bnd_tg[TQ_MAX_INNER];

    // (1) Cooperative load: x → TGM as f32 (one element per thread).
    x_tg[lid] = float(x_bf16[x_base + lid]);

    // (2) Cooperative load of boundaries (small).
    for (uint j = lid; j < n_inner; j += D) {
        bnd_tg[j] = boundaries[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // (3) Each thread computes y[lid] = sum_k x[k] * R[k, lid].
    //     Loop over k advances by 1 row of R; reads are coalesced across
    //     threads (lid varies fastest in the column index).
    float y = 0.0f;
    for (uint k = 0; k < D; k++) {
        y = fma(x_tg[k], R_f32[k * D + lid], y);
    }

    // (4) σ reduce on y² via simdgroup + cross-simdgroup.
    float ysq = y * y;
    float sg_sum = simd_sum(ysq);
    if (sg_tid == 0) {
        sg_sums[sg_idx] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_idx == 0) {
        uint n_sg = D / 32u;
        float v = (sg_tid < n_sg) ? sg_sums[sg_tid] : 0.0f;
        float total = simd_sum(v);
        if (sg_tid == 0) {
            float sigma = sqrt(total / float(D));
            sigma_tg = sigma;
            sigma_out[row] = sigma;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // (5) Normalize + Lloyd-Max nearest-centroid encode (one thread per
    // output element).
    float sigma = sigma_tg;
    float y_norm = y / sigma;
    uint code = 0;
    for (uint j = 0; j < n_inner; j++) {
        code += (uint)(y_norm >= bnd_tg[j]);
    }
    codes[x_base + lid] = (uchar)code;
}

// ── TurboQuant Q @ K_codes inline matmul (decode-shape) ────────────────────
//
// Computes attention scores S[B, H, T, N] = (Q · K_dq^T) without materializing
// K_dq. Inline Lloyd-Max dequant: each K element is one uint8 code indexing a
// shared centroids LUT, scaled by per-K-vector σ. Eliminates the 32 MB K_dq
// DRAM round-trip per decode step at 8K context (4-stage roadmap Stage 2).
//
// Shape:
//   Q       : bfloat16 [B, H,    T, D]   (T = 1 for decode; T > 1 ok)
//   K_codes : uint8    [B, H_kv, N, D]   (one byte per code; bits ≤ 4)
//   K_sigma : float32  [B, H_kv, N]      (per-K-vector σ; reshape from
//                                         [..., 1] in the dispatch)
//   centrds : float32  [n_levels]        (Lloyd-Max LUT; n_levels ≤ 16)
//   scores  : bfloat16 [B, H,    T, N]
//
// GQA: h_kv = h * H_kv / H. H % H_kv == 0 required.
//
// Tile (modeled after mlx qmv_fast_impl, which is the canonical
// small-M × large-N × dequant-B kernel):
//   num_simdgroups        = 2
//   results_per_simdgroup = 4   (N-rows per simdgroup)
//   values_per_thread     = 8   (D-elements per thread; 32·8 = 256)
//   block_size            = 256 (single inner-pass when D ≤ 256)
// Each TG processes (num_simdgroups · RPS = 8) N rows for one (b, h, t) tile.
//
// Grid:    (ceil(N / 8), T, B · H)
// TG-size: (num_simdgroups · SIMD_SIZE = 64, 1, 1)
//
// First-iteration constraints (enforced at factory):
//   - D == 256  (Gemma 4 head_dim; later iterations can template on D)
//   - n_levels ≤ 16 (bits ≤ 4)

constant uint TQ_QK_NSG        = 2;    // simdgroups per TG
constant uint TQ_QK_RPS        = 4;    // N-rows per simdgroup
constant uint TQ_QK_VPT        = 8;    // D-elements per thread
constant uint TQ_QK_ROWS_PER_TG = TQ_QK_NSG * TQ_QK_RPS;  // = 8
constant uint TQ_QK_MAX_NCENTROIDS = 16;  // bits ≤ 4

kernel void lumen_tq_qk_inline(
    device const bfloat* __restrict__ Q         [[buffer(0)]],
    device const uchar*  __restrict__ K_codes   [[buffer(1)]],
    device const float*  __restrict__ K_sigma   [[buffer(2)]],
    device const float*  __restrict__ centroids [[buffer(3)]],
    device       bfloat*              scores    [[buffer(4)]],
    constant uint& B    [[buffer(5)]],
    constant uint& H    [[buffer(6)]],
    constant uint& H_kv [[buffer(7)]],
    constant uint& T    [[buffer(8)]],
    constant uint& N    [[buffer(9)]],
    constant uint& D    [[buffer(10)]],
    constant uint& n_levels [[buffer(11)]],
    uint3 tid     [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]]
) {
    threadgroup float centroids_tg[TQ_QK_MAX_NCENTROIDS];

    // Decode grid index.
    const uint nb = tid.x;
    const uint t  = tid.y;
    const uint bh = tid.z;
    const uint b  = bh / H;
    const uint h  = bh % H;
    const uint h_kv = h * H_kv / H;  // GQA: H % H_kv == 0

    const uint n_base = nb * TQ_QK_ROWS_PER_TG + simd_gid * TQ_QK_RPS;

    // Cooperative load of centroids LUT (≤ 16 floats = 64 B) into TGM.
    if (simd_gid == 0 && simd_lid < n_levels) {
        centroids_tg[simd_lid] = centroids[simd_lid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load this thread's slice of the Q row into registers.
    float x_thread[TQ_QK_VPT];
    const uint q_base = ((b * H + h) * T + t) * D + simd_lid * TQ_QK_VPT;
    for (uint v = 0; v < TQ_QK_VPT; v++) {
        x_thread[v] = float(Q[q_base + v]);
    }

    // Per-row accumulator (RPS rows per simdgroup).
    float result[TQ_QK_RPS] = {0.0f, 0.0f, 0.0f, 0.0f};

    const uint K_block_base = (b * H_kv + h_kv) * N * D;

    for (uint row = 0; row < TQ_QK_RPS; row++) {
        uint n = n_base + row;
        if (n >= N) {
            continue;
        }
        const device uchar* K_row =
            K_codes + K_block_base + n * D + simd_lid * TQ_QK_VPT;
        float partial = 0.0f;
        for (uint v = 0; v < TQ_QK_VPT; v++) {
            uchar code = K_row[v];
            partial = fma(x_thread[v], centroids_tg[code], partial);
        }
        result[row] = partial;
    }

    // simd_sum across the 32 threads in this simdgroup → one result per
    // (simdgroup, row). Apply per-K-vector σ and store as bf16.
    const uint sigma_base  = (b * H_kv + h_kv) * N;
    const uint scores_base = ((b * H + h) * T + t) * N;

    for (uint row = 0; row < TQ_QK_RPS; row++) {
        uint n = n_base + row;
        float sum = simd_sum(result[row]);
        if (simd_lid == 0 && n < N) {
            float sigma_val = K_sigma[sigma_base + n];
            scores[scores_base + n] = bfloat(sum * sigma_val);
        }
    }
}

// ── TurboQuant Stage-1 fused encode (PACKED 4-bit) ─────────────────────────
//
// Same as `lumen_tq_encode_fused` but emits codes as packed uint32 instead
// of uint8 — 8 codes per uint32 (4-bit each, low-nibble-first ordering).
// Halves K_codes / V_codes DRAM footprint and read bandwidth in the inline
// matmul kernels.
//
// Packing convention: code at logical index i goes into bits [(i%8)*4,
// (i%8)*4 + 3] of uint32 at index i/8 within the row. Matches mlx's
// quantized.h pack ordering for 4-bit weights.
//
// Buffers:
//   [0] x_rot      [N_rows × D] bfloat16
//   [1] boundaries [n_inner]    float32 (must imply n_levels=16 for 4-bit)
//   [2] codes_pkd  [N_rows × D/8] uint32 (PACKED)
//   [3] sigma_out  [N_rows]     float32
//   [4] n_inner    uint32
//   [5] D          uint32       (must be multiple of 8)

kernel void lumen_tq_encode_fused_packed4(
    device const bfloat* __restrict__ x_rot      [[buffer(0)]],
    device const float*  __restrict__ boundaries [[buffer(1)]],
    device       uint*                codes_pkd  [[buffer(2)]],
    device       float*               sigma_out  [[buffer(3)]],
    constant uint& n_inner [[buffer(4)]],
    constant uint& D       [[buffer(5)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint sg_tid [[thread_index_in_simdgroup]]
) {
    const uint row = tgid;
    const uint elem_idx = row * D + lid;

    threadgroup float sg_sums[TQ_MAX_SG];
    threadgroup float sigma_tg;
    threadgroup float bnd_tg[TQ_MAX_INNER];
    threadgroup uint  pack_tg[1024 / 8];  // up to D=1024 → D/8 packed words

    // Load x + per-thread square (same as encode_fused).
    float x   = float(x_rot[elem_idx]);
    float xsq = x * x;
    float sg_sum = simd_sum(xsq);
    if (sg_tid == 0) {
        sg_sums[sg_idx] = sg_sum;
    }

    for (uint j = lid; j < n_inner; j += D) {
        bnd_tg[j] = boundaries[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_idx == 0) {
        uint n_sg = D / 32u;
        float v = (sg_tid < n_sg) ? sg_sums[sg_tid] : 0.0f;
        float total = simd_sum(v);
        if (sg_tid == 0) {
            float sigma = sqrt(total / float(D));
            sigma_tg = sigma;
            sigma_out[row] = sigma;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Normalize + nearest-centroid encode (per-thread 4-bit code).
    float sigma = sigma_tg;
    float x_norm = x / sigma;
    uint code = 0;
    for (uint j = 0; j < n_inner; j++) {
        code += (uint)(x_norm >= bnd_tg[j]);
    }
    // 4-bit code ∈ [0, 15].

    // Pack: 8 codes (32 bits) per uint32. Use atomic-OR on TGM to combine.
    // Each thread contributes one nibble at bit-offset (lid%8)*4 within
    // the word at index lid/8. Lane-collisions within the same word are
    // resolved by initializing pack_tg to 0 and OR-accumulating.
    uint word_idx = lid >> 3;            // lid / 8
    uint bit_off  = (lid & 7u) << 2;     // (lid % 8) * 4

    // Zero pack_tg cooperatively. One word per 8 threads → D/8 words.
    if ((lid & 7u) == 0u) {
        pack_tg[word_idx] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    atomic_fetch_or_explicit(
        (threadgroup atomic_uint*)&pack_tg[word_idx],
        (code & 0xFu) << bit_off,
        memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // One thread per packed word writes to DRAM.
    if ((lid & 7u) == 0u) {
        uint n_packed = D >> 3;
        codes_pkd[row * n_packed + word_idx] = pack_tg[word_idx];
    }
}

// ── TurboQuant Q @ K_codes_packed inline matmul (4-bit packed) ────────────
//
// Same shape & tile as `lumen_tq_qk_inline` but reads PACKED uint32 K_codes
// (8 codes per word, 4-bit each). Halves DRAM read bandwidth for K vs the
// uint8 variant.
//
// Per thread inner loop: VPT=8 means 8 codes per K row. With 4-bit packing,
// 8 codes fit in ONE uint32 word → ONE 4-byte coalesced load per thread per
// row (vs 8 separate 1-byte loads in the unpacked kernel).
//
// Code extraction: code_k = (word >> (k * 4)) & 0xF for k = 0..7.

kernel void lumen_tq_qk_inline_packed4(
    device const bfloat* __restrict__ Q         [[buffer(0)]],
    device const uint*   __restrict__ K_codes_pkd [[buffer(1)]],
    device const float*  __restrict__ K_sigma   [[buffer(2)]],
    device const float*  __restrict__ centroids [[buffer(3)]],
    device       bfloat*              scores    [[buffer(4)]],
    constant uint& B    [[buffer(5)]],
    constant uint& H    [[buffer(6)]],
    constant uint& H_kv [[buffer(7)]],
    constant uint& T    [[buffer(8)]],
    constant uint& N    [[buffer(9)]],
    constant uint& D    [[buffer(10)]],
    constant uint& n_levels [[buffer(11)]],
    uint3 tid     [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]]
) {
    threadgroup float centroids_tg[TQ_QK_MAX_NCENTROIDS];

    const uint nb = tid.x;
    const uint t  = tid.y;
    const uint bh = tid.z;
    const uint b  = bh / H;
    const uint h  = bh % H;
    const uint h_kv = h * H_kv / H;

    const uint n_base = nb * TQ_QK_ROWS_PER_TG + simd_gid * TQ_QK_RPS;

    if (simd_gid == 0 && simd_lid < n_levels) {
        centroids_tg[simd_lid] = centroids[simd_lid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load Q slice into registers (same as unpacked variant).
    float x_thread[TQ_QK_VPT];
    const uint q_base = ((b * H + h) * T + t) * D + simd_lid * TQ_QK_VPT;
    for (uint v = 0; v < TQ_QK_VPT; v++) {
        x_thread[v] = float(Q[q_base + v]);
    }

    float result[TQ_QK_RPS] = {0.0f, 0.0f, 0.0f, 0.0f};

    // PACKED: codes stored as uint32 [..., D/8]. Each thread reads ONE
    // uint32 per row (its simd_lid'th word covers VPT=8 codes).
    const uint n_packed = D >> 3;   // D/8 words per row
    const uint K_block_base_pkd = (b * H_kv + h_kv) * N * n_packed;

    for (uint row = 0; row < TQ_QK_RPS; row++) {
        uint n = n_base + row;
        if (n >= N) continue;
        // Word at index simd_lid covers codes [simd_lid*8, simd_lid*8+8).
        uint word =
            K_codes_pkd[K_block_base_pkd + n * n_packed + simd_lid];
        float partial = 0.0f;
        #pragma clang loop unroll(full)
        for (uint v = 0; v < TQ_QK_VPT; v++) {
            uchar code = (uchar)((word >> (v * 4u)) & 0xFu);
            partial = fma(x_thread[v], centroids_tg[code], partial);
        }
        result[row] = partial;
    }

    const uint sigma_base  = (b * H_kv + h_kv) * N;
    const uint scores_base = ((b * H + h) * T + t) * N;

    for (uint row = 0; row < TQ_QK_RPS; row++) {
        uint n = n_base + row;
        float sum = simd_sum(result[row]);
        if (simd_lid == 0 && n < N) {
            float sigma_val = K_sigma[sigma_base + n];
            scores[scores_base + n] = bfloat(sum * sigma_val);
        }
    }
}

// ── TurboQuant softmax_scores @ V_codes inline matmul (decode-shape) ──────
//
// Computes attention output O[B, H, T, D] = (S · V_dq) without materializing
// V_dq. Inline Lloyd-Max dequant: each V element is one uint8 code that
// indexes a shared centroids LUT, scaled by per-V-vector σ. Eliminates the
// 32 MB V_dq DRAM round-trip per decode step at 8K context (4-stage roadmap
// Stage 3 — symmetric to Stage 2's K-side kernel).
//
// Shape:
//   S         : bfloat16 [B, H,    T, N]   (softmax-normalized scores)
//   V_codes   : uint8    [B, H_kv, N, D]
//   V_sigma   : float32  [B, H_kv, N]
//   centroids : float32  [n_levels]        (Lloyd-Max LUT; n_levels ≤ 16)
//   O         : bfloat16 [B, H,    T, D]
//
// **Tile v2 — qvm-style (parallel N reduction)**. The v1 "one thread per
// D output, serial N loop" layout regressed −20% decode in the A/B
// (gemma4_tq_sv_inline_negative.md): only B·H=16 TGs of parallelism and
// each thread's N reduction was fully serial. v2 switches to mlx's
// `qvm_impl` parallelism profile:
//
//   - 32 threads per simdgroup process 32 N values in parallel.
//   - Each thread accumulates `result[D_TILE_PER_SG=32]` outputs in
//     registers (128 B per thread, well within Apple Silicon's reg budget).
//   - After the N loop, `simd_sum(result[k])` reduces across the 32
//     threads — collapsing the N dimension that was striped across them.
//   - 2 simdgroups per TG → 64 D outputs per TG.
//
// Grid: (D / D_TILE_PER_TG = D/64, T, B·H) = 4 × T × B·H TGs (4× v1's TG
// count for Gemma 4: 64 instead of 16). Inner N loop is now O(N/32)
// per thread instead of O(N).
//
// Memory access (per N-chunk iteration):
//   - S[t, n_chunk..n_chunk+32]      : 32 bf16 (64 B) coalesced across SG
//   - V_sigma[n_chunk..n_chunk+32]   : 32 f32 (128 B) coalesced across SG
//   - V_codes[n_chunk+simd_lid, d_base..d_base+32]
//     : 32 separate 32-byte loads (row-strided across threads). NOT
//       fully coalesced — each thread reads its own row's 32-byte
//       slice. Apple GPU's L2 caches the rows (V_codes ≤ 16 MB at
//       8K context, fits in L2) → in practice the strided pattern
//       costs less than v1's serial N loop saved.
//
// Constraints:
//   - D == 256 (Gemma 4 head_dim). Other D values require new
//     instantiation (NSG / D_TILE adjustments).
//   - n_levels ≤ 16 (bits ≤ 4)
//   - H % H_kv == 0  (GQA)

constant uint TQ_SV_MAX_NCENTROIDS  = 16;
constant uint TQ_SV_NSG             = 2;
constant uint TQ_SV_D_TILE_PER_SG   = 32;
constant uint TQ_SV_D_TILE_PER_TG   = TQ_SV_NSG * TQ_SV_D_TILE_PER_SG;  // 64
constant uint TQ_SV_SIMD_WIDTH      = 32;

kernel void lumen_tq_sv_inline(
    device const bfloat* __restrict__ S         [[buffer(0)]],
    device const uchar*  __restrict__ V_codes   [[buffer(1)]],
    device const float*  __restrict__ V_sigma   [[buffer(2)]],
    device const float*  __restrict__ centroids [[buffer(3)]],
    device       bfloat*              O         [[buffer(4)]],
    constant uint& B    [[buffer(5)]],
    constant uint& H    [[buffer(6)]],
    constant uint& H_kv [[buffer(7)]],
    constant uint& T    [[buffer(8)]],
    constant uint& N    [[buffer(9)]],
    constant uint& D    [[buffer(10)]],
    constant uint& n_levels [[buffer(11)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]]
) {
    threadgroup float centroids_tg[TQ_SV_MAX_NCENTROIDS];

    // Grid: (d_tile_idx, t, b·h). D outputs in [d_base, d_base+D_TILE_PER_SG)
    // per simdgroup.
    const uint d_tile_idx = tid.x;
    const uint t          = tid.y;
    const uint bh         = tid.z;
    const uint b          = bh / H;
    const uint h          = bh % H;
    const uint h_kv       = h * H_kv / H;
    const uint d_base     = d_tile_idx * TQ_SV_D_TILE_PER_TG
                          + simd_gid * TQ_SV_D_TILE_PER_SG;

    // Cooperative LUT load (first n_levels threads of first simdgroup).
    if (simd_gid == 0 && simd_lid < n_levels) {
        centroids_tg[simd_lid] = centroids[simd_lid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint s_base       = ((b * H    + h)    * T + t) * N;
    const uint sigma_base   = (b * H_kv  + h_kv) * N;
    const uint v_codes_base = (b * H_kv  + h_kv) * N * D;

    // Per-thread accumulators for the simdgroup's D_TILE_PER_SG outputs.
    float result[TQ_SV_D_TILE_PER_SG] = {0};

    // Walk N in chunks of SIMD_WIDTH=32. Each thread handles one n per
    // iteration (n = n_chunk + simd_lid); 32 threads in parallel.
    for (uint n_chunk = 0; n_chunk < N; n_chunk += TQ_SV_SIMD_WIDTH) {
        uint n = n_chunk + simd_lid;
        float w_n = 0.0f;
        if (n < N) {
            float x_n   = float(S[s_base + n]);
            float sig_n = V_sigma[sigma_base + n];
            w_n = x_n * sig_n;
        }
        if (n < N) {
            device const uchar* v_row =
                V_codes + v_codes_base + n * D + d_base;
            #pragma clang loop unroll(full)
            for (uint k = 0; k < TQ_SV_D_TILE_PER_SG; k++) {
                uchar code = v_row[k];
                result[k] = fma(w_n, centroids_tg[code], result[k]);
            }
        }
    }

    // simd_sum across the 32 threads in the simdgroup reduces the N
    // dimension (which was striped across threads). simd_lid==0 writes
    // out each output element.
    #pragma clang loop unroll(full)
    for (uint k = 0; k < TQ_SV_D_TILE_PER_SG; k++) {
        float sum = simd_sum(result[k]);
        if (simd_lid == 0) {
            uint d_out = d_base + k;
            if (d_out < D) {
                O[((b * H + h) * T + t) * D + d_out] = bfloat(sum);
            }
        }
    }
}
)KMETAL";

} // anonymous namespace

void TurboquantEncode::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const auto& x = inputs[0];          // bf16 [..., D]
  const auto& boundaries = inputs[1]; // f32  [n_inner]
  auto& codes = outputs[0];           // uint8 [..., D]

  codes.set_data(allocator::malloc(codes.nbytes()));

  auto& s = stream();
  auto& d = metal::device(s.device);

  // Load (and cache) the shader library + pipeline.
  auto lib = d.get_library(tq_lib_name(), []() {
    return std::string(TQ_SHADER_SRC);
  });
  auto kernel = d.get_kernel("lumen_tq_encode", lib);

  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);
  enc.set_input_array(x, 0);
  enc.set_input_array(boundaries, 1);
  enc.set_output_array(codes, 2);

  uint32_t n_inner = static_cast<uint32_t>(boundaries.size());
  uint32_t n_elements = static_cast<uint32_t>(x.size());
  enc.set_bytes(n_inner, 3);
  enc.set_bytes(n_elements, 4);

  // Grid = ceil(n_elements / 256), TG = 256.
  constexpr uint32_t TG = 256;
  uint32_t n_groups = (n_elements + TG - 1) / TG;
  MTL::Size grid = MTL::Size(n_groups, 1, 1);
  MTL::Size tg = MTL::Size(TG, 1, 1);
  enc.dispatch_threadgroups(grid, tg);
}

namespace {

const std::string& qjl_lib_name() {
  static const std::string name = "lumen_qjl_pack";
  return name;
}

constexpr const char* QJL_PACK_SHADER_SRC = R"KMETAL(
#include <metal_stdlib>
using namespace metal;

// ── QJL pack signs: read 32 f32 values, set bit if value ≥ 0 ─────────────
//
// Buffers:
//   [0] values   [N_rows × m]            float32
//   [1] packed   [N_rows × n_words]      uint32   (n_words = ceil(m/32))
//   [2] m        uint32
//   [3] n_words  uint32
//
// Grid: one thread per output u32 word; threads laid out as
// `(N_rows × n_words, 1, 1)`.

kernel void lumen_qjl_pack_signs(
    device const float* __restrict__ values [[buffer(0)]],
    device       uint*               packed [[buffer(1)]],
    constant uint& m       [[buffer(2)]],
    constant uint& n_words [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    // gid encodes (row, word). Each row has n_words u32 outputs.
    uint row  = gid / n_words;
    uint word = gid % n_words;
    uint base_v = row * m + word * 32u;

    uint w = 0u;
    uint last = min(32u, m - word * 32u);
    for (uint b = 0; b < last; b++) {
        if (values[base_v + b] >= 0.0f) {
            w |= (1u << b);
        }
    }
    packed[gid] = w;
}

// ── QJL unpack signs: emit ±1 bf16 per bit ───────────────────────────────
//
// Buffers:
//   [0] packed  [N_rows × n_words] uint32
//   [1] signs   [N_rows × m]       bfloat16   ({-1, +1})
//   [2] m       uint32
//   [3] n_words uint32

kernel void lumen_qjl_unpack_signs(
    device const uint*   __restrict__ packed [[buffer(0)]],
    device       bfloat*              signs  [[buffer(1)]],
    constant uint& m       [[buffer(2)]],
    constant uint& n_words [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    // gid encodes (row, i) — one thread per output sign.
    uint row = gid / m;
    uint i   = gid % m;
    uint word_idx = i >> 5;     // i / 32
    uint bit_idx  = i & 31u;    // i % 32
    uint w = packed[row * n_words + word_idx];
    bool is_pos = ((w >> bit_idx) & 1u) != 0u;
    signs[gid] = is_pos ? bfloat(1.0) : bfloat(-1.0);
}
)KMETAL";

} // anonymous namespace

void QjlPackSigns::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const auto& values = inputs[0];
  auto& packed = outputs[0];

  packed.set_data(allocator::malloc(packed.nbytes()));
  if (values.size() == 0) {
    return;
  }

  auto& s = stream();
  auto& d = metal::device(s.device);

  auto lib = d.get_library(qjl_lib_name(), []() {
    return std::string(QJL_PACK_SHADER_SRC);
  });
  auto kernel = d.get_kernel("lumen_qjl_pack_signs", lib);

  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);
  enc.set_input_array(values, 0);
  enc.set_output_array(packed, 1);

  uint32_t m = static_cast<uint32_t>(m_);
  uint32_t n_words = (m + 31u) / 32u;
  enc.set_bytes(m, 2);
  enc.set_bytes(n_words, 3);

  uint32_t total = static_cast<uint32_t>(packed.size());
  constexpr uint32_t TG = 256;
  uint32_t n_groups = (total + TG - 1) / TG;
  MTL::Size grid = MTL::Size(total, 1, 1);
  MTL::Size tg = MTL::Size(std::min<uint32_t>(total, TG), 1, 1);
  (void)n_groups;
  enc.dispatch_threads(grid, tg);
}

void QjlUnpackSigns::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const auto& packed = inputs[0];
  auto& signs = outputs[0];

  signs.set_data(allocator::malloc(signs.nbytes()));
  if (signs.size() == 0) {
    return;
  }

  auto& s = stream();
  auto& d = metal::device(s.device);

  auto lib = d.get_library(qjl_lib_name(), []() {
    return std::string(QJL_PACK_SHADER_SRC);
  });
  auto kernel = d.get_kernel("lumen_qjl_unpack_signs", lib);

  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);
  enc.set_input_array(packed, 0);
  enc.set_output_array(signs, 1);

  uint32_t m = static_cast<uint32_t>(m_);
  uint32_t n_words = (m + 31u) / 32u;
  enc.set_bytes(m, 2);
  enc.set_bytes(n_words, 3);

  uint32_t total = static_cast<uint32_t>(signs.size());
  constexpr uint32_t TG = 256;
  MTL::Size grid = MTL::Size(total, 1, 1);
  MTL::Size tg = MTL::Size(std::min<uint32_t>(total, TG), 1, 1);
  enc.dispatch_threads(grid, tg);
}

void TurboquantEncodeFused::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const auto& x = inputs[0];          // bf16 [..., D]
  const auto& boundaries = inputs[1]; // f32  [n_inner]
  auto& codes = outputs[0];           // uint8 [..., D]
  auto& sigma = outputs[1];           // f32   [..., 1]

  codes.set_data(allocator::malloc(codes.nbytes()));
  sigma.set_data(allocator::malloc(sigma.nbytes()));

  if (x.size() == 0) {
    return;
  }

  auto& s = stream();
  auto& d = metal::device(s.device);

  // Shared library with single-output encoder — both kernels live in
  // the same shader source compiled once.
  auto lib = d.get_library(tq_lib_name(), []() {
    return std::string(TQ_SHADER_SRC);
  });
  auto kernel = d.get_kernel("lumen_tq_encode_fused", lib);

  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);
  enc.set_input_array(x, 0);
  enc.set_input_array(boundaries, 1);
  enc.set_output_array(codes, 2);
  enc.set_output_array(sigma, 3);

  uint32_t n_inner = static_cast<uint32_t>(boundaries.size());
  uint32_t D = static_cast<uint32_t>(x.shape(-1));
  enc.set_bytes(n_inner, 4);
  enc.set_bytes(D, 5);

  // One threadgroup per row, D threads per group.
  uint32_t n_rows = static_cast<uint32_t>(x.size() / D);
  MTL::Size grid = MTL::Size(n_rows, 1, 1);
  MTL::Size tg = MTL::Size(D, 1, 1);
  enc.dispatch_threadgroups(grid, tg);
}

void TurboquantQkInline::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const auto& q         = inputs[0];  // bf16  [B, H, T, D]
  const auto& k_codes   = inputs[1];  // uint8 [B, H_kv, N, D]
  const auto& k_sigma   = inputs[2];  // f32   [B, H_kv, N]
  const auto& centroids = inputs[3];  // f32   [n_levels]
  auto& scores          = outputs[0]; // bf16  [B, H, T, N]

  scores.set_data(allocator::malloc(scores.nbytes()));

  if (scores.size() == 0) {
    return;
  }

  auto& s = stream();
  auto& d = metal::device(s.device);

  auto lib = d.get_library(tq_lib_name(), []() {
    return std::string(TQ_SHADER_SRC);
  });
  auto kernel = d.get_kernel("lumen_tq_qk_inline", lib);

  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);
  enc.set_input_array(q, 0);
  enc.set_input_array(k_codes, 1);
  enc.set_input_array(k_sigma, 2);
  enc.set_input_array(centroids, 3);
  enc.set_output_array(scores, 4);

  uint32_t B    = static_cast<uint32_t>(q.shape(0));
  uint32_t H    = static_cast<uint32_t>(q.shape(1));
  uint32_t T    = static_cast<uint32_t>(q.shape(2));
  uint32_t D    = static_cast<uint32_t>(q.shape(3));
  uint32_t H_kv = static_cast<uint32_t>(k_codes.shape(1));
  uint32_t N    = static_cast<uint32_t>(k_codes.shape(2));
  uint32_t n_levels = static_cast<uint32_t>(centroids.size());

  enc.set_bytes(B,        5);
  enc.set_bytes(H,        6);
  enc.set_bytes(H_kv,     7);
  enc.set_bytes(T,        8);
  enc.set_bytes(N,        9);
  enc.set_bytes(D,        10);
  enc.set_bytes(n_levels, 11);

  // Grid:  (ceil(N/8), T, B·H)
  // TG:    (NSG · SIMD_SIZE = 64, 1, 1)
  constexpr uint32_t ROWS_PER_TG = 8;
  constexpr uint32_t TG_SIZE = 64;
  uint32_t n_groups = (N + ROWS_PER_TG - 1) / ROWS_PER_TG;
  MTL::Size grid = MTL::Size(n_groups, T, B * H);
  MTL::Size tg   = MTL::Size(TG_SIZE, 1, 1);
  enc.dispatch_threadgroups(grid, tg);
}

void TurboquantEncodeFusedPacked4::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const auto& x = inputs[0];          // bf16 [..., D]
  const auto& boundaries = inputs[1]; // f32  [n_inner]
  auto& codes_pkd = outputs[0];       // uint32 [..., D/8]
  auto& sigma = outputs[1];           // f32   [..., 1]

  codes_pkd.set_data(allocator::malloc(codes_pkd.nbytes()));
  sigma.set_data(allocator::malloc(sigma.nbytes()));

  if (x.size() == 0) {
    return;
  }

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto lib = d.get_library(tq_lib_name(), []() {
    return std::string(TQ_SHADER_SRC);
  });
  auto kernel = d.get_kernel("lumen_tq_encode_fused_packed4", lib);

  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);
  enc.set_input_array(x, 0);
  enc.set_input_array(boundaries, 1);
  enc.set_output_array(codes_pkd, 2);
  enc.set_output_array(sigma, 3);

  uint32_t n_inner = static_cast<uint32_t>(boundaries.size());
  uint32_t D = static_cast<uint32_t>(x.shape(-1));
  enc.set_bytes(n_inner, 4);
  enc.set_bytes(D, 5);

  uint32_t n_rows = static_cast<uint32_t>(x.size() / D);
  MTL::Size grid = MTL::Size(n_rows, 1, 1);
  MTL::Size tg = MTL::Size(D, 1, 1);
  enc.dispatch_threadgroups(grid, tg);
}

void TurboquantQkInlinePacked4::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const auto& q           = inputs[0]; // bf16  [B, H, T, D]
  const auto& k_codes_pkd = inputs[1]; // uint32[B, H_kv, N, D/8]
  const auto& k_sigma     = inputs[2]; // f32   [B, H_kv, N]
  const auto& centroids   = inputs[3]; // f32   [n_levels]
  auto& scores            = outputs[0];// bf16  [B, H, T, N]

  scores.set_data(allocator::malloc(scores.nbytes()));
  if (scores.size() == 0) {
    return;
  }

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto lib = d.get_library(tq_lib_name(), []() {
    return std::string(TQ_SHADER_SRC);
  });
  auto kernel = d.get_kernel("lumen_tq_qk_inline_packed4", lib);

  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);
  enc.set_input_array(q, 0);
  enc.set_input_array(k_codes_pkd, 1);
  enc.set_input_array(k_sigma, 2);
  enc.set_input_array(centroids, 3);
  enc.set_output_array(scores, 4);

  uint32_t B    = static_cast<uint32_t>(q.shape(0));
  uint32_t H    = static_cast<uint32_t>(q.shape(1));
  uint32_t T    = static_cast<uint32_t>(q.shape(2));
  uint32_t D    = static_cast<uint32_t>(q.shape(3));
  uint32_t H_kv = static_cast<uint32_t>(k_codes_pkd.shape(1));
  uint32_t N    = static_cast<uint32_t>(k_codes_pkd.shape(2));
  uint32_t n_levels = static_cast<uint32_t>(centroids.size());

  enc.set_bytes(B,        5);
  enc.set_bytes(H,        6);
  enc.set_bytes(H_kv,     7);
  enc.set_bytes(T,        8);
  enc.set_bytes(N,        9);
  enc.set_bytes(D,        10);
  enc.set_bytes(n_levels, 11);

  constexpr uint32_t ROWS_PER_TG = 8;
  constexpr uint32_t TG_SIZE = 64;
  uint32_t n_groups = (N + ROWS_PER_TG - 1) / ROWS_PER_TG;
  MTL::Size grid = MTL::Size(n_groups, T, B * H);
  MTL::Size tg   = MTL::Size(TG_SIZE, 1, 1);
  enc.dispatch_threadgroups(grid, tg);
}

void TurboquantSvInline::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const auto& s_in      = inputs[0]; // bf16  [B, H, T, N]
  const auto& v_codes   = inputs[1]; // uint8 [B, H_kv, N, D]
  const auto& v_sigma   = inputs[2]; // f32   [B, H_kv, N]
  const auto& centroids = inputs[3]; // f32   [n_levels]
  auto& out             = outputs[0];// bf16  [B, H, T, D]

  out.set_data(allocator::malloc(out.nbytes()));

  if (out.size() == 0) {
    return;
  }

  auto& stream_ = stream();
  auto& d = metal::device(stream_.device);

  auto lib = d.get_library(tq_lib_name(), []() {
    return std::string(TQ_SHADER_SRC);
  });
  auto kernel = d.get_kernel("lumen_tq_sv_inline", lib);

  auto& enc = d.get_command_encoder(stream_.index);
  enc.set_compute_pipeline_state(kernel);
  enc.set_input_array(s_in, 0);
  enc.set_input_array(v_codes, 1);
  enc.set_input_array(v_sigma, 2);
  enc.set_input_array(centroids, 3);
  enc.set_output_array(out, 4);

  uint32_t B    = static_cast<uint32_t>(s_in.shape(0));
  uint32_t H    = static_cast<uint32_t>(s_in.shape(1));
  uint32_t T    = static_cast<uint32_t>(s_in.shape(2));
  uint32_t N    = static_cast<uint32_t>(s_in.shape(3));
  uint32_t H_kv = static_cast<uint32_t>(v_codes.shape(1));
  uint32_t D_   = static_cast<uint32_t>(v_codes.shape(3));
  uint32_t n_levels = static_cast<uint32_t>(centroids.size());

  enc.set_bytes(B,        5);
  enc.set_bytes(H,        6);
  enc.set_bytes(H_kv,     7);
  enc.set_bytes(T,        8);
  enc.set_bytes(N,        9);
  enc.set_bytes(D_,       10);
  enc.set_bytes(n_levels, 11);

  // v2 qvm-style layout: 2 simdgroups × 32 threads = 64 threads/TG.
  // Grid: (D/64, T, B·H) threadgroups. 4× v1's TG count for D=256.
  // Phase 0 sweep concluded NEUTRAL (variant A NSG=4 also matched, no
  // tile-tweak breaks the ceiling rule).
  constexpr uint32_t NSG = 2;
  constexpr uint32_t D_TILE_PER_SG = 32;
  constexpr uint32_t D_TILE_PER_TG = NSG * D_TILE_PER_SG;  // 64
  constexpr uint32_t SIMD_WIDTH = 32;
  uint32_t n_d_tiles = (D_ + D_TILE_PER_TG - 1) / D_TILE_PER_TG;
  MTL::Size grid = MTL::Size(n_d_tiles, T, B * H);
  MTL::Size tg   = MTL::Size(NSG * SIMD_WIDTH, 1, 1);
  enc.dispatch_threadgroups(grid, tg);
}

void TurboquantRotEncodeFused::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const auto& x = inputs[0];          // bf16 [..., D]
  const auto& R = inputs[1];          // f32  [D, D]
  const auto& boundaries = inputs[2]; // f32  [n_inner]
  auto& codes = outputs[0];           // uint8 [..., D]
  auto& sigma = outputs[1];           // f32   [..., 1]

  codes.set_data(allocator::malloc(codes.nbytes()));
  sigma.set_data(allocator::malloc(sigma.nbytes()));

  if (x.size() == 0) {
    return;
  }

  auto& s = stream();
  auto& d = metal::device(s.device);

  auto lib = d.get_library(tq_lib_name(), []() {
    return std::string(TQ_SHADER_SRC);
  });
  auto kernel = d.get_kernel("lumen_tq_rot_encode_fused", lib);

  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);
  enc.set_input_array(x, 0);
  enc.set_input_array(R, 1);
  enc.set_input_array(boundaries, 2);
  enc.set_output_array(codes, 3);
  enc.set_output_array(sigma, 4);

  uint32_t n_inner = static_cast<uint32_t>(boundaries.size());
  uint32_t D = static_cast<uint32_t>(x.shape(-1));
  enc.set_bytes(n_inner, 5);
  enc.set_bytes(D, 6);

  // One threadgroup per row, D threads per group.
  uint32_t n_rows = static_cast<uint32_t>(x.size() / D);
  MTL::Size grid = MTL::Size(n_rows, 1, 1);
  MTL::Size tg = MTL::Size(D, 1, 1);
  enc.dispatch_threadgroups(grid, tg);
}

} // namespace mlx::core::lumen
