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
// Constraints (enforced at factory):
//   - D == TQ_QK_VPT · SIMD_SIZE (i.e. D % 32 == 0; VPT is a function-constant
//     specialization parameter — VPT=8 for D=256 (sliding head_dim), VPT=16
//     for D=512 (Gemma 4 full-attn global_head_dim))
//   - n_levels ≤ 16 (bits ≤ 4)

constant uint TQ_QK_NSG        = 2;    // simdgroups per TG
constant uint TQ_QK_RPS        = 4;    // N-rows per simdgroup
constant uint TQ_QK_VPT        = 8;    // D-elements per thread (D=256 variant)
constant uint TQ_QK_VPT_D512   = 16;   // D-elements per thread (D=512 variant)
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

// ── TurboQuant Stage-1 qk inline matmul (D=512 variant) ───────────────────
//
// Same kernel as `lumen_tq_qk_inline` but with VPT=16 instead of VPT=8 so
// 32·16 = 512 D-elements per simdgroup → handles Gemma 4 full-attention
// global_head_dim=512. Source duplicated rather than templated because
// Metal function constants cannot be used as array sizes (compile error),
// and a max-sized scratch array of 16 floats wastes 32 B/thread for the
// D=256 case — small but real on the dispatch-bound decode path.
//
// Shape/contract identical to lumen_tq_qk_inline. Constraints:
//   - D == 512 (kernel hardcodes VPT=16)
//   - n_levels ≤ 16 (bits ≤ 4)
//   - H % H_kv == 0 (GQA)

kernel void lumen_tq_qk_inline_d512(
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
    const uint h_kv = h * H_kv / H;

    const uint n_base = nb * TQ_QK_ROWS_PER_TG + simd_gid * TQ_QK_RPS;

    if (simd_gid == 0 && simd_lid < n_levels) {
        centroids_tg[simd_lid] = centroids[simd_lid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // VPT=16 D-elements per thread → 32 threads × 16 = 512 = D.
    float x_thread[TQ_QK_VPT_D512];
    const uint q_base = ((b * H + h) * T + t) * D + simd_lid * TQ_QK_VPT_D512;
    for (uint v = 0; v < TQ_QK_VPT_D512; v++) {
        x_thread[v] = float(Q[q_base + v]);
    }

    float result[TQ_QK_RPS] = {0.0f, 0.0f, 0.0f, 0.0f};

    const uint K_block_base = (b * H_kv + h_kv) * N * D;

    for (uint row = 0; row < TQ_QK_RPS; row++) {
        uint n = n_base + row;
        if (n >= N) {
            continue;
        }
        const device uchar* K_row =
            K_codes + K_block_base + n * D + simd_lid * TQ_QK_VPT_D512;
        float partial = 0.0f;
        for (uint v = 0; v < TQ_QK_VPT_D512; v++) {
            uchar code = K_row[v];
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

// ── sv_inline v3 ─ uchar4 vectorized V_codes load ────────────────────────
// Same threading pattern + memory layout as v2; only the V_codes load
// changes from byte-by-byte `v_row[k]` (32 individual 1-byte loads per
// N-iteration per thread) to `uchar4` vector (8 4-byte loads per
// N-iteration per thread). Apple Silicon GPU coalesces the 4-byte loads
// into a single transaction, cutting DRAM ops 4× on the V_codes side.
//
// Constraint: D_TILE_PER_SG must be a multiple of 4 (it's 32 — safe).
// V_codes pointer must be 4-byte aligned (`v_codes_base + n*D + d_base`
// — D is multiple of 64 on Gemma 4, d_base is multiple of 32, so the
// effective offset is multiple of 32 ≥ 4 → aligned).

kernel void lumen_tq_sv_inline_v3(
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

    const uint d_tile_idx = tid.x;
    const uint t          = tid.y;
    const uint bh         = tid.z;
    const uint b          = bh / H;
    const uint h          = bh % H;
    const uint h_kv       = h * H_kv / H;
    const uint d_base     = d_tile_idx * TQ_SV_D_TILE_PER_TG
                          + simd_gid * TQ_SV_D_TILE_PER_SG;

    if (simd_gid == 0 && simd_lid < n_levels) {
        centroids_tg[simd_lid] = centroids[simd_lid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint s_base       = ((b * H    + h)    * T + t) * N;
    const uint sigma_base   = (b * H_kv  + h_kv) * N;
    const uint v_codes_base = (b * H_kv  + h_kv) * N * D;

    float result[TQ_SV_D_TILE_PER_SG] = {0};

    for (uint n_chunk = 0; n_chunk < N; n_chunk += TQ_SV_SIMD_WIDTH) {
        uint n = n_chunk + simd_lid;
        float w_n = 0.0f;
        if (n < N) {
            float x_n   = float(S[s_base + n]);
            float sig_n = V_sigma[sigma_base + n];
            w_n = x_n * sig_n;
        }
        if (n < N) {
            // Cast to uchar4 for 4-byte vector loads. D_TILE_PER_SG/4 = 8
            // loads of 4 bytes vs 32 loads of 1 byte (v2). Apple Metal
            // coalesces aligned 4-byte loads into a single DRAM transaction.
            device const uchar4* v_row4 = (device const uchar4*)(
                V_codes + v_codes_base + n * D + d_base);
            #pragma clang loop unroll(full)
            for (uint k4 = 0; k4 < TQ_SV_D_TILE_PER_SG / 4; k4++) {
                uchar4 codes4 = v_row4[k4];
                const uint k = k4 * 4;
                result[k + 0] = fma(w_n, centroids_tg[codes4.x], result[k + 0]);
                result[k + 1] = fma(w_n, centroids_tg[codes4.y], result[k + 1]);
                result[k + 2] = fma(w_n, centroids_tg[codes4.z], result[k + 2]);
                result[k + 3] = fma(w_n, centroids_tg[codes4.w], result[k + 3]);
            }
        }
    }

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

// ── TurboQuant fused attention (online softmax + inline K/V dequant) ─────
//
// Computes O[B, H, T=1, D] = softmax(Q · K_dq^T / sqrt(D)) · V_dq in a
// single dispatch. Mirrors mlx's `sdpa_vector` flash-attention pattern but
// loads K and V via Lloyd-Max inline dequant (centroids LUT × per-vector σ)
// instead of bf16 reads. Eliminates the qk_inline + softmax + sv_inline
// 3-dispatch chain currently used for the TQ decode path.
//
// Shape:
//   Q         : bfloat16 [B, H,    T, D]   (T must equal 1 for now)
//   K_codes   : uint8    [B, H_kv, N, D]
//   K_sigma   : float32  [B, H_kv, N]
//   V_codes   : uint8    [B, H_kv, N, D]
//   V_sigma   : float32  [B, H_kv, N]
//   centrds   : float32  [n_levels]        (Lloyd-Max LUT; n_levels ≤ 16)
//   O         : bfloat16 [B, H,    T, D]
//
// Threading (sdpa_vector layout, source-duplicated for D = {256, 512}):
//   - BN = 32 simdgroups per threadgroup, BD = 32 threads/SG = simd width
//   - Each thread holds qk_per_thread = D/BD = {8, 16} elements of Q
//     plus v_per_thread = D/BD elements of running V accumulator
//   - Each simdgroup processes N rows i = simd_gid, simd_gid + BN, ...
//   - Within a simdgroup, simd_lid covers the D dimension (one BD slice
//     per thread); simd_sum reduces the partial QK score
//   - Online softmax: register-resident max_score, sum_exp_score updated
//     per N row; final cross-simdgroup reduce in threadgroup memory
//
// Inline dequant pattern (matches sv_inline / qk_inline math, just fused):
//   k[j]  = centroids[K_codes[b, h_kv, i, d_thread + j]] * K_sigma[b, h_kv, i]
//   v[j]  = centroids[V_codes[b, h_kv, i, d_thread + j]] * V_sigma[b, h_kv, i]
//
// Constraints:
//   - T == 1 (decode shape)
//   - D ∈ {256, 512}  (source-duplicated kernel pairs)
//   - n_levels ≤ 16  (bits ≤ 4)
//   - H % H_kv == 0  (GQA)

constant uint TQ_FA_MAX_NCENTROIDS = 16;
constant uint TQ_FA_BN             = 32;  // simdgroups per TG
constant uint TQ_FA_BD             = 32;  // threads per SG (= simd width)

// D=256 variant (sliding head_dim).
kernel void lumen_tq_fused_attn_d256(
    const device bfloat* __restrict__ Q         [[buffer(0)]],
    const device uchar*  __restrict__ K_codes   [[buffer(1)]],
    const device float*  __restrict__ K_sigma   [[buffer(2)]],
    const device uchar*  __restrict__ V_codes   [[buffer(3)]],
    const device float*  __restrict__ V_sigma   [[buffer(4)]],
    const device float*  __restrict__ centroids [[buffer(5)]],
    device       bfloat*              O         [[buffer(6)]],
    constant uint& B        [[buffer(7)]],
    constant uint& H        [[buffer(8)]],
    constant uint& H_kv     [[buffer(9)]],
    constant uint& T_q      [[buffer(10)]],
    constant uint& N        [[buffer(11)]],
    constant uint& n_levels [[buffer(12)]],
    constant float& scale   [[buffer(13)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]]
) {
    constexpr uint D = 256;
    constexpr uint qk_per_thread = D / TQ_FA_BD;   // 8
    constexpr uint v_per_thread  = D / TQ_FA_BD;   // 8

    typedef float U;

    thread U q[qk_per_thread];
    thread U k[qk_per_thread];
    thread U o[v_per_thread];

    threadgroup float centroids_tg[TQ_FA_MAX_NCENTROIDS];
    threadgroup U max_scores[TQ_FA_BN];
    threadgroup U sum_exp_scores[TQ_FA_BN];
    threadgroup U outputs[TQ_FA_BN * TQ_FA_BD];

    // Cooperative centroids LUT load (first n_levels threads of first SG).
    if (simd_gid == 0 && simd_lid < n_levels) {
        centroids_tg[simd_lid] = centroids[simd_lid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Grid: (B*H, T_q, 1). T_q must be 1 — guarded host-side.
    const uint q_batch_head_idx = tid.x;
    const uint t                = tid.y;
    const uint b                = q_batch_head_idx / H;
    const uint h                = q_batch_head_idx % H;
    const uint h_kv             = h * H_kv / H;

    const uint q_offset       = ((b * H    + h)    * T_q + t) * D;
    const uint kv_code_base   = (b * H_kv  + h_kv) * N   * D;
    const uint sigma_base     = (b * H_kv  + h_kv) * N;
    const uint o_offset       = q_offset;

    // Load Q with attention scale. thread `simd_lid` holds D-slice
    // [simd_lid * qk_per_thread, (simd_lid + 1) * qk_per_thread).
    #pragma clang loop unroll(full)
    for (uint j = 0; j < qk_per_thread; j++) {
        q[j] = U(scale) * U(Q[q_offset + simd_lid * qk_per_thread + j]);
    }
    #pragma clang loop unroll(full)
    for (uint j = 0; j < v_per_thread; j++) {
        o[j] = 0;
    }

    U max_score = -INFINITY;
    U sum_exp_score = 0;

    // Iterate over N (KV positions). simd_gid stripes the N dimension
    // across the BN simdgroups — each SG handles rows i = simd_gid,
    // simd_gid + BN, ...
    for (uint i = simd_gid; i < N; i += TQ_FA_BN) {
        // Load K row: dequant inline. simd_lid handles its D-slice.
        const uint k_row_base = kv_code_base + i * D + simd_lid * qk_per_thread;
        const U    k_sig      = K_sigma[sigma_base + i];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < qk_per_thread; j++) {
            uchar code = K_codes[k_row_base + j];
            k[j] = centroids_tg[code] * k_sig;
        }

        // Compute partial score over the thread's D-slice + simd-wide
        // reduce to get the full score for this (q, k_i).
        U score = 0;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < qk_per_thread; j++) {
            score += q[j] * k[j];
        }
        score = simd_sum(score);

        // Online softmax accumulation.
        U new_max = max(max_score, score);
        U factor    = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score     - new_max);

        max_score     = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        // Load V row inline-dequant and update output accumulator.
        const uint v_row_base = kv_code_base + i * D + simd_lid * v_per_thread;
        const U    v_sig      = V_sigma[sigma_base + i];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < v_per_thread; j++) {
            uchar code = V_codes[v_row_base + j];
            U v_val = centroids_tg[code] * v_sig;
            o[j] = o[j] * factor + exp_score * v_val;
        }
    }

    // Cross-simdgroup reduction of max_score + sum_exp_score (BN partial
    // streams → one global stream).
    if (simd_lid == 0) {
        max_scores[simd_gid]     = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread reads one BN value (simd_lid < BN), simd reduce gives
    // global max / sum. simd_lid runs 0..BD-1 = 0..31 and BN=32 — perfect
    // alignment so all 32 threads of SG 0 read all BN entries.
    U sg_max = max_scores[simd_lid];
    U new_max = simd_max(sg_max);
    U factor    = fast::exp(sg_max - new_max);
    U sg_sum    = simd_sum(sum_exp_scores[simd_lid] * factor);

    // Aggregate outputs (mlx sdpa_vector pattern). Per-iteration:
    //   1. each thread (simd_lid, simd_gid) writes its o[j] into
    //      outputs[simd_lid * BD + simd_gid] — fills a 32×32 staging tile
    //   2. threadgroup barrier so all writes are visible
    //   3. each thread (simd_lid, simd_gid) reads outputs[simd_gid * BD +
    //      simd_lid] (transposed access) and simd_sum across the
    //      simdgroup, weighted by factor (which is simd_lid-dependent).
    //   4. divide by global sum_exp; store back into register o[j]
    //
    // After the v_per_thread loop completes, each simdgroup `simd_gid`
    // owns the contiguous v_per_thread elements of the final output at
    // offset (simd_gid * v_per_thread). simd_lid==0 writes those out.
    #pragma clang loop unroll(full)
    for (uint j = 0; j < v_per_thread; j++) {
        outputs[simd_lid * TQ_FA_BD + simd_gid] = o[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        U partial  = outputs[simd_gid * TQ_FA_BD + simd_lid];
        U combined = simd_sum(partial * factor);
        if (sg_sum != 0) {
            combined = combined / sg_sum;
        }
        o[j] = combined;  // store back into register
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (simd_lid == 0) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < v_per_thread; j++) {
            const uint d_out = simd_gid * v_per_thread + j;
            O[o_offset + d_out] = bfloat(o[j]);
        }
    }
}

// D=512 variant (full-attn global_head_dim). Same algorithm, larger
// per-thread D slice (qk_per_thread = 16). Register pressure ≈ 16*4 + 16*4
// + 16*4 = 192 B/thread for q/k/o — within Apple Silicon thread context.
kernel void lumen_tq_fused_attn_d512(
    const device bfloat* __restrict__ Q         [[buffer(0)]],
    const device uchar*  __restrict__ K_codes   [[buffer(1)]],
    const device float*  __restrict__ K_sigma   [[buffer(2)]],
    const device uchar*  __restrict__ V_codes   [[buffer(3)]],
    const device float*  __restrict__ V_sigma   [[buffer(4)]],
    const device float*  __restrict__ centroids [[buffer(5)]],
    device       bfloat*              O         [[buffer(6)]],
    constant uint& B        [[buffer(7)]],
    constant uint& H        [[buffer(8)]],
    constant uint& H_kv     [[buffer(9)]],
    constant uint& T_q      [[buffer(10)]],
    constant uint& N        [[buffer(11)]],
    constant uint& n_levels [[buffer(12)]],
    constant float& scale   [[buffer(13)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]]
) {
    constexpr uint D = 512;
    constexpr uint qk_per_thread = D / TQ_FA_BD;   // 16
    constexpr uint v_per_thread  = D / TQ_FA_BD;   // 16

    typedef float U;

    thread U q[qk_per_thread];
    thread U k[qk_per_thread];
    thread U o[v_per_thread];

    threadgroup float centroids_tg[TQ_FA_MAX_NCENTROIDS];
    threadgroup U max_scores[TQ_FA_BN];
    threadgroup U sum_exp_scores[TQ_FA_BN];
    threadgroup U outputs[TQ_FA_BN * TQ_FA_BD];

    if (simd_gid == 0 && simd_lid < n_levels) {
        centroids_tg[simd_lid] = centroids[simd_lid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint q_batch_head_idx = tid.x;
    const uint t                = tid.y;
    const uint b                = q_batch_head_idx / H;
    const uint h                = q_batch_head_idx % H;
    const uint h_kv             = h * H_kv / H;

    const uint q_offset       = ((b * H    + h)    * T_q + t) * D;
    const uint kv_code_base   = (b * H_kv  + h_kv) * N   * D;
    const uint sigma_base     = (b * H_kv  + h_kv) * N;
    const uint o_offset       = q_offset;

    #pragma clang loop unroll(full)
    for (uint j = 0; j < qk_per_thread; j++) {
        q[j] = U(scale) * U(Q[q_offset + simd_lid * qk_per_thread + j]);
    }
    #pragma clang loop unroll(full)
    for (uint j = 0; j < v_per_thread; j++) {
        o[j] = 0;
    }

    U max_score = -INFINITY;
    U sum_exp_score = 0;

    for (uint i = simd_gid; i < N; i += TQ_FA_BN) {
        const uint k_row_base = kv_code_base + i * D + simd_lid * qk_per_thread;
        const U    k_sig      = K_sigma[sigma_base + i];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < qk_per_thread; j++) {
            uchar code = K_codes[k_row_base + j];
            k[j] = centroids_tg[code] * k_sig;
        }

        U score = 0;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < qk_per_thread; j++) {
            score += q[j] * k[j];
        }
        score = simd_sum(score);

        U new_max   = max(max_score, score);
        U factor    = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score     - new_max);

        max_score     = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        const uint v_row_base = kv_code_base + i * D + simd_lid * v_per_thread;
        const U    v_sig      = V_sigma[sigma_base + i];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < v_per_thread; j++) {
            uchar code = V_codes[v_row_base + j];
            U v_val = centroids_tg[code] * v_sig;
            o[j] = o[j] * factor + exp_score * v_val;
        }
    }

    if (simd_lid == 0) {
        max_scores[simd_gid]     = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    U sg_max = max_scores[simd_lid];
    U new_max = simd_max(sg_max);
    U factor  = fast::exp(sg_max - new_max);
    U sg_sum  = simd_sum(sum_exp_scores[simd_lid] * factor);

    #pragma clang loop unroll(full)
    for (uint j = 0; j < v_per_thread; j++) {
        outputs[simd_lid * TQ_FA_BD + simd_gid] = o[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        U partial  = outputs[simd_gid * TQ_FA_BD + simd_lid];
        U combined = simd_sum(partial * factor);
        if (sg_sum != 0) {
            combined = combined / sg_sum;
        }
        o[j] = combined;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (simd_lid == 0) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < v_per_thread; j++) {
            const uint d_out = simd_gid * v_per_thread + j;
            O[o_offset + d_out] = bfloat(o[j]);
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

  uint32_t B    = static_cast<uint32_t>(q.shape(0));
  uint32_t H    = static_cast<uint32_t>(q.shape(1));
  uint32_t T    = static_cast<uint32_t>(q.shape(2));
  uint32_t D    = static_cast<uint32_t>(q.shape(3));
  uint32_t H_kv = static_cast<uint32_t>(k_codes.shape(1));
  uint32_t N    = static_cast<uint32_t>(k_codes.shape(2));
  uint32_t n_levels = static_cast<uint32_t>(centroids.size());

  // Dispatch by D to the matching kernel variant. D=256 hits VPT=8
  // (sliding-attn head_dim); D=512 hits VPT=16 (full-attn global_head_dim).
  // Function constants would let us share one source, but Apple's Metal
  // compiler rejects function-constant array sizes (`float a[FC]`), so we
  // duplicate the kernel rather than waste 32 B/thread on a max-size array.
  const char* kernel_name;
  if (D == 256) {
    kernel_name = "lumen_tq_qk_inline";
  } else if (D == 512) {
    kernel_name = "lumen_tq_qk_inline_d512";
  } else {
    throw std::invalid_argument(
        "lumen::TurboquantQkInline: only D ∈ {256, 512} supported; got D=" +
        std::to_string(D));
  }
  auto kernel = d.get_kernel(kernel_name, lib);

  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);
  enc.set_input_array(q, 0);
  enc.set_input_array(k_codes, 1);
  enc.set_input_array(k_sigma, 2);
  enc.set_input_array(centroids, 3);
  enc.set_output_array(scores, 4);

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
  // v3 = uchar4 vectorized V_codes load (4× DRAM transactions reduction).
  // Same threading layout as v2 — output is bit-equivalent.
  const char* env_v3 = std::getenv("LUMEN_TQ_SV_INLINE_V3");
  const bool use_v3 = env_v3 != nullptr && env_v3[0] == '1';
  auto kernel = d.get_kernel(
      use_v3 ? "lumen_tq_sv_inline_v3" : "lumen_tq_sv_inline", lib);

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

void TurboquantFusedAttn::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const auto& q         = inputs[0]; // bf16  [B, H, T, D]
  const auto& k_codes   = inputs[1]; // uint8 [B, H_kv, N, D]
  const auto& k_sigma   = inputs[2]; // f32   [B, H_kv, N]
  const auto& v_codes   = inputs[3]; // uint8 [B, H_kv, N, D]
  const auto& v_sigma   = inputs[4]; // f32   [B, H_kv, N]
  const auto& centroids = inputs[5]; // f32   [n_levels]
  auto& out             = outputs[0]; // bf16 [B, H, T, D]

  out.set_data(allocator::malloc(out.nbytes()));

  if (out.size() == 0) {
    return;
  }

  auto& stream_ = stream();
  auto& d = metal::device(stream_.device);

  auto lib = d.get_library(tq_lib_name(), []() {
    return std::string(TQ_SHADER_SRC);
  });

  // Source-duplicated kernels — dispatch by D. Q/K/V/O all share head_dim.
  uint32_t B    = static_cast<uint32_t>(q.shape(0));
  uint32_t H    = static_cast<uint32_t>(q.shape(1));
  uint32_t T    = static_cast<uint32_t>(q.shape(2));
  uint32_t D_   = static_cast<uint32_t>(q.shape(3));
  uint32_t H_kv = static_cast<uint32_t>(k_codes.shape(1));
  uint32_t N    = static_cast<uint32_t>(k_codes.shape(2));
  uint32_t n_levels = static_cast<uint32_t>(centroids.size());
  float    scale_val = scale();

  const char* kernel_name = nullptr;
  if (D_ == 256) {
    kernel_name = "lumen_tq_fused_attn_d256";
  } else if (D_ == 512) {
    kernel_name = "lumen_tq_fused_attn_d512";
  } else {
    std::ostringstream msg;
    msg << "[TurboquantFusedAttn] D must be 256 or 512, got " << D_;
    throw std::invalid_argument(msg.str());
  }
  auto kernel = d.get_kernel(kernel_name, lib);

  auto& enc = d.get_command_encoder(stream_.index);
  enc.set_compute_pipeline_state(kernel);
  enc.set_input_array(q,         0);
  enc.set_input_array(k_codes,   1);
  enc.set_input_array(k_sigma,   2);
  enc.set_input_array(v_codes,   3);
  enc.set_input_array(v_sigma,   4);
  enc.set_input_array(centroids, 5);
  enc.set_output_array(out,      6);

  enc.set_bytes(B,        7);
  enc.set_bytes(H,        8);
  enc.set_bytes(H_kv,     9);
  enc.set_bytes(T,        10);
  enc.set_bytes(N,        11);
  enc.set_bytes(n_levels, 12);
  enc.set_bytes(scale_val, 13);

  // Grid: (B*H, T, 1) threadgroups × (BN * BD, 1, 1) threads.
  // Matches sdpa_vector layout — one TG per (b, h, t) output position;
  // online softmax + inline dequant runs inside that TG.
  constexpr uint32_t BN = 32;  // simdgroups per TG
  constexpr uint32_t BD = 32;  // threads per SG = simd width
  MTL::Size grid = MTL::Size(B * H, T, 1);
  MTL::Size tg   = MTL::Size(BN * BD, 1, 1);
  enc.dispatch_threadgroups(grid, tg);
}

} // namespace mlx::core::lumen
