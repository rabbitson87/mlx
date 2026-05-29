// lumen-rs Phase 1.8 M4.8 — frontend glue for the FlashAttnBf16 primitive.

#include <sstream>
#include <stdexcept>

#include "mlx/array.h"
#include "mlx/lumen_primitives.h"
#include "mlx/ops.h"  // for astype()

namespace mlx::core::lumen {

namespace {

void check_shape(const array& a, const char* name) {
  if (a.ndim() != 4) {
    std::ostringstream msg;
    msg << "[lumen::flash_attn_bf16] " << name
        << " must be rank-4, got shape " << a.shape();
    throw std::invalid_argument(msg.str());
  }
  if (a.dtype() != bfloat16) {
    std::ostringstream msg;
    msg << "[lumen::flash_attn_bf16] " << name << " must be bfloat16, got "
        << a.dtype();
    throw std::invalid_argument(msg.str());
  }
}

} // anonymous namespace

array flash_attn_bf16(
    const array& q,
    const array& k,
    const array& v,
    float scale,
    std::optional<array> mask,
    StreamOrDevice s_) {
  check_shape(q, "queries");
  check_shape(k, "keys");
  check_shape(v, "values");

  if (q.shape(3) != 256) {
    std::ostringstream msg;
    msg << "[lumen::flash_attn_bf16] head_dim must be 256, got "
        << q.shape(3);
    throw std::invalid_argument(msg.str());
  }
  if (k.shape() != v.shape()) {
    std::ostringstream msg;
    msg << "[lumen::flash_attn_bf16] K/V shape mismatch: k=" << k.shape()
        << " v=" << v.shape();
    throw std::invalid_argument(msg.str());
  }
  if (k.shape(0) != q.shape(0) || k.shape(3) != q.shape(3)) {
    std::ostringstream msg;
    msg << "[lumen::flash_attn_bf16] K/V mismatch with Q: q=" << q.shape()
        << " k=" << k.shape();
    throw std::invalid_argument(msg.str());
  }
  if (k.shape(1) == 0 || q.shape(1) % k.shape(1) != 0) {
    std::ostringstream msg;
    msg << "[lumen::flash_attn_bf16] H=" << q.shape(1)
        << " must be a non-zero multiple of H_kv=" << k.shape(1);
    throw std::invalid_argument(msg.str());
  }

  bool has_mask = mask.has_value();
  std::vector<array> inputs = {q, k, v};
  if (has_mask) {
    if (mask->dtype() != bfloat16) {
      std::ostringstream msg;
      msg << "[lumen::flash_attn_bf16] mask must be bfloat16, got "
          << mask->dtype();
      throw std::invalid_argument(msg.str());
    }
    if (mask->ndim() != 2 || mask->shape(0) != q.shape(2) ||
        mask->shape(1) != k.shape(2)) {
      std::ostringstream msg;
      msg << "[lumen::flash_attn_bf16] mask shape " << mask->shape()
          << " must be [Sq=" << q.shape(2) << ", Skv=" << k.shape(2) << "]";
      throw std::invalid_argument(msg.str());
    }
    inputs.push_back(*mask);
  }

  auto s = to_stream(s_);
  Shape out_shape = {q.shape(0), q.shape(1), q.shape(2), v.shape(-1)};
  return array(
      std::move(out_shape),
      bfloat16,
      std::make_shared<FlashAttnBf16>(s, scale, has_mask),
      std::move(inputs));
}

array flash_attn_prefill_bf16(
    const array& q,
    const array& k,
    const array& v,
    float scale,
    uint32_t window_size,
    uint32_t kv_offset,
    StreamOrDevice s_) {
  check_shape(q, "queries");
  check_shape(k, "keys");
  check_shape(v, "values");

  int head_dim = q.shape(3);
  if (head_dim != 256 && head_dim != 512) {
    std::ostringstream msg;
    msg << "[lumen::flash_attn_prefill_bf16] head_dim must be 256 or 512, got "
        << head_dim;
    throw std::invalid_argument(msg.str());
  }
  if (q.shape(2) <= 1) {
    std::ostringstream msg;
    msg << "[lumen::flash_attn_prefill_bf16] Sq must be > 1 for prefill, got "
        << q.shape(2);
    throw std::invalid_argument(msg.str());
  }
  if (k.shape() != v.shape()) {
    std::ostringstream msg;
    msg << "[lumen::flash_attn_prefill_bf16] K/V shape mismatch: k="
        << k.shape() << " v=" << v.shape();
    throw std::invalid_argument(msg.str());
  }
  if (k.shape(0) != q.shape(0) || k.shape(3) != q.shape(3)) {
    std::ostringstream msg;
    msg << "[lumen::flash_attn_prefill_bf16] K/V mismatch with Q: q="
        << q.shape() << " k=" << k.shape();
    throw std::invalid_argument(msg.str());
  }
  if (k.shape(1) == 0 || q.shape(1) % k.shape(1) != 0) {
    std::ostringstream msg;
    msg << "[lumen::flash_attn_prefill_bf16] H=" << q.shape(1)
        << " must be a non-zero multiple of H_kv=" << k.shape(1);
    throw std::invalid_argument(msg.str());
  }
  if (head_dim == 512 && window_size != 0) {
    throw std::invalid_argument(
        "[lumen::flash_attn_prefill_bf16] head_dim=512 path is causal-only "
        "(window_size must be 0)");
  }

  auto s = to_stream(s_);
  Shape out_shape = {q.shape(0), q.shape(1), q.shape(2), v.shape(-1)};
  std::vector<array> inputs = {q, k, v};
  return array(
      std::move(out_shape),
      bfloat16,
      std::make_shared<FlashAttnPrefillBf16>(s, scale, window_size, kv_offset),
      std::move(inputs));
}

array turboquant_encode(
    const array& x_norm,
    const array& boundaries,
    StreamOrDevice s_) {
  if (x_norm.dtype() != bfloat16) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_encode] x_norm must be bfloat16, got "
        << x_norm.dtype();
    throw std::invalid_argument(msg.str());
  }
  if (boundaries.dtype() != float32) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_encode] boundaries must be float32, got "
        << boundaries.dtype();
    throw std::invalid_argument(msg.str());
  }
  if (boundaries.ndim() != 1) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_encode] boundaries must be rank-1, got shape "
        << boundaries.shape();
    throw std::invalid_argument(msg.str());
  }
  int n_inner = boundaries.shape(0);
  if (n_inner < 1 || n_inner > 255) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_encode] n_inner (= n_levels - 1) must be in "
           "[1, 255], got "
        << n_inner;
    throw std::invalid_argument(msg.str());
  }

  auto s = to_stream(s_);
  return array(
      x_norm.shape(),
      uint8,
      std::make_shared<TurboquantEncode>(s),
      {x_norm, boundaries});
}

array qjl_pack_signs(const array& values, int m, StreamOrDevice s_) {
  if (values.ndim() < 1) {
    throw std::invalid_argument(
        "[lumen::qjl_pack_signs] input must be rank ≥ 1");
  }
  if (m < 1 || m > 4096) {
    std::ostringstream msg;
    msg << "[lumen::qjl_pack_signs] m must be in [1, 4096], got " << m;
    throw std::invalid_argument(msg.str());
  }
  if (values.shape(-1) != m) {
    std::ostringstream msg;
    msg << "[lumen::qjl_pack_signs] last axis must equal m=" << m
        << ", got " << values.shape(-1);
    throw std::invalid_argument(msg.str());
  }

  // Cast input to float32 for sign extraction; downstream kernel reads
  // f32 only.
  array vf = (values.dtype() == float32)
      ? values
      : astype(values, float32, s_);

  auto s = to_stream(s_);
  Shape out_shape = values.shape();
  out_shape.back() = (m + 31) / 32;
  return array(
      std::move(out_shape),
      uint32,
      std::make_shared<QjlPackSigns>(s, m),
      {vf});
}

array qjl_unpack_signs(const array& packed, int m, StreamOrDevice s_) {
  if (packed.ndim() < 1) {
    throw std::invalid_argument(
        "[lumen::qjl_unpack_signs] input must be rank ≥ 1");
  }
  if (packed.dtype() != uint32) {
    std::ostringstream msg;
    msg << "[lumen::qjl_unpack_signs] packed must be uint32, got "
        << packed.dtype();
    throw std::invalid_argument(msg.str());
  }
  if (m < 1 || m > 4096) {
    std::ostringstream msg;
    msg << "[lumen::qjl_unpack_signs] m must be in [1, 4096], got " << m;
    throw std::invalid_argument(msg.str());
  }
  int expected_words = (m + 31) / 32;
  if (packed.shape(-1) != expected_words) {
    std::ostringstream msg;
    msg << "[lumen::qjl_unpack_signs] last axis must equal ceil(m/32)="
        << expected_words << ", got " << packed.shape(-1);
    throw std::invalid_argument(msg.str());
  }

  auto s = to_stream(s_);
  Shape out_shape = packed.shape();
  out_shape.back() = m;
  return array(
      std::move(out_shape),
      bfloat16,
      std::make_shared<QjlUnpackSigns>(s, m),
      {packed});
}

std::vector<array> turboquant_encode_fused(
    const array& x_rot,
    const array& boundaries,
    StreamOrDevice s_) {
  if (x_rot.dtype() != bfloat16) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_encode_fused] x_rot must be bfloat16, got "
        << x_rot.dtype();
    throw std::invalid_argument(msg.str());
  }
  if (boundaries.dtype() != float32) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_encode_fused] boundaries must be float32, got "
        << boundaries.dtype();
    throw std::invalid_argument(msg.str());
  }
  if (boundaries.ndim() != 1) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_encode_fused] boundaries must be rank-1, got "
           "shape "
        << boundaries.shape();
    throw std::invalid_argument(msg.str());
  }
  int n_inner = boundaries.shape(0);
  if (n_inner < 1 || n_inner > 255) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_encode_fused] n_inner (= n_levels - 1) must be "
           "in [1, 255], got "
        << n_inner;
    throw std::invalid_argument(msg.str());
  }
  if (x_rot.ndim() < 1) {
    throw std::invalid_argument(
        "[lumen::turboquant_encode_fused] x_rot must be rank ≥ 1");
  }
  int D = x_rot.shape(-1);
  if (D <= 0 || D > 1024 || (D % 32) != 0) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_encode_fused] last-axis D must be a positive "
           "multiple of 32 and ≤ 1024, got "
        << D;
    throw std::invalid_argument(msg.str());
  }

  Shape sigma_shape = x_rot.shape();
  sigma_shape.back() = 1;

  auto s = to_stream(s_);
  return array::make_arrays(
      {x_rot.shape(), std::move(sigma_shape)},
      {uint8, float32},
      std::make_shared<TurboquantEncodeFused>(s),
      {x_rot, boundaries});
}

array turboquant_qk_inline(
    const array& q,
    const array& k_codes,
    const array& k_sigma,
    const array& centroids,
    StreamOrDevice s_) {
  if (q.dtype() != bfloat16) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_qk_inline] q must be bfloat16, got "
        << q.dtype();
    throw std::invalid_argument(msg.str());
  }
  if (k_codes.dtype() != uint8) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_qk_inline] k_codes must be uint8, got "
        << k_codes.dtype();
    throw std::invalid_argument(msg.str());
  }
  if (k_sigma.dtype() != float32) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_qk_inline] k_sigma must be float32, got "
        << k_sigma.dtype();
    throw std::invalid_argument(msg.str());
  }
  if (centroids.dtype() != float32) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_qk_inline] centroids must be float32, got "
        << centroids.dtype();
    throw std::invalid_argument(msg.str());
  }
  if (q.ndim() != 4) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_qk_inline] q must be rank-4, got shape "
        << q.shape();
    throw std::invalid_argument(msg.str());
  }
  if (k_codes.ndim() != 4) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_qk_inline] k_codes must be rank-4, got shape "
        << k_codes.shape();
    throw std::invalid_argument(msg.str());
  }
  if (k_sigma.ndim() != 3) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_qk_inline] k_sigma must be rank-3 [B,H_kv,N], "
           "got shape "
        << k_sigma.shape();
    throw std::invalid_argument(msg.str());
  }
  if (centroids.ndim() != 1) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_qk_inline] centroids must be rank-1, got shape "
        << centroids.shape();
    throw std::invalid_argument(msg.str());
  }
  int B = q.shape(0);
  int H = q.shape(1);
  int T = q.shape(2);
  int D = q.shape(3);
  int H_kv = k_codes.shape(1);
  int N = k_codes.shape(2);
  if (k_codes.shape(0) != B || k_codes.shape(3) != D) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_qk_inline] q/k_codes shape mismatch: q="
        << q.shape() << " k_codes=" << k_codes.shape();
    throw std::invalid_argument(msg.str());
  }
  if (k_sigma.shape(0) != B || k_sigma.shape(1) != H_kv ||
      k_sigma.shape(2) != N) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_qk_inline] k_sigma shape "
        << k_sigma.shape() << " must be [B=" << B << ",H_kv=" << H_kv
        << ",N=" << N << "]";
    throw std::invalid_argument(msg.str());
  }
  if (H_kv == 0 || H % H_kv != 0) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_qk_inline] H=" << H
        << " must be a non-zero multiple of H_kv=" << H_kv;
    throw std::invalid_argument(msg.str());
  }
  if (D != 256 && D != 512) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_qk_inline] D must be 256 (sliding-attn head_dim) "
        << "or 512 (full-attn global_head_dim), got " << D;
    throw std::invalid_argument(msg.str());
  }
  int n_levels = centroids.shape(0);
  if (n_levels < 2 || n_levels > 16) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_qk_inline] n_levels must be in [2, 16], got "
        << n_levels;
    throw std::invalid_argument(msg.str());
  }

  auto s = to_stream(s_);
  Shape out_shape = {B, H, T, N};
  // K_codes / K_sigma typically come from a sliding-ring cache view whose
  // strides match the full ring (not the sliced extent). The kernel uses
  // linear indexing assuming packed shape, so materialize a contiguous copy
  // before dispatch. Q and centroids are typically already contiguous; skip
  // the contiguous() primitive when the input is already row-contiguous —
  // mlx's `contiguous()` always emits a Contiguous primitive (verified
  // 2026-05-26 via primhist: best-TQ Contiguous=240/step ≈ 4 calls × 2
  // kernels × 30 layers, all from this and sv_inline; eliminating no-ops
  // drops the count for the truly-contiguous inputs).
  auto ensure_rc = [&](const array& a) -> array {
    return a.flags().row_contiguous
        ? a
        : contiguous(a, /*allow_col_major=*/false, s_);
  };
  array q_c         = ensure_rc(q);
  array k_codes_c   = ensure_rc(k_codes);
  array k_sigma_c   = ensure_rc(k_sigma);
  array centroids_c = ensure_rc(centroids);
  return array(
      std::move(out_shape),
      bfloat16,
      std::make_shared<TurboquantQkInline>(s),
      {q_c, k_codes_c, k_sigma_c, centroids_c});
}

array turboquant_sv_inline(
    const array& s,
    const array& v_codes,
    const array& v_sigma,
    const array& centroids,
    StreamOrDevice s_) {
  if (s.dtype() != bfloat16) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_sv_inline] s must be bfloat16, got "
        << s.dtype();
    throw std::invalid_argument(msg.str());
  }
  if (v_codes.dtype() != uint8) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_sv_inline] v_codes must be uint8, got "
        << v_codes.dtype();
    throw std::invalid_argument(msg.str());
  }
  if (v_sigma.dtype() != float32) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_sv_inline] v_sigma must be float32, got "
        << v_sigma.dtype();
    throw std::invalid_argument(msg.str());
  }
  if (centroids.dtype() != float32) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_sv_inline] centroids must be float32, got "
        << centroids.dtype();
    throw std::invalid_argument(msg.str());
  }
  if (s.ndim() != 4) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_sv_inline] s must be rank-4, got shape "
        << s.shape();
    throw std::invalid_argument(msg.str());
  }
  if (v_codes.ndim() != 4) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_sv_inline] v_codes must be rank-4, got shape "
        << v_codes.shape();
    throw std::invalid_argument(msg.str());
  }
  if (v_sigma.ndim() != 3) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_sv_inline] v_sigma must be rank-3 [B,H_kv,N], "
           "got shape "
        << v_sigma.shape();
    throw std::invalid_argument(msg.str());
  }
  if (centroids.ndim() != 1) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_sv_inline] centroids must be rank-1, got shape "
        << centroids.shape();
    throw std::invalid_argument(msg.str());
  }
  int B = s.shape(0);
  int H = s.shape(1);
  int T = s.shape(2);
  int N = s.shape(3);
  int H_kv = v_codes.shape(1);
  int D = v_codes.shape(3);
  if (v_codes.shape(0) != B || v_codes.shape(2) != N) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_sv_inline] s/v_codes shape mismatch: s="
        << s.shape() << " v_codes=" << v_codes.shape();
    throw std::invalid_argument(msg.str());
  }
  if (v_sigma.shape(0) != B || v_sigma.shape(1) != H_kv ||
      v_sigma.shape(2) != N) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_sv_inline] v_sigma shape "
        << v_sigma.shape() << " must be [B=" << B << ",H_kv=" << H_kv
        << ",N=" << N << "]";
    throw std::invalid_argument(msg.str());
  }
  if (H_kv == 0 || H % H_kv != 0) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_sv_inline] H=" << H
        << " must be a non-zero multiple of H_kv=" << H_kv;
    throw std::invalid_argument(msg.str());
  }
  // Kernel + factory are D-agnostic — grid scales as ceil(D/64) d-tiles.
  // Empirically used: D=256 (sliding-attn) and D=512 (full-attn). Require
  // D to be a multiple of 64 (D_TILE_PER_TG) so tile alignment holds.
  if (D == 0 || D % 64 != 0) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_sv_inline] D must be a non-zero multiple of 64, "
        << "got " << D;
    throw std::invalid_argument(msg.str());
  }
  int n_levels = centroids.shape(0);
  if (n_levels < 2 || n_levels > 16) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_sv_inline] n_levels must be in [2, 16], got "
        << n_levels;
    throw std::invalid_argument(msg.str());
  }

  // V_codes / V_sigma typically come from a sliding-ring cache view whose
  // strides match the full ring (not the sliced extent). The kernel uses
  // linear indexing assuming packed shape, so materialize a contiguous copy
  // before dispatch — same bug fix as the Q@K_codes kernel. Skip the
  // primitive when input is already row-contiguous (see qk_inline above
  // for the primhist rationale).
  auto ensure_rc_sv = [&](const array& a) -> array {
    return a.flags().row_contiguous
        ? a
        : contiguous(a, /*allow_col_major=*/false, s_);
  };
  array s_c         = ensure_rc_sv(s);
  array v_codes_c   = ensure_rc_sv(v_codes);
  array v_sigma_c   = ensure_rc_sv(v_sigma);
  array centroids_c = ensure_rc_sv(centroids);

  auto stream_ = to_stream(s_);
  Shape out_shape = {B, H, T, D};
  return array(
      std::move(out_shape),
      bfloat16,
      std::make_shared<TurboquantSvInline>(stream_),
      {s_c, v_codes_c, v_sigma_c, centroids_c});
}

array turboquant_fused_attn(
    const array& q,
    const array& k_codes,
    const array& k_sigma,
    const array& v_codes,
    const array& v_sigma,
    const array& centroids,
    float scale,
    StreamOrDevice s_) {
  // ── dtype checks ──
  if (q.dtype() != bfloat16) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_fused_attn] q must be bfloat16, got "
        << q.dtype();
    throw std::invalid_argument(msg.str());
  }
  if (k_codes.dtype() != uint8 || v_codes.dtype() != uint8) {
    throw std::invalid_argument(
        "[lumen::turboquant_fused_attn] k_codes and v_codes must be uint8");
  }
  if (k_sigma.dtype() != float32 || v_sigma.dtype() != float32) {
    throw std::invalid_argument(
        "[lumen::turboquant_fused_attn] k_sigma and v_sigma must be float32");
  }
  if (centroids.dtype() != float32) {
    throw std::invalid_argument(
        "[lumen::turboquant_fused_attn] centroids must be float32");
  }

  // ── rank + shape checks ──
  if (q.ndim() != 4) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_fused_attn] q must be rank-4 [B,H,T,D], got "
        << q.shape();
    throw std::invalid_argument(msg.str());
  }
  if (k_codes.ndim() != 4 || v_codes.ndim() != 4) {
    throw std::invalid_argument(
        "[lumen::turboquant_fused_attn] k_codes/v_codes must be rank-4 "
        "[B,H_kv,N,D]");
  }
  if (k_sigma.ndim() != 3 || v_sigma.ndim() != 3) {
    throw std::invalid_argument(
        "[lumen::turboquant_fused_attn] k_sigma/v_sigma must be rank-3 "
        "[B,H_kv,N]");
  }
  if (centroids.ndim() != 1) {
    throw std::invalid_argument(
        "[lumen::turboquant_fused_attn] centroids must be rank-1");
  }

  int B    = q.shape(0);
  int H    = q.shape(1);
  int T    = q.shape(2);
  int D    = q.shape(3);
  int H_kv = k_codes.shape(1);
  int N    = k_codes.shape(2);

  if (T != 1) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_fused_attn] T must be 1 (decode shape), got "
        << T;
    throw std::invalid_argument(msg.str());
  }
  if (D != 256 && D != 512) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_fused_attn] D must be 256 (sliding) or 512 "
           "(full-attn), got " << D;
    throw std::invalid_argument(msg.str());
  }
  if (k_codes.shape(0) != B || k_codes.shape(3) != D) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_fused_attn] k_codes shape " << k_codes.shape()
        << " must be [B=" << B << ",H_kv,N," << D << "]";
    throw std::invalid_argument(msg.str());
  }
  if (v_codes.shape(0) != B || v_codes.shape(1) != H_kv ||
      v_codes.shape(2) != N || v_codes.shape(3) != D) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_fused_attn] v_codes shape " << v_codes.shape()
        << " must match k_codes shape " << k_codes.shape();
    throw std::invalid_argument(msg.str());
  }
  if (k_sigma.shape(0) != B || k_sigma.shape(1) != H_kv ||
      k_sigma.shape(2) != N) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_fused_attn] k_sigma shape " << k_sigma.shape()
        << " must be [B=" << B << ",H_kv=" << H_kv << ",N=" << N << "]";
    throw std::invalid_argument(msg.str());
  }
  if (v_sigma.shape(0) != B || v_sigma.shape(1) != H_kv ||
      v_sigma.shape(2) != N) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_fused_attn] v_sigma shape " << v_sigma.shape()
        << " must be [B=" << B << ",H_kv=" << H_kv << ",N=" << N << "]";
    throw std::invalid_argument(msg.str());
  }
  if (H_kv == 0 || H % H_kv != 0) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_fused_attn] H=" << H
        << " must be a non-zero multiple of H_kv=" << H_kv;
    throw std::invalid_argument(msg.str());
  }
  int n_levels = centroids.shape(0);
  if (n_levels < 2 || n_levels > 16) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_fused_attn] n_levels must be in [2, 16], got "
        << n_levels;
    throw std::invalid_argument(msg.str());
  }

  // Unconditional contiguous() required for correctness (2026-05-26): with
  // flag-based skip (a.flags().row_contiguous ? a : contiguous(a, ...))
  // e2e produced garbage output even though row_contiguous reported true
  // for all 6 inputs. Diagnostic logging confirmed flags = (1,1,1,1,1,1)
  // matched isolated unit-test conditions, yet only the unconditional
  // path produces sane decode tokens. Hypothesis: the extra Contiguous
  // primitive (no-op when input is already RC) acts as an implicit
  // evaluation barrier that resolves an mlx graph ordering bug specific
  // to this fused kernel's input set.
  //
  // Cost: +180 Contiguous primitives/step at decode (6 inputs × 30 layers).
  // mlx evaluates these as no-ops when row_contiguous=true so GPU time is
  // ~0, but primitive overhead is ~1-2 µs each ≈ 180-360 µs/step.
  // Combined with the 2-dispatch saving per layer from fused (~720 µs/step),
  // the algebra suggests a small net win — but end-to-end measurement
  // (2026-05-26 cooldown-bracketed) shows fused_attn delivers −7.4% decode
  // vs the 3-dispatch (qk_inline + softmax_axis + sv_inline) baseline.
  // The custom online-softmax kernel is slower than mlx's tuned individual
  // kernels, consistent with the v3 sv_inline finding that mlx's tile-MMA
  // matmul beats hand-written Metal at these shapes.
  //
  // **fused_attn is shipped behind LUMEN_GEMMA4_TQ_FUSED_ATTN=1 (default
  // OFF)** for future revisit when a tile-MMA rewrite of the kernel is
  // possible. Until then it's not on the production path.
  array q_c         = contiguous(q,         /*allow_col_major=*/false, s_);
  array k_codes_c   = contiguous(k_codes,   /*allow_col_major=*/false, s_);
  array k_sigma_c   = contiguous(k_sigma,   /*allow_col_major=*/false, s_);
  array v_codes_c   = contiguous(v_codes,   /*allow_col_major=*/false, s_);
  array v_sigma_c   = contiguous(v_sigma,   /*allow_col_major=*/false, s_);
  array centroids_c = contiguous(centroids, /*allow_col_major=*/false, s_);

  auto stream_ = to_stream(s_);
  Shape out_shape = {B, H, T, D};
  return array(
      std::move(out_shape),
      bfloat16,
      std::make_shared<TurboquantFusedAttn>(stream_, scale),
      {q_c, k_codes_c, k_sigma_c, v_codes_c, v_sigma_c, centroids_c});
}

std::vector<array> turboquant_encode_fused_packed4(
    const array& x_rot,
    const array& boundaries,
    StreamOrDevice s_) {
  if (x_rot.dtype() != bfloat16) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_encode_fused_packed4] x_rot must be bfloat16, "
           "got " << x_rot.dtype();
    throw std::invalid_argument(msg.str());
  }
  if (boundaries.dtype() != float32 || boundaries.ndim() != 1) {
    throw std::invalid_argument(
        "[lumen::turboquant_encode_fused_packed4] boundaries must be rank-1 "
        "float32");
  }
  int n_inner = boundaries.shape(0);
  if (n_inner != 15) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_encode_fused_packed4] only 4-bit supported "
           "(n_inner must = 15, got " << n_inner << ")";
    throw std::invalid_argument(msg.str());
  }
  if (x_rot.ndim() < 1) {
    throw std::invalid_argument(
        "[lumen::turboquant_encode_fused_packed4] x_rot rank ≥ 1 required");
  }
  int D = x_rot.shape(-1);
  if (D <= 0 || D > 1024 || (D % 32) != 0 || (D % 8) != 0) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_encode_fused_packed4] D must be multiple of 32 "
           "(and 8), ≤ 1024, got " << D;
    throw std::invalid_argument(msg.str());
  }

  Shape codes_shape = x_rot.shape();
  codes_shape.back() = D / 8;
  Shape sigma_shape = x_rot.shape();
  sigma_shape.back() = 1;

  auto s = to_stream(s_);
  return array::make_arrays(
      {std::move(codes_shape), std::move(sigma_shape)},
      {uint32, float32},
      std::make_shared<TurboquantEncodeFusedPacked4>(s),
      {x_rot, boundaries});
}

array turboquant_qk_inline_packed4(
    const array& q,
    const array& k_codes_pkd,
    const array& k_sigma,
    const array& centroids,
    StreamOrDevice s_) {
  if (q.dtype() != bfloat16 || q.ndim() != 4) {
    throw std::invalid_argument(
        "[lumen::turboquant_qk_inline_packed4] q must be rank-4 bfloat16");
  }
  if (k_codes_pkd.dtype() != uint32 || k_codes_pkd.ndim() != 4) {
    throw std::invalid_argument(
        "[lumen::turboquant_qk_inline_packed4] k_codes_pkd must be rank-4 "
        "uint32");
  }
  if (k_sigma.dtype() != float32 || k_sigma.ndim() != 3) {
    throw std::invalid_argument(
        "[lumen::turboquant_qk_inline_packed4] k_sigma must be rank-3 "
        "float32 [B, H_kv, N]");
  }
  if (centroids.dtype() != float32 || centroids.ndim() != 1) {
    throw std::invalid_argument(
        "[lumen::turboquant_qk_inline_packed4] centroids must be rank-1 "
        "float32");
  }

  int B = q.shape(0);
  int H = q.shape(1);
  int T = q.shape(2);
  int D = q.shape(3);
  int H_kv = k_codes_pkd.shape(1);
  int N = k_codes_pkd.shape(2);
  int n_packed = k_codes_pkd.shape(3);

  if (D != 256) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_qk_inline_packed4] D must be 256, got " << D;
    throw std::invalid_argument(msg.str());
  }
  if (n_packed != D / 8) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_qk_inline_packed4] k_codes_pkd last axis must "
           "be D/8 = " << (D / 8) << ", got " << n_packed;
    throw std::invalid_argument(msg.str());
  }
  if (k_codes_pkd.shape(0) != B) {
    throw std::invalid_argument(
        "[lumen::turboquant_qk_inline_packed4] batch mismatch");
  }
  if (k_sigma.shape(0) != B || k_sigma.shape(1) != H_kv ||
      k_sigma.shape(2) != N) {
    throw std::invalid_argument(
        "[lumen::turboquant_qk_inline_packed4] k_sigma shape mismatch");
  }
  if (H_kv == 0 || H % H_kv != 0) {
    throw std::invalid_argument(
        "[lumen::turboquant_qk_inline_packed4] H must be multiple of H_kv");
  }
  int n_levels = centroids.shape(0);
  if (n_levels != 16) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_qk_inline_packed4] only 4-bit supported "
           "(n_levels=16), got " << n_levels;
    throw std::invalid_argument(msg.str());
  }

  auto ensure_rc_qkp = [&](const array& a) -> array {
    return a.flags().row_contiguous ? a : contiguous(a, false, s_);
  };
  array q_c           = ensure_rc_qkp(q);
  array k_codes_pkd_c = ensure_rc_qkp(k_codes_pkd);
  array k_sigma_c     = ensure_rc_qkp(k_sigma);
  array centroids_c   = ensure_rc_qkp(centroids);

  auto s = to_stream(s_);
  Shape out_shape = {B, H, T, N};
  return array(
      std::move(out_shape),
      bfloat16,
      std::make_shared<TurboquantQkInlinePacked4>(s),
      {q_c, k_codes_pkd_c, k_sigma_c, centroids_c});
}

std::vector<array> turboquant_rot_encode_fused(
    const array& x_bf16,
    const array& R_f32,
    const array& boundaries,
    StreamOrDevice s_) {
  if (x_bf16.dtype() != bfloat16) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_rot_encode_fused] x_bf16 must be bfloat16, got "
        << x_bf16.dtype();
    throw std::invalid_argument(msg.str());
  }
  if (R_f32.dtype() != float32) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_rot_encode_fused] R must be float32, got "
        << R_f32.dtype();
    throw std::invalid_argument(msg.str());
  }
  if (R_f32.ndim() != 2) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_rot_encode_fused] R must be rank-2, got shape "
        << R_f32.shape();
    throw std::invalid_argument(msg.str());
  }
  if (boundaries.dtype() != float32) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_rot_encode_fused] boundaries must be float32, "
           "got "
        << boundaries.dtype();
    throw std::invalid_argument(msg.str());
  }
  if (boundaries.ndim() != 1) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_rot_encode_fused] boundaries must be rank-1, "
           "got shape "
        << boundaries.shape();
    throw std::invalid_argument(msg.str());
  }
  int n_inner = boundaries.shape(0);
  if (n_inner < 1 || n_inner > 255) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_rot_encode_fused] n_inner (= n_levels - 1) "
           "must be in [1, 255], got "
        << n_inner;
    throw std::invalid_argument(msg.str());
  }
  if (x_bf16.ndim() < 1) {
    throw std::invalid_argument(
        "[lumen::turboquant_rot_encode_fused] x_bf16 must be rank ≥ 1");
  }
  int D = x_bf16.shape(-1);
  if (D <= 0 || D > 1024 || (D % 32) != 0) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_rot_encode_fused] last-axis D must be a "
           "positive multiple of 32 and ≤ 1024, got "
        << D;
    throw std::invalid_argument(msg.str());
  }
  if (R_f32.shape(0) != D || R_f32.shape(1) != D) {
    std::ostringstream msg;
    msg << "[lumen::turboquant_rot_encode_fused] R must be [D=" << D
        << ", D=" << D << "], got " << R_f32.shape();
    throw std::invalid_argument(msg.str());
  }

  Shape sigma_shape = x_bf16.shape();
  sigma_shape.back() = 1;

  auto s = to_stream(s_);
  return array::make_arrays(
      {x_bf16.shape(), std::move(sigma_shape)},
      {uint8, float32},
      std::make_shared<TurboquantRotEncodeFused>(s),
      {x_bf16, R_f32, boundaries});
}

} // namespace mlx::core::lumen
