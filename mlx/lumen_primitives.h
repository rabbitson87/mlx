// lumen-rs Phase 1.8 M4.8 — custom flash-attention primitive.
//
// Declares a Primitive subclass that runs our bf16 flash-attention kernel
// inside mlx's lazy graph (no per-call eval sync). Mirrors the structure of
// `mlx::core::fast::ScaledDotProductAttention` but lives in a separate
// namespace so lumen-specific code stays cleanly isolated.

#pragma once

#include <optional>

#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core::lumen {

class FlashAttnBf16 : public Primitive {
 public:
  FlashAttnBf16(Stream stream, float scale, bool has_mask)
      : Primitive(stream), scale_(scale), has_mask_(has_mask) {}

  void eval_cpu(
      const std::vector<array>& /*inputs*/,
      std::vector<array>& /*outputs*/) override {
    throw std::runtime_error(
        "lumen::FlashAttnBf16: CPU evaluation not implemented");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override;

  const char* name() const override {
    return "LumenFlashAttnBf16";
  }

  bool is_equivalent(const Primitive& other) const override {
    const auto& o = static_cast<const FlashAttnBf16&>(other);
    return scale_ == o.scale_ && has_mask_ == o.has_mask_;
  }

  std::vector<Shape> output_shapes(
      const std::vector<array>& inputs) override {
    const auto& q = inputs[0];
    const auto& v = inputs[2];
    return {{q.shape(0), q.shape(1), q.shape(2), v.shape(-1)}};
  }

  float scale() const {
    return scale_;
  }
  bool has_mask() const {
    return has_mask_;
  }

 private:
  float scale_;
  bool has_mask_;
};

/// Build a lazy array node for the custom flash-attention kernel.
/// Inputs:
///   q     : [B, H,    Sq,  D]  bfloat16, D must be 256
///   k     : [B, H_kv, Skv, D]  bfloat16, H % H_kv == 0
///   v     : same shape as k    bfloat16
///   mask  : [Sq, Skv]          bfloat16, additive (optional)
/// Output: [B, H, Sq, D] bfloat16.
array flash_attn_bf16(
    const array& q,
    const array& k,
    const array& v,
    float scale,
    std::optional<array> mask = std::nullopt,
    StreamOrDevice s = {});

// ─────────────────────────────────────────────────────────────────────────
// FlashAttnPrefillBf16 — bf16 Flash-Attn-2 prefill (Sq>1).
//
// Q-tiled FA-2 with in-register causal + optional sliding-window mask. No
// additive mask Array materialized. Supports head_dim=256 (sliding) and
// head_dim=512 (full causal-only). Separate primitive from FlashAttnBf16
// so the decode code path stays untouched.
// ─────────────────────────────────────────────────────────────────────────

class FlashAttnPrefillBf16 : public Primitive {
 public:
  FlashAttnPrefillBf16(
      Stream stream,
      float scale,
      uint32_t window_size,
      uint32_t kv_offset)
      : Primitive(stream),
        scale_(scale),
        window_size_(window_size),
        kv_offset_(kv_offset) {}

  void eval_cpu(
      const std::vector<array>& /*inputs*/,
      std::vector<array>& /*outputs*/) override {
    throw std::runtime_error(
        "lumen::FlashAttnPrefillBf16: CPU evaluation not implemented");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override;

  const char* name() const override {
    return "LumenFlashAttnPrefillBf16";
  }

  bool is_equivalent(const Primitive& other) const override {
    const auto& o = static_cast<const FlashAttnPrefillBf16&>(other);
    return scale_ == o.scale_ && window_size_ == o.window_size_ &&
           kv_offset_ == o.kv_offset_;
  }

  std::vector<Shape> output_shapes(
      const std::vector<array>& inputs) override {
    const auto& q = inputs[0];
    const auto& v = inputs[2];
    return {{q.shape(0), q.shape(1), q.shape(2), v.shape(-1)}};
  }

  float scale() const { return scale_; }
  uint32_t window_size() const { return window_size_; }
  uint32_t kv_offset() const { return kv_offset_; }

 private:
  float scale_;
  uint32_t window_size_;  // 0 = no window (full causal)
  uint32_t kv_offset_;    // global offset of Q[0] (for cache rotation)
};

/// Build a lazy array node for the FA-2 prefill kernel.
///   q     : [B, H,    Sq,  D]  bfloat16, D ∈ {256, 512}, Sq > 1
///   k     : [B, H_kv, Skv, D]  bfloat16, H % H_kv == 0
///   v     : same shape as k    bfloat16
///   scale       : QK^T multiplier
///   window_size : 0 for full causal, >0 for sliding window of given size
///   kv_offset   : absolute index of Q[0] in the K/V stream
array flash_attn_prefill_bf16(
    const array& q,
    const array& k,
    const array& v,
    float scale,
    uint32_t window_size,
    uint32_t kv_offset,
    StreamOrDevice s = {});

// ─────────────────────────────────────────────────────────────────────────
// TurboquantEncode — Lloyd-Max nearest-centroid Stage-1 encode.
//
// Replaces the mlx-ops argmin-broadcast path for KV-cache quantization.
// Inputs:
//   x_norm     : [..., D]            bfloat16  (pre-normalized, σ-divided)
//   boundaries : [n_inner = 2^bits-1] float32   (inner Lloyd-Max boundaries)
// Output:
//   codes      : same shape as x_norm  uint8
// ─────────────────────────────────────────────────────────────────────────

class TurboquantEncode : public Primitive {
 public:
  explicit TurboquantEncode(Stream stream) : Primitive(stream) {}

  void eval_cpu(
      const std::vector<array>& /*inputs*/,
      std::vector<array>& /*outputs*/) override {
    throw std::runtime_error(
        "lumen::TurboquantEncode: CPU evaluation not implemented");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override;

  const char* name() const override {
    return "LumenTurboquantEncode";
  }

  bool is_equivalent(const Primitive& /*other*/) const override {
    return true; // stateless aside from stream
  }

  std::vector<Shape> output_shapes(
      const std::vector<array>& inputs) override {
    return {inputs[0].shape()};
  }
};

/// Build a lazy array node for the TurboQuant Lloyd-Max encode kernel.
///   x_norm     : [..., D] bfloat16, pre-normalized so x ≈ N(0, 1)
///   boundaries : [n_levels - 1] float32 (inner boundaries; ignore the
///                 -INF / +INF endpoints from `LloydMaxCodebook.boundaries`)
/// Output: same shape as x_norm, uint8 codes in [0, n_levels - 1].
array turboquant_encode(
    const array& x_norm,
    const array& boundaries,
    StreamOrDevice s = {});

// ─────────────────────────────────────────────────────────────────────────
// TurboquantEncodeFused — σ + normalize + Lloyd-Max encode in one kernel.
//
// Replaces the multi-dispatch chain (cast → square → sum → divide → sqrt →
// divide → cast → encode = ~8 dispatches) with a single threadgroup-per-row
// kernel that computes σ via simdgroup reduce, normalizes in-register, and
// emits both Lloyd-Max codes and the per-row σ in one pass.
//
// Inputs:
//   x_rot      : [..., D]            bfloat16  (rotated K/V; NOT normalized)
//   boundaries : [n_inner = 2^bits-1] float32   (inner Lloyd-Max boundaries)
// Outputs:
//   codes  : same shape as x_rot     uint8
//   sigma  : x_rot.shape with last=1 float32
//
// Constraints:
//   - D must be a positive multiple of 32 and ≤ 1024 (simdgroup reduction
//     plus single-TG-per-row layout)
//   - n_inner ∈ [1, 255]
// ─────────────────────────────────────────────────────────────────────────

class TurboquantEncodeFused : public Primitive {
 public:
  explicit TurboquantEncodeFused(Stream stream) : Primitive(stream) {}

  void eval_cpu(
      const std::vector<array>& /*inputs*/,
      std::vector<array>& /*outputs*/) override {
    throw std::runtime_error(
        "lumen::TurboquantEncodeFused: CPU evaluation not implemented");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override;

  const char* name() const override {
    return "LumenTurboquantEncodeFused";
  }

  bool is_equivalent(const Primitive& /*other*/) const override {
    return true; // stateless aside from stream
  }

  std::vector<Shape> output_shapes(
      const std::vector<array>& inputs) override {
    Shape sigma_shape = inputs[0].shape();
    if (!sigma_shape.empty()) {
      sigma_shape.back() = 1;
    }
    return {inputs[0].shape(), sigma_shape};
  }
};

/// Build lazy array nodes for the fused TurboQuant encode kernel. Returns
/// `{codes, sigma}` where codes is uint8 same-shape, sigma is float32 with
/// last axis = 1.
///   x_rot      : [..., D] bfloat16 (rotated, σ not yet applied)
///   boundaries : [n_levels - 1] float32 (inner boundaries)
std::vector<array> turboquant_encode_fused(
    const array& x_rot,
    const array& boundaries,
    StreamOrDevice s = {});

// ─────────────────────────────────────────────────────────────────────────
// TurboquantRotEncodeFused — (x @ R) + σ + normalize + Lloyd-Max encode in
// one kernel.
//
// Used when the rotated tensor is only consumed by the encode (e.g. V side
// of TurboQuant). Skips the bf16 round-trip on the rotated intermediate
// plus the separate matmul and cast dispatches.
//
// Inputs:
//   x_bf16     : [..., D]             bfloat16  (un-rotated input)
//   R_f32      : [D, D]               float32   (Haar orthogonal rotation)
//   boundaries : [n_inner = 2^bits-1] float32   (inner Lloyd-Max boundaries)
// Outputs:
//   codes  : same shape as x_bf16     uint8
//   sigma  : x_bf16.shape with last=1 float32
//
// Constraints (validated in the factory):
//   - D must be a positive multiple of 32 and ≤ 1024
//   - R_f32.shape == [D, D]
//   - n_inner ∈ [1, 255]
// ─────────────────────────────────────────────────────────────────────────

class TurboquantRotEncodeFused : public Primitive {
 public:
  explicit TurboquantRotEncodeFused(Stream stream) : Primitive(stream) {}

  void eval_cpu(
      const std::vector<array>& /*inputs*/,
      std::vector<array>& /*outputs*/) override {
    throw std::runtime_error(
        "lumen::TurboquantRotEncodeFused: CPU evaluation not implemented");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override;

  const char* name() const override {
    return "LumenTurboquantRotEncodeFused";
  }

  bool is_equivalent(const Primitive& /*other*/) const override {
    return true; // stateless aside from stream
  }

  std::vector<Shape> output_shapes(
      const std::vector<array>& inputs) override {
    Shape sigma_shape = inputs[0].shape();
    if (!sigma_shape.empty()) {
      sigma_shape.back() = 1;
    }
    return {inputs[0].shape(), sigma_shape};
  }
};

/// Build lazy array nodes for the fused TurboQuant rotate+encode kernel.
/// Returns `{codes, sigma}` — codes is uint8 same-shape as x_bf16, sigma is
/// float32 with last axis = 1.
///   x_bf16     : [..., D] bfloat16 (un-rotated)
///   R_f32      : [D, D]   float32  (Haar orthogonal)
///   boundaries : [n_levels - 1] float32
std::vector<array> turboquant_rot_encode_fused(
    const array& x_bf16,
    const array& R_f32,
    const array& boundaries,
    StreamOrDevice s = {});

// ─────────────────────────────────────────────────────────────────────────
// TurboquantQkInline — Q @ K_dq^T attention scores without materializing K_dq.
//
// Inline Lloyd-Max dequant: each K element is one uint8 code that indexes
// a shared centroids LUT, scaled by per-K-vector σ. Eliminates the K_dq
// DRAM round-trip in the TQ-Stage-1 decode path (4-stage roadmap Stage 2).
//
// Inputs:
//   q         : [B, H,    T, D]   bfloat16  (T = 1 typical for decode)
//   k_codes   : [B, H_kv, N, D]   uint8     (Lloyd-Max codes; bits ≤ 4)
//   k_sigma   : [B, H_kv, N]      float32   (per-K-vector σ; squeeze the
//                                            trailing-1 from the encode
//                                            output before calling)
//   centroids : [n_levels]        float32   (Lloyd-Max LUT; n_levels ≤ 16)
// Output:
//   scores    : [B, H,    T, N]   bfloat16
//
// Constraints:
//   - D ∈ {256, 512}  (kernel specialized via VPT function-constant:
//     VPT=8 for D=256 — sliding-attention head_dim;
//     VPT=16 for D=512 — Gemma 4 full-attention global_head_dim)
//   - n_levels ≤ 16 (bits ≤ 4)
//   - q.shape(0) == k_codes.shape(0)
//   - q.shape(3) == k_codes.shape(3)
//   - q.shape(1) % k_codes.shape(1) == 0   (GQA group ratio integer)
// ─────────────────────────────────────────────────────────────────────────

class TurboquantQkInline : public Primitive {
 public:
  explicit TurboquantQkInline(Stream stream) : Primitive(stream) {}

  void eval_cpu(
      const std::vector<array>& /*inputs*/,
      std::vector<array>& /*outputs*/) override {
    throw std::runtime_error(
        "lumen::TurboquantQkInline: CPU evaluation not implemented");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override;

  const char* name() const override {
    return "LumenTurboquantQkInline";
  }

  bool is_equivalent(const Primitive& /*other*/) const override {
    return true;  // stateless aside from stream
  }

  std::vector<Shape> output_shapes(
      const std::vector<array>& inputs) override {
    const auto& q = inputs[0];
    const auto& k = inputs[1];
    return {{q.shape(0), q.shape(1), q.shape(2), k.shape(2)}};
  }
};

/// Build a lazy array node for the TurboQuant Q @ K_codes inline matmul.
/// See class doc above for shape/dtype constraints.
array turboquant_qk_inline(
    const array& q,
    const array& k_codes,
    const array& k_sigma,
    const array& centroids,
    StreamOrDevice s = {});

// ─────────────────────────────────────────────────────────────────────────
// TurboquantSvInline — Attention output O = S · V_dq without materializing
// V_dq. Symmetric V-side counterpart to TurboquantQkInline (4-stage roadmap
// Stage 3). Inline Lloyd-Max dequant: each V element is one uint8 code
// indexing a shared centroids LUT, scaled by per-V-vector σ.
//
// Inputs:
//   s         : [B, H,    T, N]   bfloat16  (softmax-normalized scores)
//   v_codes   : [B, H_kv, N, D]   uint8     (Lloyd-Max codes; bits ≤ 4)
//   v_sigma   : [B, H_kv, N]      float32   (per-V-vector σ; squeeze the
//                                            trailing-1 from the encode
//                                            output before calling)
//   centroids : [n_levels]        float32   (Lloyd-Max LUT; n_levels ≤ 16)
// Output:
//   o         : [B, H,    T, D]   bfloat16
//
// Constraints:
//   - D must be a multiple of 64 (D_TILE_PER_TG); validated at factory.
//     Empirically used: D=256 (sliding-attn) and D=512 (full-attn). The
//     kernel + factory are D-agnostic — grid scales as ceil(D/64) d-tiles
//     so larger D just dispatches more threadgroups.
//   - n_levels ≤ 16 (bits ≤ 4)
//   - s.shape(0) == v_codes.shape(0)
//   - s.shape(3) == v_codes.shape(2) == N
//   - s.shape(1) % v_codes.shape(1) == 0   (GQA group ratio integer)
// ─────────────────────────────────────────────────────────────────────────

class TurboquantSvInline : public Primitive {
 public:
  explicit TurboquantSvInline(Stream stream) : Primitive(stream) {}

  void eval_cpu(
      const std::vector<array>& /*inputs*/,
      std::vector<array>& /*outputs*/) override {
    throw std::runtime_error(
        "lumen::TurboquantSvInline: CPU evaluation not implemented");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override;

  const char* name() const override {
    return "LumenTurboquantSvInline";
  }

  bool is_equivalent(const Primitive& /*other*/) const override {
    return true;  // stateless aside from stream
  }

  std::vector<Shape> output_shapes(
      const std::vector<array>& inputs) override {
    const auto& s = inputs[0];   // [B, H, T, N]
    const auto& v = inputs[1];   // [B, H_kv, N, D]
    return {{s.shape(0), s.shape(1), s.shape(2), v.shape(3)}};
  }
};

/// Build a lazy array node for the TurboQuant softmax_scores @ V_codes
/// inline matmul. See class doc above for shape/dtype constraints.
array turboquant_sv_inline(
    const array& s,
    const array& v_codes,
    const array& v_sigma,
    const array& centroids,
    StreamOrDevice stream = {});

// ─────────────────────────────────────────────────────────────────────────
// TurboquantFusedAttn — full attention (QK^T, softmax, AV) fused into one
// Metal dispatch with inline Lloyd-Max K/V dequant. Replaces the
// (turboquant_qk_inline + softmax + turboquant_sv_inline) 3-dispatch chain
// at decode (T=1).
//
// Inputs:
//   q         : [B, H,    T=1, D]   bfloat16   (Q in head_dim space)
//   k_codes   : [B, H_kv, N,   D]   uint8
//   k_sigma   : [B, H_kv, N]        float32
//   v_codes   : [B, H_kv, N,   D]   uint8
//   v_sigma   : [B, H_kv, N]        float32
//   centroids : [n_levels]          float32   (Lloyd-Max LUT; n_levels ≤ 16)
//   scale     : (host-side scalar)  float32   (attention scale = 1/sqrt(D)
//                                              * any external softcap factor)
// Output:
//   o         : [B, H,    T=1, D]   bfloat16
//
// Constraints:
//   - T == 1 (decode shape; prefill handled by mlx fast::sdpa)
//   - D ∈ {256, 512}  (source-duplicated kernels for sliding/full-attn)
//   - n_levels ≤ 16  (bits ≤ 4)
//   - H % H_kv == 0  (GQA)
//
// Threading: BN=32 simdgroups × BD=32 threads/SG; online softmax updates
// register-resident max + sum per N stride; final cross-SG aggregate
// produces normalized output. See lumen_turboquant.cpp for the kernel.
// ─────────────────────────────────────────────────────────────────────────

class TurboquantFusedAttn : public Primitive {
 public:
  explicit TurboquantFusedAttn(Stream stream, float scale)
      : Primitive(stream), scale_(scale) {}

  void eval_cpu(
      const std::vector<array>& /*inputs*/,
      std::vector<array>& /*outputs*/) override {
    throw std::runtime_error(
        "lumen::TurboquantFusedAttn: CPU evaluation not implemented");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override;

  const char* name() const override {
    return "LumenTurboquantFusedAttn";
  }

  bool is_equivalent(const Primitive& other) const override {
    const auto& o = static_cast<const TurboquantFusedAttn&>(other);
    return scale_ == o.scale_;
  }

  std::vector<Shape> output_shapes(
      const std::vector<array>& inputs) override {
    const auto& q = inputs[0];           // [B, H, T, D]
    return {{q.shape(0), q.shape(1), q.shape(2), q.shape(3)}};
  }

  float scale() const { return scale_; }

 private:
  float scale_;
};

/// Build a lazy array node for the fused TurboQuant attention. See class
/// doc above for shape/dtype constraints. `scale` is the standard attn
/// scale (= 1/sqrt(head_dim)) optionally multiplied by external softcap.
array turboquant_fused_attn(
    const array& q,
    const array& k_codes,
    const array& k_sigma,
    const array& v_codes,
    const array& v_sigma,
    const array& centroids,
    float scale,
    StreamOrDevice stream = {});

// ─────────────────────────────────────────────────────────────────────────
// TurboquantEncodeFusedPacked4 — same as TurboquantEncodeFused but emits
// 4-bit codes packed into uint32 (8 codes per word). Halves K/V cache
// storage and the inline-kernel DRAM read bandwidth.
//
// Inputs:
//   x_rot      : [..., D]    bfloat16 (rotated; not normalized)
//   boundaries : [n_inner]   float32 (must imply n_levels=16 for 4-bit)
// Outputs:
//   codes_pkd  : [..., D/8]  uint32 (packed)
//   sigma      : [..., 1]    float32
//
// Constraints: D ≤ 1024, D % 32 == 0, D % 8 == 0, n_inner ∈ [1, 255].
// Currently 4-bit only (n_levels=16). bits=2/3 packed variants are future
// work.
// ─────────────────────────────────────────────────────────────────────────

class TurboquantEncodeFusedPacked4 : public Primitive {
 public:
  explicit TurboquantEncodeFusedPacked4(Stream stream) : Primitive(stream) {}

  void eval_cpu(
      const std::vector<array>& /*inputs*/,
      std::vector<array>& /*outputs*/) override {
    throw std::runtime_error(
        "lumen::TurboquantEncodeFusedPacked4: CPU eval not implemented");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override;

  const char* name() const override {
    return "LumenTurboquantEncodeFusedPacked4";
  }

  bool is_equivalent(const Primitive& /*other*/) const override {
    return true;
  }

  std::vector<Shape> output_shapes(
      const std::vector<array>& inputs) override {
    Shape codes_shape = inputs[0].shape();
    Shape sigma_shape = inputs[0].shape();
    if (!codes_shape.empty()) {
      codes_shape.back() = inputs[0].shape(-1) / 8;
      sigma_shape.back() = 1;
    }
    return {codes_shape, sigma_shape};
  }
};

std::vector<array> turboquant_encode_fused_packed4(
    const array& x_rot,
    const array& boundaries,
    StreamOrDevice stream = {});

// ─────────────────────────────────────────────────────────────────────────
// TurboquantQkInlinePacked4 — Q @ K_codes_packed inline matmul. Same shape
// contract as TurboquantQkInline but K_codes is packed uint32 (4 bits per
// code, 8 codes per word).
//
// Inputs:
//   q         : [B, H,    T, D=256]      bfloat16
//   k_codes_pkd : [B, H_kv, N, D/8 = 32] uint32 (PACKED 4-bit)
//   k_sigma   : [B, H_kv, N]             float32
//   centroids : [16]                     float32 (n_levels=16)
// Output:
//   scores    : [B, H, T, N]             bfloat16
// ─────────────────────────────────────────────────────────────────────────

class TurboquantQkInlinePacked4 : public Primitive {
 public:
  explicit TurboquantQkInlinePacked4(Stream stream) : Primitive(stream) {}

  void eval_cpu(
      const std::vector<array>& /*inputs*/,
      std::vector<array>& /*outputs*/) override {
    throw std::runtime_error(
        "lumen::TurboquantQkInlinePacked4: CPU eval not implemented");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override;

  const char* name() const override {
    return "LumenTurboquantQkInlinePacked4";
  }

  bool is_equivalent(const Primitive& /*other*/) const override {
    return true;
  }

  std::vector<Shape> output_shapes(
      const std::vector<array>& inputs) override {
    const auto& q = inputs[0];
    const auto& k = inputs[1];
    return {{q.shape(0), q.shape(1), q.shape(2), k.shape(2)}};
  }
};

array turboquant_qk_inline_packed4(
    const array& q,
    const array& k_codes_pkd,
    const array& k_sigma,
    const array& centroids,
    StreamOrDevice stream = {});

// ─────────────────────────────────────────────────────────────────────────
// QjlPackSigns / QjlUnpackSigns — bit-pack ±1 signs into u32 words and
// back. Cuts QJL Stage-2 cache footprint by ~16× (bf16 [..., m] → u32
// [..., ceil(m/32)]). Pack reads any float-like tensor and emits the
// sign-bit of each element; unpack does the inverse, returning bf16 ±1.
// ─────────────────────────────────────────────────────────────────────────

class QjlPackSigns : public Primitive {
 public:
  QjlPackSigns(Stream stream, int m) : Primitive(stream), m_(m) {}

  void eval_cpu(
      const std::vector<array>& /*inputs*/,
      std::vector<array>& /*outputs*/) override {
    throw std::runtime_error("lumen::QjlPackSigns: CPU eval not implemented");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override;

  const char* name() const override { return "LumenQjlPackSigns"; }

  bool is_equivalent(const Primitive& other) const override {
    return m_ == static_cast<const QjlPackSigns&>(other).m_;
  }

  std::vector<Shape> output_shapes(
      const std::vector<array>& inputs) override {
    Shape out = inputs[0].shape();
    if (!out.empty()) {
      out.back() = (m_ + 31) / 32;
    }
    return {out};
  }

  int m() const { return m_; }

 private:
  int m_;
};

class QjlUnpackSigns : public Primitive {
 public:
  QjlUnpackSigns(Stream stream, int m) : Primitive(stream), m_(m) {}

  void eval_cpu(
      const std::vector<array>& /*inputs*/,
      std::vector<array>& /*outputs*/) override {
    throw std::runtime_error("lumen::QjlUnpackSigns: CPU eval not implemented");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override;

  const char* name() const override { return "LumenQjlUnpackSigns"; }

  bool is_equivalent(const Primitive& other) const override {
    return m_ == static_cast<const QjlUnpackSigns&>(other).m_;
  }

  std::vector<Shape> output_shapes(
      const std::vector<array>& inputs) override {
    Shape out = inputs[0].shape();
    if (!out.empty()) {
      out.back() = m_;
    }
    return {out};
  }

  int m() const { return m_; }

 private:
  int m_;
};

/// Pack the signs of a float32 input into u32 words. `values` has shape
/// `[..., m]` (any float-like dtype is cast to f32 first); output has shape
/// `[..., ceil(m/32)]` uint32 with bit-j of word-w set iff
/// `values[...,w*32+j] >= 0`. Bits past `m` are zero-padded.
array qjl_pack_signs(const array& values, int m, StreamOrDevice s = {});

/// Unpack u32 sign-bits back to bf16 ±1. Input shape `[..., ceil(m/32)]`;
/// output `[..., m]` with each element ∈ {-1, +1}.
array qjl_unpack_signs(const array& packed, int m, StreamOrDevice s = {});

} // namespace mlx::core::lumen
