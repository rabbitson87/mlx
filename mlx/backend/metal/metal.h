// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>

#include "mlx/api.h"

namespace mlx::core::metal {

/* Check if the Metal backend is available. */
MLX_API bool is_available();

/** Capture a GPU trace, saving it to an absolute file `path` */
MLX_API void start_capture(std::string path = "");
MLX_API void stop_capture();

/** lumen-rs Phase 1.5 deep-dive: kernel-cache hit/miss counters.
 *
 * Returns the current cumulative counts for `Device::get_kernel(...)`
 * lookups on the default GPU device. Incremented on every dispatch path
 * that resolves a `(name, hash, func_consts)` tuple to a Metal pipeline
 * state. `reset_*` zeroes both counters atomically so the caller can
 * measure deltas around a benchmark window. */
MLX_API uint64_t kernel_cache_hits();
MLX_API uint64_t kernel_cache_misses();
MLX_API void reset_kernel_cache_stats();

/** lumen-rs Phase 1.5 Step C: command buffer batching counters. */
MLX_API uint64_t cmd_buffer_commits();
MLX_API uint64_t cmd_buffer_ops_total();
MLX_API void reset_cmd_buffer_stats();

/** lumen-rs Phase 1.5 Step D (H2): Scheduler contention counters.
 *
 * `new_task_count` / `completion_count` count the per-stream task lifecycle
 * notifications from `eval.cpp` (called per command buffer commit). The
 * scheduler holds a single global mutex (`mlx::core::scheduler::mtx`)
 * around `n_active_tasks_++/--`. `lock_wait_ns` accumulates the nanoseconds
 * spent waiting to acquire that mutex across both notification sites —
 * if our Rust pattern induces multi-thread contention vs mlx-lm's
 * GIL-serialized Python, this is where it will show.
 *
 * `max_active_tasks` records the peak depth of `n_active_tasks_` since
 * the last reset (high values = many in-flight cmd buffers waiting for
 * GPU completion). */
MLX_API uint64_t scheduler_new_task_count();
MLX_API uint64_t scheduler_completion_count();
MLX_API uint64_t scheduler_lock_wait_ns();
MLX_API int32_t scheduler_max_active_tasks();
MLX_API void reset_scheduler_stats();

/** Step E: per-primitive gpu::eval encode time accumulator. */
MLX_API uint64_t eval_gpu_calls();
MLX_API uint64_t eval_gpu_ns();
MLX_API void reset_eval_gpu_stats();

/** Step F: primitive-type histogram (top 6 + Compiled + Other). */
MLX_API uint64_t prim_hist_rms_norm();
MLX_API uint64_t prim_hist_qmm();
MLX_API uint64_t prim_hist_reshape();
MLX_API uint64_t prim_hist_broadcast();
MLX_API uint64_t prim_hist_multiply();
MLX_API uint64_t prim_hist_transpose();
MLX_API uint64_t prim_hist_compiled();
MLX_API uint64_t prim_hist_other();
MLX_API void reset_prim_hist();

/** Step F2: dynamic primitive-type histogram dump. */
MLX_API int prim_hist_dump_dynamic(char* buf, int buf_size);
MLX_API void reset_prim_hist_dynamic();

/** Step F3: AsType dtype-pair counters. */
MLX_API uint64_t astype_bf16_to_f32();
MLX_API uint64_t astype_f32_to_bf16();
MLX_API uint64_t astype_noop();
MLX_API uint64_t astype_other_pair();
MLX_API void reset_astype_pair();

/** Get information about the GPU and system settings. */
MLX_API const
    std::unordered_map<std::string, std::variant<std::string, size_t>>&
    device_info();

/* Set a custom path to mlx.metallib. Must be called before any MLX operation.
 */
MLX_API void set_metallib_path(const std::string& path);
MLX_API const std::string& get_metallib_path();

} // namespace mlx::core::metal
