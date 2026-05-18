// Copyright © 2023 Apple Inc.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "mlx/api.h"
#include "mlx/backend/gpu/eval.h"
#include "mlx/device.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mlx::core::scheduler {

// lumen-rs Phase 1.5 Step D (H2) — Scheduler contention stats.
// These atomics live in the header as `inline static` so all TUs share
// one instance without needing a separate .cpp link target. They're
// incremented by Scheduler methods below and read via getters in
// `mlx/backend/metal/metal.h`.
inline std::atomic<uint64_t> g_sched_new_task_count{0};
inline std::atomic<uint64_t> g_sched_completion_count{0};
inline std::atomic<uint64_t> g_sched_lock_wait_ns{0};
inline std::atomic<int32_t> g_sched_max_active_tasks{0};

// Step E (encode-cost hypothesis): per-primitive gpu::eval(arr) wall
// time accumulator. Incremented in transforms.cpp around the
// `gpu::eval(arr)` call inside eval_impl. `calls` counts the number
// of primitives encoded; `ns` accumulates total encode wall.
inline std::atomic<uint64_t> g_eval_gpu_calls{0};
inline std::atomic<uint64_t> g_eval_gpu_ns{0};

// Step F (primitive-type histogram): named counters for the top
// primitive types in the Gemma 4 26B-A4B decode graph dump. Indices:
//   0 RMSNorm   1 QuantizedMatmul  2 Reshape    3 Broadcast
//   4 Multiply  5 Transpose        6 Compiled   7 Other
// Compiled covers any name starting with "Compiled" (compile slots).
// Other captures everything else (ScaledDotProductAttention, RoPE,
// Softmax, Slice, etc. — uncategorized).
inline std::atomic<uint64_t> g_prim_hist_rms_norm{0};
inline std::atomic<uint64_t> g_prim_hist_qmm{0};
inline std::atomic<uint64_t> g_prim_hist_reshape{0};
inline std::atomic<uint64_t> g_prim_hist_broadcast{0};
inline std::atomic<uint64_t> g_prim_hist_multiply{0};
inline std::atomic<uint64_t> g_prim_hist_transpose{0};
inline std::atomic<uint64_t> g_prim_hist_compiled{0};
inline std::atomic<uint64_t> g_prim_hist_other{0};

inline void prim_hist_record(const char* name) {
  if (name == nullptr) {
    g_prim_hist_other.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  // strncmp instead of strcmp so "Compiled..." matches.
  if (std::strcmp(name, "RMSNorm") == 0) {
    g_prim_hist_rms_norm.fetch_add(1, std::memory_order_relaxed);
  } else if (std::strcmp(name, "QuantizedMatmul") == 0) {
    g_prim_hist_qmm.fetch_add(1, std::memory_order_relaxed);
  } else if (std::strcmp(name, "Reshape") == 0) {
    g_prim_hist_reshape.fetch_add(1, std::memory_order_relaxed);
  } else if (std::strcmp(name, "Broadcast") == 0) {
    g_prim_hist_broadcast.fetch_add(1, std::memory_order_relaxed);
  } else if (std::strcmp(name, "Multiply") == 0) {
    g_prim_hist_multiply.fetch_add(1, std::memory_order_relaxed);
  } else if (std::strcmp(name, "Transpose") == 0) {
    g_prim_hist_transpose.fetch_add(1, std::memory_order_relaxed);
  } else if (std::strncmp(name, "Compiled", 8) == 0) {
    g_prim_hist_compiled.fetch_add(1, std::memory_order_relaxed);
  } else {
    g_prim_hist_other.fetch_add(1, std::memory_order_relaxed);
  }
}

inline void prim_hist_reset() {
  g_prim_hist_rms_norm.store(0, std::memory_order_relaxed);
  g_prim_hist_qmm.store(0, std::memory_order_relaxed);
  g_prim_hist_reshape.store(0, std::memory_order_relaxed);
  g_prim_hist_broadcast.store(0, std::memory_order_relaxed);
  g_prim_hist_multiply.store(0, std::memory_order_relaxed);
  g_prim_hist_transpose.store(0, std::memory_order_relaxed);
  g_prim_hist_compiled.store(0, std::memory_order_relaxed);
  g_prim_hist_other.store(0, std::memory_order_relaxed);
}

// Step F3 — AsType dtype-pair counters. Track each AsType primitive by
// (input dtype, output dtype) to identify the upstream source. Common
// pairs for the Gemma 4 bf16 path are bf16->f32 (upcast) and
// f32->bf16 (downcast); a "noop" pair (input==output) indicates a
// redundant cast.
inline std::atomic<uint64_t> g_astype_bf16_to_f32{0};
inline std::atomic<uint64_t> g_astype_f32_to_bf16{0};
inline std::atomic<uint64_t> g_astype_noop{0};
inline std::atomic<uint64_t> g_astype_other_pair{0};

inline void astype_pair_reset() {
  g_astype_bf16_to_f32.store(0, std::memory_order_relaxed);
  g_astype_f32_to_bf16.store(0, std::memory_order_relaxed);
  g_astype_noop.store(0, std::memory_order_relaxed);
  g_astype_other_pair.store(0, std::memory_order_relaxed);
}

// Step F2 — dynamic primitive-type histogram. Captures every distinct
// primitive name with its call count. Mutex-protected (mutex acquire
// ~25 ns × ~3000 prims/step = 75 μs/step, negligible vs 33 ms step).
// Use via `prim_hist_dump_dynamic` to retrieve a newline-separated
// "name=count" string for analysis from the consumer side.
inline std::mutex g_prim_hist_dyn_mtx;
inline std::unordered_map<std::string, uint64_t> g_prim_hist_dyn_map;

inline void prim_hist_dyn_record(const char* name) {
  std::lock_guard<std::mutex> lk(g_prim_hist_dyn_mtx);
  if (name == nullptr) {
    g_prim_hist_dyn_map["<null>"] += 1;
  } else {
    g_prim_hist_dyn_map[name] += 1;
  }
}

inline void prim_hist_dyn_reset() {
  std::lock_guard<std::mutex> lk(g_prim_hist_dyn_mtx);
  g_prim_hist_dyn_map.clear();
}

// Writes "name1=count1\nname2=count2\n..." into `buf`. Returns the
// number of bytes written (excluding NUL terminator) or -1 on truncation.
inline int prim_hist_dyn_dump(char* buf, int buf_size) {
  if (buf == nullptr || buf_size <= 0) {
    return -1;
  }
  std::lock_guard<std::mutex> lk(g_prim_hist_dyn_mtx);
  int written = 0;
  buf[0] = '\0';
  for (auto& kv : g_prim_hist_dyn_map) {
    int remaining = buf_size - written;
    int n = std::snprintf(
        buf + written,
        static_cast<size_t>(remaining),
        "%s=%llu\n",
        kv.first.c_str(),
        static_cast<unsigned long long>(kv.second));
    if (n < 0 || n >= remaining) {
      return -1; // truncated
    }
    written += n;
  }
  return written;
}

inline void record_max_active(int n) {
  int prev = g_sched_max_active_tasks.load(std::memory_order_relaxed);
  while (n > prev && !g_sched_max_active_tasks.compare_exchange_weak(
                         prev, n, std::memory_order_relaxed)) {
    // retry with updated `prev`
  }
}


class StreamThread;

class MLX_API Scheduler {
 public:
  Scheduler();
  ~Scheduler();

  // Not copyable or moveable
  Scheduler(const Scheduler&) = delete;
  Scheduler(Scheduler&&) = delete;
  Scheduler& operator=(const Scheduler&) = delete;
  Scheduler& operator=(Scheduler&&) = delete;

  void enqueue(Stream s, std::function<void()> task);
  void wait_event(Stream s, Event event, std::function<void(Event&)> task);
  void signal_event(Stream s, Event event, std::function<void(Event&)> task);
  void check_error(Stream s);

  void notify_new_task(const Stream& stream) {
    auto t0 = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lk(mtx);
      auto t1 = std::chrono::steady_clock::now();
      g_sched_lock_wait_ns.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
              .count(),
          std::memory_order_relaxed);
      g_sched_new_task_count.fetch_add(1, std::memory_order_relaxed);
      n_active_tasks_++;
      record_max_active(n_active_tasks_);
    }
    completion_cv.notify_all();
  }

  void notify_task_completion(const Stream& stream) {
    auto t0 = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lk(mtx);
      auto t1 = std::chrono::steady_clock::now();
      g_sched_lock_wait_ns.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
              .count(),
          std::memory_order_relaxed);
      g_sched_completion_count.fetch_add(1, std::memory_order_relaxed);
      n_active_tasks_--;
    }
    completion_cv.notify_all();
  }

  int n_active_tasks() const {
    return n_active_tasks_;
  }

  void wait_for_one() {
    std::unique_lock<std::mutex> lk(mtx);
    int n_tasks_old = n_active_tasks();
    if (n_tasks_old > 1) {
      completion_cv.wait(lk, [this, n_tasks_old] {
        return this->n_active_tasks() < n_tasks_old;
      });
    }
  }

 private:
  friend Stream mlx::core::new_stream(Device d);

  StreamThread& get_thread(Stream s);

  int n_active_tasks_{0};
  std::unordered_map<int, std::unique_ptr<StreamThread>> threads_;
  std::shared_mutex threads_mtx_;
  std::condition_variable completion_cv;
  std::mutex mtx;
};

MLX_API Scheduler& scheduler();

template <typename F>
inline void enqueue(Stream s, F&& f) {
  scheduler().enqueue(s, std::forward<F>(f));
}

// Like enqueue but the task is used for processing the passed event.
template <typename F>
inline void wait_event(Stream s, Event event, F&& f) {
  scheduler().wait_event(s, std::move(event), std::forward<F>(f));
}

template <typename F>
inline void signal_event(Stream s, Event event, F&& f) {
  scheduler().signal_event(s, std::move(event), std::forward<F>(f));
}

// Throw and clear the error stored in the stream, if any.
inline void check_error(Stream s) {
  scheduler().check_error(s);
}

inline int n_active_tasks() {
  return scheduler().n_active_tasks();
}

inline void notify_new_task(const Stream& stream) {
  scheduler().notify_new_task(stream);
}

inline void notify_task_completion(const Stream& stream) {
  scheduler().notify_task_completion(stream);
}

inline void wait_for_one() {
  scheduler().wait_for_one();
}

} // namespace mlx::core::scheduler
