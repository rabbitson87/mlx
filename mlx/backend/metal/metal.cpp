// Copyright © 2023-2024 Apple Inc.
#include <memory>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/scheduler.h"

namespace mlx::core::metal {

namespace {

std::string g_metallib_path;

} // namespace

bool is_available() {
  return true;
}

void start_capture(std::string path, NS::Object* object) {
  auto pool = new_scoped_memory_pool();

  auto descriptor = MTL::CaptureDescriptor::alloc()->init()->autorelease();
  descriptor->setCaptureObject(object);

  if (!path.empty()) {
    auto string = NS::String::string(path.c_str(), NS::UTF8StringEncoding);
    auto url = NS::URL::fileURLWithPath(string);
    descriptor->setDestination(MTL::CaptureDestinationGPUTraceDocument);
    descriptor->setOutputURL(url);
  }

  auto manager = MTL::CaptureManager::sharedCaptureManager();
  NS::Error* error;
  bool started = manager->startCapture(descriptor, &error);
  if (!started) {
    std::ostringstream msg;
    msg << "[metal::start_capture] Failed to start: "
        << error->localizedDescription()->utf8String();
    throw std::runtime_error(msg.str());
  }
}

void start_capture(std::string path) {
  auto& device = metal::device(mlx::core::Device::gpu);
  return start_capture(path, device.mtl_device());
}

void stop_capture() {
  auto pool = new_scoped_memory_pool();
  auto manager = MTL::CaptureManager::sharedCaptureManager();
  manager->stopCapture();
}

void set_metallib_path(const std::string& path) {
  g_metallib_path = path;
}

const std::string& get_metallib_path() {
  return g_metallib_path;
}

// lumen-rs Phase 1.5 deep-dive — kernel cache stats getters/reset.
uint64_t kernel_cache_hits() {
  auto& d = metal::device(mlx::core::Device::gpu);
  return d.kernel_cache_hits();
}

uint64_t kernel_cache_misses() {
  auto& d = metal::device(mlx::core::Device::gpu);
  return d.kernel_cache_misses();
}

void reset_kernel_cache_stats() {
  auto& d = metal::device(mlx::core::Device::gpu);
  d.reset_kernel_cache_stats();
}

// Step C: command buffer batching stats.
uint64_t cmd_buffer_commits() {
  auto& d = metal::device(mlx::core::Device::gpu);
  return d.cmd_buffer_commits();
}

uint64_t cmd_buffer_ops_total() {
  auto& d = metal::device(mlx::core::Device::gpu);
  return d.cmd_buffer_ops_total();
}

void reset_cmd_buffer_stats() {
  auto& d = metal::device(mlx::core::Device::gpu);
  d.reset_cmd_buffer_stats();
}

// Step D (H2): scheduler contention stats.
uint64_t scheduler_new_task_count() {
  return mlx::core::scheduler::g_sched_new_task_count.load(
      std::memory_order_relaxed);
}

uint64_t scheduler_completion_count() {
  return mlx::core::scheduler::g_sched_completion_count.load(
      std::memory_order_relaxed);
}

uint64_t scheduler_lock_wait_ns() {
  return mlx::core::scheduler::g_sched_lock_wait_ns.load(
      std::memory_order_relaxed);
}

int32_t scheduler_max_active_tasks() {
  return mlx::core::scheduler::g_sched_max_active_tasks.load(
      std::memory_order_relaxed);
}

void reset_scheduler_stats() {
  mlx::core::scheduler::g_sched_new_task_count.store(
      0, std::memory_order_relaxed);
  mlx::core::scheduler::g_sched_completion_count.store(
      0, std::memory_order_relaxed);
  mlx::core::scheduler::g_sched_lock_wait_ns.store(
      0, std::memory_order_relaxed);
  mlx::core::scheduler::g_sched_max_active_tasks.store(
      0, std::memory_order_relaxed);
}

// Step E: per-primitive gpu::eval encode time.
uint64_t eval_gpu_calls() {
  return mlx::core::scheduler::g_eval_gpu_calls.load(
      std::memory_order_relaxed);
}

uint64_t eval_gpu_ns() {
  return mlx::core::scheduler::g_eval_gpu_ns.load(
      std::memory_order_relaxed);
}

void reset_eval_gpu_stats() {
  mlx::core::scheduler::g_eval_gpu_calls.store(
      0, std::memory_order_relaxed);
  mlx::core::scheduler::g_eval_gpu_ns.store(
      0, std::memory_order_relaxed);
}

// Step F: primitive-type histogram.
uint64_t prim_hist_rms_norm() {
  return mlx::core::scheduler::g_prim_hist_rms_norm.load(
      std::memory_order_relaxed);
}
uint64_t prim_hist_qmm() {
  return mlx::core::scheduler::g_prim_hist_qmm.load(
      std::memory_order_relaxed);
}
uint64_t prim_hist_reshape() {
  return mlx::core::scheduler::g_prim_hist_reshape.load(
      std::memory_order_relaxed);
}
uint64_t prim_hist_broadcast() {
  return mlx::core::scheduler::g_prim_hist_broadcast.load(
      std::memory_order_relaxed);
}
uint64_t prim_hist_multiply() {
  return mlx::core::scheduler::g_prim_hist_multiply.load(
      std::memory_order_relaxed);
}
uint64_t prim_hist_transpose() {
  return mlx::core::scheduler::g_prim_hist_transpose.load(
      std::memory_order_relaxed);
}
uint64_t prim_hist_compiled() {
  return mlx::core::scheduler::g_prim_hist_compiled.load(
      std::memory_order_relaxed);
}
uint64_t prim_hist_other() {
  return mlx::core::scheduler::g_prim_hist_other.load(
      std::memory_order_relaxed);
}
void reset_prim_hist() {
  mlx::core::scheduler::prim_hist_reset();
}

// Step F2: dynamic primitive-type histogram.
int prim_hist_dump_dynamic(char* buf, int buf_size) {
  return mlx::core::scheduler::prim_hist_dyn_dump(buf, buf_size);
}

void reset_prim_hist_dynamic() {
  mlx::core::scheduler::prim_hist_dyn_reset();
}

// Step F3: AsType dtype-pair counters.
uint64_t astype_bf16_to_f32() {
  return mlx::core::scheduler::g_astype_bf16_to_f32.load(
      std::memory_order_relaxed);
}
uint64_t astype_f32_to_bf16() {
  return mlx::core::scheduler::g_astype_f32_to_bf16.load(
      std::memory_order_relaxed);
}
uint64_t astype_noop() {
  return mlx::core::scheduler::g_astype_noop.load(
      std::memory_order_relaxed);
}
uint64_t astype_other_pair() {
  return mlx::core::scheduler::g_astype_other_pair.load(
      std::memory_order_relaxed);
}
void reset_astype_pair() {
  mlx::core::scheduler::astype_pair_reset();
}

} // namespace mlx::core::metal
