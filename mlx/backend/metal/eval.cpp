// Copyright © 2023-2024 Apple Inc.
#include <memory>
#include <mutex>
#include <string>

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

namespace mlx::core::gpu {

void new_stream(Stream stream) {
  if (stream.device == mlx::core::Device::gpu) {
    metal::device(stream.device).new_queue(stream.index);
  }
}

inline void check_error(MTL::CommandBuffer* cbuf) {
  if (cbuf->status() == MTL::CommandBufferStatusError) {
    std::ostringstream msg;
    msg << "[METAL] Command buffer execution failed: "
        << cbuf->error()->localizedDescription()->utf8String();
    throw std::runtime_error(msg.str());
  }
}

// lumen-rs: async command-buffer failures must not be thrown where they land.
//
// `addCompletedHandler` callbacks are invoked by the Metal driver on its own
// dispatch thread. There is no `catch` anywhere on that stack, so throwing from
// one calls std::terminate and takes the whole process down — which is exactly
// what a GPU out-of-memory did: `libc++abi: terminating due to uncaught
// exception of type std::runtime_error: [METAL] Command buffer execution
// failed: Insufficient Memory`. A server serving many requests died because one
// of them asked for too much memory.
//
// It cannot be caught downstream either: mlx-c wraps `mlx_eval` in try/catch
// and would happily turn the exception into a status code, but that handler is
// on a different thread entirely.
//
// So the failure is recorded here and re-thrown on the CALLING thread at the
// next entry point — `gpu::eval` and `synchronize`, both of which run inline on
// the thread that called into MLX. From there mlx-c's existing try/catch turns
// it into a status code and the caller gets an error instead of a corpse.
//
// The slot is cleared when it is thrown. A command buffer that ran out of
// memory leaves the device perfectly usable; keeping the error sticky would
// fail every later request for one earlier request's mistake.
namespace {

std::mutex& async_error_mutex() {
  static std::mutex m;
  return m;
}

std::string& async_error_slot() {
  static std::string s;
  return s;
}

// Runs on Metal's completion thread. Records; never throws.
void record_error(MTL::CommandBuffer* cbuf) {
  if (cbuf->status() != MTL::CommandBufferStatusError) {
    return;
  }
  std::ostringstream msg;
  msg << "[METAL] Command buffer execution failed: "
      << cbuf->error()->localizedDescription()->utf8String();
  std::lock_guard<std::mutex> lock(async_error_mutex());
  // First failure wins: it is the one with a cause, and the ones after it are
  // usually its fallout.
  if (async_error_slot().empty()) {
    async_error_slot() = msg.str();
  }
}

// Runs on the calling thread, where an exception has somewhere to go.
void throw_if_async_error() {
  std::string err;
  {
    std::lock_guard<std::mutex> lock(async_error_mutex());
    if (async_error_slot().empty()) {
      return;
    }
    err.swap(async_error_slot());
  }
  throw std::runtime_error(err);
}

} // namespace

void eval(array& arr) {
  // A command buffer that failed asynchronously is reported here, on the
  // thread that called in, rather than from the Metal callback that noticed it.
  throw_if_async_error();
  auto pool = metal::new_scoped_memory_pool();
  auto s = arr.primitive().stream();
  auto& d = metal::device(s.device);
  auto command_buffer = d.get_command_buffer(s.index);

  auto outputs = arr.outputs();
  {
    // If the array is a tracer hold a reference
    // to its inputs so they don't get donated
    std::vector<array> inputs;
    if (arr.is_tracer()) {
      inputs = arr.inputs();
    }

    debug_set_primitive_buffer_label(command_buffer, arr.primitive());
    arr.primitive().eval_gpu(arr.inputs(), outputs);
  }
  std::unordered_set<std::shared_ptr<array::Data>> buffers;
  for (auto& in : arr.inputs()) {
    buffers.insert(in.data_shared_ptr());
  }
  for (auto& s : arr.siblings()) {
    buffers.insert(s.data_shared_ptr());
  }
  // Remove the output if it was donated to by an input
  if (auto it = buffers.find(arr.data_shared_ptr()); it != buffers.end()) {
    buffers.erase(it);
  }

  if (d.command_buffer_needs_commit(s.index)) {
    d.end_encoding(s.index);
    scheduler::notify_new_task(s);
    command_buffer->addCompletedHandler(
        [s, buffers = std::move(buffers)](MTL::CommandBuffer* cbuf) {
          scheduler::notify_task_completion(s);
          record_error(cbuf);
        });
    d.commit_command_buffer(s.index);
    d.get_command_buffer(s.index);
  } else {
    command_buffer->addCompletedHandler(
        [buffers = std::move(buffers)](MTL::CommandBuffer* cbuf) {
          record_error(cbuf);
        });
  }
}

void finalize(Stream s) {
  auto pool = metal::new_scoped_memory_pool();
  auto& d = metal::device(s.device);
  auto cb = d.get_command_buffer(s.index);
  d.end_encoding(s.index);
  cb->addCompletedHandler([](MTL::CommandBuffer* cbuf) { record_error(cbuf); });
  d.commit_command_buffer(s.index);
  d.get_command_buffer(s.index);
}

void synchronize(Stream s) {
  auto pool = metal::new_scoped_memory_pool();
  auto& d = metal::device(s.device);
  auto cb = d.get_command_buffer(s.index);
  cb->retain();
  d.end_encoding(s.index);
  d.commit_command_buffer(s.index);
  cb->waitUntilCompleted();
  check_error(cb);
  cb->release();
  // This buffer was fine, but an earlier one on the stream may not have been.
  throw_if_async_error();
}

} // namespace mlx::core::gpu
