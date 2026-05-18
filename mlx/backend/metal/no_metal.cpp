// Copyright © 2025 Apple Inc.

#include <stdexcept>

#include "mlx/backend/metal/metal.h"

namespace mlx::core {

namespace metal {

bool is_available() {
  return false;
}

void start_capture(std::string) {}
void stop_capture() {}

uint64_t kernel_cache_hits() {
  return 0;
}
uint64_t kernel_cache_misses() {
  return 0;
}
void reset_kernel_cache_stats() {}

uint64_t cmd_buffer_commits() {
  return 0;
}
uint64_t cmd_buffer_ops_total() {
  return 0;
}
void reset_cmd_buffer_stats() {}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info() {
  throw std::runtime_error(
      "[metal::device_info] Cannot get device info without metal backend");
};

void set_metallib_path(const std::string& path) {}

const std::string& get_metallib_path() {
  throw std::runtime_error(
      "[metal::get_metallib_path] Cannot get metallib path without metal backend");
}

} // namespace metal

} // namespace mlx::core
