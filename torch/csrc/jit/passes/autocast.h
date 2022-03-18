
#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <aten/src/ATen/autocast_mode.h>

namespace torch {
namespace jit {

namespace {
at::autocast::AutocastContext default_init_context = {
  false,
  false,
  at::kHalf,
  at::kBFloat16,
};
} // namespace

TORCH_API void Autocast(const std::shared_ptr<Graph>& graph, at::autocast::AutocastContext init_context = default_init_context);

TORCH_API bool setAutocastMode(bool value);
TORCH_API bool autocastEnabled();

} // namespace jit
} // namespace torch
