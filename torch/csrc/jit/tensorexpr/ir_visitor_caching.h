#pragma once
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <unordered_set>

namespace torch {
namespace jit {
namespace tensorexpr {

class IRVisitorCaching : public IRVisitor {
 public:

 private:
  std::unordered_set<void*> cache_;

  template <typename T>
  bool is_cached(const std::shared_ptr<T>& ptr);
  template <typename T>
  void save_cache(const std::shared_ptr<T>& ptr);
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
