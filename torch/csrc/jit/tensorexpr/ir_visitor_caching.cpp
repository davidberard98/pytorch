#include <torch/csrc/jit/tensorexpr/ir_visitor_caching.h>

namespace torch {
namespace jit {
namespace tensorexpr {

template <typename T>
bool IRVisitorCaching::is_cached(const std::shared_ptr<T>& ptr) {
  return cache_.count((void*) ptr.get());
}

template <typename T>
void IRVisitorCaching::save_cache(const std::shared_ptr<T>& ptr) {
  cache_.insert((void*) ptr.get());
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
