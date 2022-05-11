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

#define DEFINE_VISIT(NamePtr)                    \
  void IRVisitorCaching::visit(NamePtr v) {      \
    if (is_cached(v)) {                          \
      return;                                    \
    }                                            \
    save_cache(v);                               \
    visit_impl(v);                               \
  }                                              \
                                                 \
  void IRVisitorCaching::visit_impl(NamePtr v) { \
    IRVisitor::visit(v);                         \
  }
#undef DEFINE_VISIT

#define IMM_DEFINE_VISIT(Type, Name)                  \
  void IRVisitorCaching::visit(Name##ImmPtr v) {      \
    if (is_cached(v)) {                               \
      return;                                         \
    }                                                 \
    save_cache(v);                                    \
    visit_impl(v);                                    \
  }                                                   \
                                                      \
  void IRVisitorCaching::visit_impl(Name##ImmPtr v) { \
    IRVisitor::visit(v);                              \
  }
#undef IMM_DEFINE_VISIT

} // namespace tensorexpr
} // namespace jit
} // namespace torch
