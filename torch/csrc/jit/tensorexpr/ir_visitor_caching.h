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

#define PRINT_VISIT(NamePtr)                    \
  virtual void visit(NamePtr v) override final; \
  virtual void visit_impl(NamePtr v);
  FORALL_IR_VISITORS(PRINT_VISIT)
#undef PRINT_VISIT

#define IMM_PRINT_VISIT(Type, Name)                  \
  virtual void visit(Name##ImmPtr v) override final; \
  virtual void visit_impl(Name##ImmPtr v);
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_PRINT_VISIT)
#undef IMM_PRINT_VISIT

  template <typename T>
  bool is_cached(const std::shared_ptr<T>& ptr);
  template <typename T>
  void save_cache(const std::shared_ptr<T>& ptr);
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
