#include <torch/csrc/jit/tensorexpr/ir_mutator_caching.h>

namespace torch {
namespace jit {
namespace tensorexpr {

template <typename T>
StmtPtr IRMutatorCaching::get_stmt_from_cache(const std::shared_ptr<T>& ptr) {
  auto it = stmt_cache_.find((void*) ptr.get());
  if (it == stmt_cache_.end()) {
    return nullptr;
  }
  return it->second;
}

template <typename T>
ExprPtr IRMutatorCaching::get_expr_from_cache(const std::shared_ptr<T>& ptr) {
  auto it = expr_cache_.find((void*) ptr.get());
  if (it == expr_cache_.end()) {
    return nullptr;
  }
  return it->second;
}

template <typename T>
StmtPtr IRMutatorCaching::cache_stmt(const std::shared_ptr<T>& ptr, StmtPtr copy) {
  stmt_cache_[(void*) ptr.get()] = copy;
  return copy;
}

template <typename T>
ExprPtr IRMutatorCaching::cache_expr(const std::shared_ptr<T>& ptr, ExprPtr copy) {
  expr_cache_[(void*) ptr.get()] = copy;
  return copy;
}

#define EXPR_TRY_RETURN_CACHED(v)             \
if (auto cached = get_expr_from_cache(v))   { \
  return cached;                              \
}

#define EXPR_DEFINE(Type)                            \
ExprPtr IRMutatorCaching::mutate(Type##Ptr v) {      \
  if (auto cached = get_expr_from_cache(v)) {        \
    return cached;                                   \
  }                                                  \
  return cache_expr(v, mutate_impl(v));              \
}                                                    \
                                                     \
ExprPtr IRMutatorCaching::mutate_impl(Type##Ptr v) { \
  return IRMutator::mutate(v);                       \
}

  EXPR_DEFINE(Add);
  EXPR_DEFINE(Sub);
  EXPR_DEFINE(Mul);
  EXPR_DEFINE(Div);
  EXPR_DEFINE(Mod);
  EXPR_DEFINE(Max);
  EXPR_DEFINE(Min);
  EXPR_DEFINE(And);
  EXPR_DEFINE(Or);
  EXPR_DEFINE(Xor);
  EXPR_DEFINE(Lshift);
  EXPR_DEFINE(Rshift);
  EXPR_DEFINE(CompareSelect);
  EXPR_DEFINE(Cast);

#define IMM_MUTATE_DECLARE(Type, Name) EXPR_DEFINE(Name##Imm)
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_MUTATE_DECLARE);
#undef IMM_MUTATE_DECLARE

  EXPR_DEFINE(BitCast);
  EXPR_DEFINE(Var);
  EXPR_DEFINE(Buf);
  EXPR_DEFINE(Ramp);
  EXPR_DEFINE(Load);
  EXPR_DEFINE(Broadcast);
  EXPR_DEFINE(IfThenElse);
  EXPR_DEFINE(Intrinsics);

  EXPR_DEFINE(Term);
  EXPR_DEFINE(Polynomial);
  EXPR_DEFINE(RoundOff);
  EXPR_DEFINE(MaxTerm);
  EXPR_DEFINE(MinTerm);

  EXPR_DEFINE(ReduceOp);

#undef EXPR_DEFINE

#define STMT_DEFINE(Type)                            \
StmtPtr IRMutatorCaching::mutate(Type##Ptr v) {      \
  if (auto cached = get_stmt_from_cache(v)) {        \
    return cached;                                   \
  }                                                  \
  return cache_stmt(v, mutate_impl(v));              \
}                                                    \
                                                     \
StmtPtr IRMutatorCaching::mutate_impl(Type##Ptr v) { \
  return IRMutator::mutate(v);                       \
}

  STMT_DEFINE(For);
  STMT_DEFINE(Block);
  STMT_DEFINE(Store);
  STMT_DEFINE(AtomicAdd);
  STMT_DEFINE(SyncThreads);
  STMT_DEFINE(ExternalCall);
  STMT_DEFINE(ExternalCallWithAlloc);

  STMT_DEFINE(Allocate);
  STMT_DEFINE(Free);
  STMT_DEFINE(FreeExt);
  STMT_DEFINE(PlacementAllocate);
  STMT_DEFINE(Let);
  STMT_DEFINE(Cond);

#undef STMT_DEFINE

} // namespace tensorexpr
} // namespace jit
} // namespace torch
