#include <torch/csrc/jit/tensorexpr/ir_mutator_caching.h>

namespace torch {
namespace jit {
namespace tensorexpr {

IRMutatorCaching::IRMutatorCaching(IRMutator impl) : impl_(std::move(impl)) {}

template<typename T>
ExprPtr IRMutatorCaching::get_cached_expr(const std::shared_ptr<T>& ptr) {
  auto it = expr_cache_.find((void*) ptr.get());
  if (it == expr_cache_.end()) {
    return nullptr;
  }
  return it->second;
}

template<typename T>
StmtPtr IRMutatorCaching::get_cached_stmt(const std::shared_ptr<T>& ptr) {
  auto it = stmt_cache_.find((void*) ptr.get());
  if (it == stmt_cache_.end()) {
    return nullptr;
  }
  return it->second;
}

template<typename T>
ExprPtr IRMutatorCaching::set_cached_expr(const std::shared_ptr<T>& ptr, ExprPtr v) {
  expr_cache_[(void*) ptr.get()] = v;
  return v;
}

template<typename T>
StmtPtr IRMutatorCaching::set_cached_stmt(const std::shared_ptr<T>& ptr, StmtPtr v) {
  stmt_cache_[(void*) ptr.get()] = v;
  return v;
}

#define DEFINE_EXPR_MUTATOR_CACHING(PtrType)  \
ExprPtr IRMutatorCaching::mutate(PtrType v) { \
  if (auto cached = get_cached_expr(v)) {     \
    return cached;                            \
  }                                           \
  return set_cached_expr(v, impl_.mutate(v)); \
}


#define DEFINE_STMT_MUTATOR_CACHING(PtrType)  \
StmtPtr IRMutatorCaching::mutate(PtrType v) { \
  if (auto cached = get_cached_stmt(v)) {     \
    return cached;                            \
  }                                           \
  return set_cached_stmt(v, impl_.mutate(v)); \
}

DEFINE_EXPR_MUTATOR_CACHING(AddPtr)
DEFINE_EXPR_MUTATOR_CACHING(SubPtr)
DEFINE_EXPR_MUTATOR_CACHING(MulPtr)
DEFINE_EXPR_MUTATOR_CACHING(DivPtr)
DEFINE_EXPR_MUTATOR_CACHING(ModPtr)
DEFINE_EXPR_MUTATOR_CACHING(MaxPtr)
DEFINE_EXPR_MUTATOR_CACHING(MinPtr)
DEFINE_EXPR_MUTATOR_CACHING(AndPtr)
DEFINE_EXPR_MUTATOR_CACHING(OrPtr)
DEFINE_EXPR_MUTATOR_CACHING(XorPtr)
DEFINE_EXPR_MUTATOR_CACHING(LshiftPtr)
DEFINE_EXPR_MUTATOR_CACHING(RshiftPtr)
DEFINE_EXPR_MUTATOR_CACHING(CompareSelectPtr)

#define IMM_MUTATE_DECLARE(Type, Name) DEFINE_EXPR_MUTATOR_CACHING(Name##ImmPtr)
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_MUTATE_DECLARE);
#undef IMM_MUTATE_DECLARE

DEFINE_EXPR_MUTATOR_CACHING(CastPtr)
DEFINE_EXPR_MUTATOR_CACHING(BitCastPtr)
DEFINE_EXPR_MUTATOR_CACHING(VarPtr)
DEFINE_EXPR_MUTATOR_CACHING(BufPtr)
DEFINE_EXPR_MUTATOR_CACHING(RampPtr)
DEFINE_EXPR_MUTATOR_CACHING(LoadPtr)
DEFINE_EXPR_MUTATOR_CACHING(BroadcastPtr)
DEFINE_EXPR_MUTATOR_CACHING(IfThenElsePtr)
DEFINE_EXPR_MUTATOR_CACHING(IntrinsicsPtr)

DEFINE_EXPR_MUTATOR_CACHING(TermPtr)
DEFINE_EXPR_MUTATOR_CACHING(PolynomialPtr)
DEFINE_EXPR_MUTATOR_CACHING(RoundOffPtr)
DEFINE_EXPR_MUTATOR_CACHING(MaxTermPtr)
DEFINE_EXPR_MUTATOR_CACHING(MinTermPtr)

DEFINE_EXPR_MUTATOR_CACHING(ReduceOpPtr)

DEFINE_STMT_MUTATOR_CACHING(ForPtr)
DEFINE_STMT_MUTATOR_CACHING(BlockPtr)
DEFINE_STMT_MUTATOR_CACHING(StorePtr)
DEFINE_STMT_MUTATOR_CACHING(AtomicAddPtr)
DEFINE_STMT_MUTATOR_CACHING(SyncThreadsPtr)
DEFINE_STMT_MUTATOR_CACHING(ExternalCallPtr)
DEFINE_STMT_MUTATOR_CACHING(ExternalCallWithAllocPtr)

DEFINE_STMT_MUTATOR_CACHING(AllocatePtr)
DEFINE_STMT_MUTATOR_CACHING(FreePtr)
DEFINE_STMT_MUTATOR_CACHING(LetPtr)
DEFINE_STMT_MUTATOR_CACHING(CondPtr)

#undef DEFINE_EXPR_MUTATOR_CACHING
#undef DEFINE_STMT_MUTATOR_CACHING

} // namespace tensorexpr
} // namespace jit
} // namespace torch
