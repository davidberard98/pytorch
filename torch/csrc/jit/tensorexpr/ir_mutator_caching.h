#pragma once
#include <torch/csrc/jit/tensorexpr/fwd_decls.h>
#include <torch/csrc/jit/tensorexpr/ir_mutator.h>

#include <memory>
#include <unordered_map>

namespace torch {
namespace jit {
namespace tensorexpr {

class IRMutatorCaching : public IRMutator {
 public:
  virtual ~IRMutatorCaching() = default;
#define EXPR_DECLARE(Type) \
  virtual ExprPtr mutate(Type##Ptr v) final; \
  virtual ExprPtr mutate_impl(Type##Ptr v);

  EXPR_DECLARE(Add);
  EXPR_DECLARE(Sub);
  EXPR_DECLARE(Mul);
  EXPR_DECLARE(Div);
  EXPR_DECLARE(Mod);
  EXPR_DECLARE(Max);
  EXPR_DECLARE(Min);
  EXPR_DECLARE(And);
  EXPR_DECLARE(Or);
  EXPR_DECLARE(Xor);
  EXPR_DECLARE(Lshift);
  EXPR_DECLARE(Rshift);
  EXPR_DECLARE(CompareSelect);
  EXPR_DECLARE(Cast);

#define IMM_MUTATE_DECLARE(Type, Name) EXPR_DECLARE(Name##Imm);
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_MUTATE_DECLARE);
#undef IMM_MUTATE_DECLARE

  EXPR_DECLARE(BitCast);
  EXPR_DECLARE(Var);
  EXPR_DECLARE(Buf);
  EXPR_DECLARE(Ramp);
  EXPR_DECLARE(Load);
  EXPR_DECLARE(Broadcast);
  EXPR_DECLARE(IfThenElse);
  EXPR_DECLARE(Intrinsics);

  EXPR_DECLARE(Term);
  EXPR_DECLARE(Polynomial);
  EXPR_DECLARE(RoundOff);
  EXPR_DECLARE(MaxTerm);
  EXPR_DECLARE(MinTerm);

  EXPR_DECLARE(ReduceOp);

#undef EXPR_DECLARE

#define STMT_DECLARE(Type) \
  virtual StmtPtr mutate(Type##Ptr v) final; \
  virtual StmtPtr mutate_impl(Type##Ptr v);

  STMT_DECLARE(For);
  STMT_DECLARE(Block);
  STMT_DECLARE(Store);
  STMT_DECLARE(AtomicAdd);
  STMT_DECLARE(SyncThreads);
  STMT_DECLARE(ExternalCall);
  STMT_DECLARE(ExternalCallWithAlloc);

  STMT_DECLARE(Allocate);
  STMT_DECLARE(Free);
  STMT_DECLARE(FreeExt);
  STMT_DECLARE(PlacementAllocate);
  STMT_DECLARE(Let);
  STMT_DECLARE(Cond);
#undef STMT_DECLARE

 private:
  std::unordered_map<void*, StmtPtr> stmt_cache_;
  std::unordered_map<void*, ExprPtr> expr_cache_;

  template <typename T>
  StmtPtr get_stmt_from_cache(const std::shared_ptr<T>& ptr);
  template <typename T>
  ExprPtr get_expr_from_cache(const std::shared_ptr<T>& ptr);
  template <typename T>
  StmtPtr cache_stmt(const std::shared_ptr<T>& ptr, StmtPtr copy);
  template <typename T>
  ExprPtr cache_expr(const std::shared_ptr<T>& ptr, ExprPtr copy);
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
