#pragma once
#include <c10/core/ScalarType.h>
#include <torch/csrc/Export.h>
#include <vector>

#include <torch/csrc/jit/tensorexpr/ir_mutator_caching.h>

namespace torch {
namespace jit {
namespace tensorexpr {

class TORCH_API IRCloner : public IRMutatorCaching {
 public:
  ~IRCloner() override = default;
  ExprPtr mutate_impl(AddPtr v) override;
  ExprPtr mutate_impl(SubPtr v) override;
  ExprPtr mutate_impl(MulPtr v) override;
  ExprPtr mutate_impl(DivPtr v) override;
  ExprPtr mutate_impl(ModPtr v) override;
  ExprPtr mutate_impl(MaxPtr v) override;
  ExprPtr mutate_impl(MinPtr v) override;
  ExprPtr mutate_impl(AndPtr v) override;
  ExprPtr mutate_impl(OrPtr v) override;
  ExprPtr mutate_impl(XorPtr v) override;
  ExprPtr mutate_impl(LshiftPtr v) override;
  ExprPtr mutate_impl(RshiftPtr v) override;
  ExprPtr mutate_impl(CompareSelectPtr v) override;
#define IMM_MUTATE_DECLARE(Type, Name) ExprPtr mutate_impl(Name##ImmPtr v) override;
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_MUTATE_DECLARE);
#undef IMM_MUTATE_DECLARE
  ExprPtr mutate_impl(CastPtr v) override;
  ExprPtr mutate_impl(BitCastPtr v) override;
  ExprPtr mutate_impl(VarPtr v) override;
  ExprPtr mutate_impl(BufPtr v) override;
  ExprPtr mutate_impl(RampPtr v) override;
  ExprPtr mutate_impl(LoadPtr v) override;
  ExprPtr mutate_impl(BroadcastPtr v) override;
  ExprPtr mutate_impl(IfThenElsePtr v) override;
  ExprPtr mutate_impl(IntrinsicsPtr v) override;

  ExprPtr mutate_impl(TermPtr v) override;
  ExprPtr mutate_impl(PolynomialPtr v) override;
  ExprPtr mutate_impl(RoundOffPtr v) override;
  ExprPtr mutate_impl(MaxTermPtr v) override;
  ExprPtr mutate_impl(MinTermPtr v) override;

  ExprPtr mutate_impl(ReduceOpPtr v) override;

  StmtPtr mutate_impl(ForPtr v) override;
  StmtPtr mutate_impl(BlockPtr v) override;
  StmtPtr mutate_impl(StorePtr v) override;
  StmtPtr mutate_impl(AtomicAddPtr v) override;
  StmtPtr mutate_impl(SyncThreadsPtr v) override;
  StmtPtr mutate_impl(ExternalCallPtr v) override;
  StmtPtr mutate_impl(ExternalCallWithAllocPtr v) override;

  StmtPtr mutate_impl(AllocatePtr v) override;
  StmtPtr mutate_impl(FreePtr v) override;
  StmtPtr mutate_impl(LetPtr v) override;
  StmtPtr mutate_impl(CondPtr v) override;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
