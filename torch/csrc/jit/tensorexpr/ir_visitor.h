#pragma once
#include <c10/core/ScalarType.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/tensorexpr/fwd_decls.h>

#define FORALL_IR_VISITORS(M) \
  M(AddPtr)                   \
  M(SubPtr)                   \
  M(MulPtr)                   \
  M(DivPtr)                   \
  M(ModPtr)                   \
  M(MaxPtr)                   \
  M(MinPtr)                   \
  M(AndPtr)                   \
  M(OrPtr)                    \
  M(XorPtr)                   \
  M(LshiftPtr)                \
  M(RshiftPtr)                \
  M(CompareSelectPtr)         \
  M(CastPtr)                  \
  M(BitCastPtr)               \
  M(VarPtr)                   \
  M(BufPtr)                   \
  M(RampPtr)                  \
  M(LoadPtr)                  \
  M(ForPtr)                   \
  M(BlockPtr)                 \
  M(StorePtr)                 \
  M(BroadcastPtr)             \
  M(IfThenElsePtr)            \
  M(IntrinsicsPtr)            \
  M(AllocatePtr)              \
  M(FreePtr)                  \
  M(FreeExtPtr)               \
  M(PlacementAllocatePtr)     \
  M(LetPtr)                   \
  M(CondPtr)                  \
  M(TermPtr)                  \
  M(PolynomialPtr)            \
  M(RoundOffPtr)              \
  M(MaxTermPtr)               \
  M(MinTermPtr)               \
  M(ReduceOpPtr)              \
  M(AtomicAddPtr)             \
  M(SyncThreadsPtr)           \
  M(ExternalCallPtr)          \
  M(ExternalCallWithAllocPtr)

namespace torch {
namespace jit {
namespace tensorexpr {

class TORCH_API IRVisitor {
 public:
  virtual ~IRVisitor() = default;

#define PRINT_VISIT(NamePtr) virtual void visit(NamePtr v);
  FORALL_IR_VISITORS(PRINT_VISIT)
#undef PRINT_VISIT

#define IMM_PRINT_VISIT(Type, Name) virtual void visit(Name##ImmPtr v);
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_PRINT_VISIT)
#undef IMM_PRINT_VISIT
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
